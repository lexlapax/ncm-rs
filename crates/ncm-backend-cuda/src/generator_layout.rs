//! Exact, bounded CUDA layout copies for 8- and 12-head × 64-wide generators.
use candle_core::cuda_backend::{
    CudaStorageSlice,
    cudarc::driver::{LaunchConfig, PushKernelArg},
};
use candle_core::{CpuStorage, CudaStorage, CustomOp1, Layout, Result, Shape};

pub(crate) const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/generator_layout.ptx"));
#[derive(Clone, Copy)]
pub(crate) enum LayoutKind {
    Pack,
    Merge,
}
pub(crate) struct LayoutCopy {
    pub(crate) kind: LayoutKind,
    pub(crate) heads: usize,
    pub(crate) function: candle_core::cuda_backend::cudarc::driver::CudaFunction,
}
impl CustomOp1 for LayoutCopy {
    fn name(&self) -> &'static str {
        match self.kind {
            LayoutKind::Pack => "ncm-pack-qkv",
            LayoutKind::Merge => "ncm-merge-heads",
        }
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("CUDA layout operation cannot run on CPU")
    }
    #[allow(unsafe_code)]
    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        if !matches!(self.heads, 8 | 12) {
            candle_core::bail!("unsupported layout head count")
        }
        let (batch, seq, count, shape, grid) = match self.kind {
            LayoutKind::Pack => {
                let [b, s, width] = layout.dims() else {
                    candle_core::bail!("invalid QKV rank")
                };
                if *width != 3 * self.heads * 64 || !(1..=128).contains(b) || !(1..=768).contains(s)
                {
                    candle_core::bail!("invalid QKV dimensions")
                }
                let count = b
                    .checked_mul(*s)
                    .and_then(|n| n.checked_mul(*width))
                    .ok_or_else(|| candle_core::Error::Msg("QKV size overflow".into()))?;
                (
                    *b,
                    *s,
                    count,
                    vec![3, *b, self.heads, *s, 64],
                    ((s * 64).div_ceil(256), b * 3 * self.heads),
                )
            }
            LayoutKind::Merge => {
                let [b, heads, s, width] = layout.dims() else {
                    candle_core::bail!("invalid attention rank")
                };
                if *heads != self.heads
                    || *width != 64
                    || !(1..=128).contains(b)
                    || !(1..=768).contains(s)
                {
                    candle_core::bail!("invalid attention dimensions")
                }
                let count = b
                    .checked_mul(*s)
                    .and_then(|n| n.checked_mul(self.heads * 64))
                    .ok_or_else(|| candle_core::Error::Msg("attention size overflow".into()))?;
                (
                    *b,
                    *s,
                    count,
                    vec![*b, *s, self.heads * 64],
                    ((self.heads * 64).div_ceil(256), b * s),
                )
            }
        };
        if !(1..=128).contains(&batch) || !(1..=768).contains(&seq) || grid.1 > 65535 {
            candle_core::bail!("CUDA layout dimensions exceed validated bounds")
        }
        let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("CUDA layout requires contiguous input".into())
        })?;
        let src = storage.as_cuda_slice::<f32>()?;
        if end.checked_sub(start) != Some(count) || end > src.len() {
            candle_core::bail!("CUDA layout storage bounds mismatch")
        }
        let src = src.slice(start..end);
        let device = &storage.device;
        // SAFETY: count was checked above; the kernel writes exactly one output
        // value per element before any consumer can read it on the same stream.
        let mut dst = unsafe { device.alloc::<f32>(count) }.map_err(candle_core::Error::wrap)?;
        let batch = u32::try_from(batch).map_err(candle_core::Error::wrap)?;
        let seq = u32::try_from(seq).map_err(candle_core::Error::wrap)?;
        let launch = LaunchConfig {
            grid_dim: (grid.0 as u32, grid.1 as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = device.cuda_stream();
        let mut builder = stream.launch_builder(&self.function);
        builder.arg(&src);
        builder.arg(&mut dst);
        builder.arg(&batch);
        builder.arg(&seq);
        // SAFETY: all pointer extents, launch dimensions, dtype and layout were
        // checked; cudarc's launch arguments retain allocation/event tracking.
        unsafe { builder.launch(launch) }.map_err(candle_core::Error::wrap)?;
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::F32(dst),
                device: device.clone(),
            },
            shape.into(),
        ))
    }
}
