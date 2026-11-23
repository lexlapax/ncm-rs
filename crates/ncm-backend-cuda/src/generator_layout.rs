//! Exact, bounded CUDA layout copies for 8- and 12-head × 64-wide generators.
use candle_core::cuda_backend::{
    CudaStorageSlice,
    cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg},
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

pub(crate) struct LayoutCopyU16 {
    pub(crate) kind: LayoutKind,
    pub(crate) heads: usize,
    pub(crate) head: usize,
    pub(crate) function: candle_core::cuda_backend::cudarc::driver::CudaFunction,
}

impl CustomOp1 for LayoutCopyU16 {
    fn name(&self) -> &'static str {
        match self.kind {
            LayoutKind::Pack => "ncm-pack-qkv-u16",
            LayoutKind::Merge => "ncm-merge-heads-u16",
        }
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("CUDA 16-bit layout operation cannot run on CPU")
    }
    #[allow(unsafe_code)]
    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        if self.heads == 0 || self.head == 0 {
            candle_core::bail!("invalid 16-bit layout head geometry")
        }
        let width = self
            .heads
            .checked_mul(self.head)
            .ok_or_else(|| candle_core::Error::Msg("layout width overflow".into()))?;
        let (batch, seq, count, shape, grid) = match self.kind {
            LayoutKind::Pack => {
                let [b, s, w] = layout.dims() else {
                    candle_core::bail!("invalid QKV rank")
                };
                if *w != 3 * width || *b == 0 || *s == 0 {
                    candle_core::bail!("invalid QKV dimensions")
                }
                let count = b
                    .checked_mul(*s)
                    .and_then(|n| n.checked_mul(*w))
                    .ok_or_else(|| candle_core::Error::Msg("QKV size overflow".into()))?;
                (
                    *b,
                    *s,
                    count,
                    vec![3, *b, self.heads, *s, self.head],
                    ((s * self.head).div_ceil(256), b * 3 * self.heads),
                )
            }
            LayoutKind::Merge => {
                let [b, heads, s, head] = layout.dims() else {
                    candle_core::bail!("invalid attention rank")
                };
                if *heads != self.heads || *head != self.head || *b == 0 || *s == 0 {
                    candle_core::bail!("invalid attention dimensions")
                }
                let count = b
                    .checked_mul(*s)
                    .and_then(|n| n.checked_mul(width))
                    .ok_or_else(|| candle_core::Error::Msg("attention size overflow".into()))?;
                (
                    *b,
                    *s,
                    count,
                    vec![*b, *s, width],
                    (width.div_ceil(256), b * s),
                )
            }
        };
        if grid.1 > 65535 {
            candle_core::bail!("CUDA 16-bit layout grid.y exceeds limit")
        }
        let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("CUDA 16-bit layout requires contiguous input".into())
        })?;
        if end.checked_sub(start) != Some(count) {
            candle_core::bail!("CUDA 16-bit layout storage bounds mismatch")
        }
        let batch_u32 = u32::try_from(batch).map_err(candle_core::Error::wrap)?;
        let seq_u32 = u32::try_from(seq).map_err(candle_core::Error::wrap)?;
        let heads_u32 = u32::try_from(self.heads).map_err(candle_core::Error::wrap)?;
        let head_u32 = u32::try_from(self.head).map_err(candle_core::Error::wrap)?;
        let launch = LaunchConfig {
            grid_dim: (grid.0 as u32, grid.1 as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let device = &storage.device;
        let stream = device.cuda_stream();
        struct LayoutU16Params {
            start: usize,
            end: usize,
            count: usize,
            dims: [u32; 4],
            launch: LaunchConfig,
        }
        #[allow(unsafe_code)]
        fn run_u16<T: DeviceRepr>(
            device: &candle_core::CudaDevice,
            stream: &candle_core::cuda_backend::cudarc::driver::CudaStream,
            function: &candle_core::cuda_backend::cudarc::driver::CudaFunction,
            src: &CudaSlice<T>,
            p: &LayoutU16Params,
        ) -> Result<CudaSlice<T>> {
            if p.end > src.len() {
                candle_core::bail!("CUDA 16-bit layout bounds mismatch")
            }
            let src_v = src.slice(p.start..p.end);
            // SAFETY: checked count; kernel writes every output element once.
            let mut dst =
                unsafe { device.alloc::<T>(p.count) }.map_err(candle_core::Error::wrap)?;
            let mut builder = stream.launch_builder(function);
            builder
                .arg(&src_v)
                .arg(&mut dst)
                .arg(&p.dims[0])
                .arg(&p.dims[1])
                .arg(&p.dims[2])
                .arg(&p.dims[3]);
            // SAFETY: all pointer extents, launch dimensions, and types validated.
            unsafe { builder.launch(p.launch) }.map_err(candle_core::Error::wrap)?;
            Ok(dst)
        }
        let params = LayoutU16Params {
            start,
            end,
            count,
            dims: [batch_u32, seq_u32, heads_u32, head_u32],
            launch,
        };
        let slice = match &storage.slice {
            CudaStorageSlice::BF16(src) => {
                CudaStorageSlice::BF16(run_u16(device, &stream, &self.function, src, &params)?)
            }
            CudaStorageSlice::F16(src) => {
                CudaStorageSlice::F16(run_u16(device, &stream, &self.function, src, &params)?)
            }
            _ => candle_core::bail!("CUDA 16-bit layout requires BF16 or F16 input"),
        };
        Ok((
            CudaStorage {
                slice,
                device: device.clone(),
            },
            shape.into(),
        ))
    }
}
