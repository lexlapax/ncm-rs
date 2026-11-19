//! Reference-order split-half rotary embedding for the XSum v2 profile.
use candle_core::cuda_backend::{
    CudaStorageSlice,
    cudarc::driver::{LaunchConfig, PushKernelArg},
};
use candle_core::{CpuStorage, CudaStorage, CustomOp3, Layout, Result, Shape, Tensor};

struct Rotary {
    function: candle_core::cuda_backend::cudarc::driver::CudaFunction,
}
impl CustomOp3 for Rotary {
    fn name(&self) -> &'static str {
        "xsum-reference-rope"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("XSum reference rotary embedding requires CUDA")
    }
    #[allow(unsafe_code)]
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        lx: &Layout,
        c: &CudaStorage,
        lc: &Layout,
        s: &CudaStorage,
        ls: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let [batch, heads, seq, width] = lx.dims() else {
            candle_core::bail!("invalid rotary rank")
        };
        if !(1..=128).contains(batch)
            // XSum and LM1B-small use 12 heads; LM1B-medium 16 and LM1B-large 20.
            || ![12, 16, 20].contains(heads)
            || !(1..=560).contains(seq)
            || *width != 64
            || lc.dims() != [*seq, 32]
            || ls.dims() != [*seq, 32]
        {
            candle_core::bail!("invalid XSum rotary dimensions")
        }
        let count = lx.shape().elem_count();
        let (x0, x1) = lx
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("noncontiguous rotary input".into()))?;
        let (c0, c1) = lc
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("noncontiguous cosine".into()))?;
        let (s0, s1) = ls
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("noncontiguous sine".into()))?;
        let xs = x.as_cuda_slice::<f32>()?;
        let cs = c.as_cuda_slice::<f32>()?;
        let ss = s.as_cuda_slice::<f32>()?;
        if x1 > xs.len()
            || c1 > cs.len()
            || s1 > ss.len()
            || x1 - x0 != count
            || c1 - c0 != seq * 32
            || s1 - s0 != seq * 32
        {
            candle_core::bail!("invalid rotary storage bounds")
        }
        let xs = xs.slice(x0..x1);
        let cs = cs.slice(c0..c1);
        let ss = ss.slice(s0..s1);
        let pairs = u32::try_from(count / 2).map_err(candle_core::Error::wrap)?;
        let seq = u32::try_from(*seq).map_err(candle_core::Error::wrap)?;
        let device = &x.device;
        // SAFETY: count is checked against all layouts; the kernel writes each output once.
        let mut out = unsafe { device.alloc::<f32>(count) }.map_err(candle_core::Error::wrap)?;
        let stream = device.cuda_stream();
        let mut launch = stream.launch_builder(&self.function);
        launch
            .arg(&xs)
            .arg(&cs)
            .arg(&ss)
            .arg(&mut out)
            .arg(&pairs)
            .arg(&seq);
        // SAFETY: device slices, extents, launch dimensions and shared stream were checked above.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (pairs.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(candle_core::Error::wrap)?;
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::F32(out),
                device: device.clone(),
            },
            lx.shape().clone(),
        ))
    }
}
pub fn apply(
    backend: &crate::CandleBackend,
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Tensor> {
    let device = backend.device().as_cuda_device()?;
    let function = device
        .get_or_load_custom_func(
            "ncm_xsum_reference_rope",
            "ncm-generator-layout-v1",
            crate::generator_layout::PTX,
        )?
        .into_cuda_function();
    x.contiguous()?
        .apply_op3_no_bwd(&cos.contiguous()?, &sin.contiguous()?, &Rotary { function })
}
