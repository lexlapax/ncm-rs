//! Experimental inverse-CDF sampling of 4096-way rows with supplied uniforms.
use candle_core::cuda_backend::{
    CudaStorageSlice,
    cudarc::driver::{LaunchConfig, PushKernelArg},
};
use candle_core::{CpuStorage, CudaStorage, CustomOp2, Layout, Result, Shape, Tensor};

struct Draw {
    function: candle_core::cuda_backend::cudarc::driver::CudaFunction,
}
impl CustomOp2 for Draw {
    fn name(&self) -> &'static str {
        "ncm-categorical-4096"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("CUDA categorical operation cannot run on CPU")
    }
    #[allow(unsafe_code)]
    fn cuda_fwd(
        &self,
        scores: &CudaStorage,
        score_layout: &Layout,
        uniforms: &CudaStorage,
        uniform_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let [batch, length, vocab] = score_layout.dims() else {
            candle_core::bail!("categorical logits must be rank 3")
        };
        if !(1..=128).contains(batch)
            || !(1..=128).contains(length)
            || *vocab != 4096
            || uniform_layout.dims() != [*batch, *length]
        {
            candle_core::bail!("unsupported categorical shape")
        }
        let rows = batch * length;
        let (lo, hi) = score_layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("scores must be contiguous".into()))?;
        let (ulo, uhi) = uniform_layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("uniforms must be contiguous".into()))?;
        let source = scores.as_cuda_slice::<f32>()?;
        let random = uniforms.as_cuda_slice::<u32>()?;
        if hi.checked_sub(lo) != Some(rows * 4096)
            || hi > source.len()
            || uhi.checked_sub(ulo) != Some(rows)
            || uhi > random.len()
        {
            candle_core::bail!("categorical storage bounds mismatch")
        }
        let source = source.slice(lo..hi);
        let random = random.slice(ulo..uhi);
        let device = &scores.device;
        // SAFETY: shapes and storage lengths are checked above; each block
        // writes exactly one output row. Invalid scores produce the sentinel.
        let mut output = unsafe { device.alloc::<u32>(rows) }.map_err(candle_core::Error::wrap)?;
        let stream = device.cuda_stream();
        let mut launch = stream.launch_builder(&self.function);
        launch.arg(&source);
        launch.arg(&random);
        launch.arg(&mut output);
        // SAFETY: argument types, extents and one 256-thread block per row are
        // checked above; cudarc retains allocation and stream event tracking.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(candle_core::Error::wrap)?;
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::U32(output),
                device: device.clone(),
            },
            vec![*batch, *length].into(),
        ))
    }
}

/// Diagnostic only: samples each 4096-way FP32 row using one supplied u32,
/// interpreted as a 32-bit uniform in [0,1). NaN, +infinity or an all-negative-
/// infinity row returns `u32::MAX`; callers must check this sentinel before use.
/// The operation does not match the PyTorch PRNG stream or support top-k/top-p.
pub fn categorical_from_uniforms(
    backend: &crate::CandleBackend,
    logits: &Tensor,
    uniforms: &Tensor,
) -> Result<Tensor> {
    if !backend.device().is_cuda()
        || !logits.device().same_device(backend.device())
        || !uniforms.device().same_device(backend.device())
    {
        candle_core::bail!("explicit matching CUDA device required")
    }
    let device = backend.device().as_cuda_device()?;
    let function = device
        .get_or_load_custom_func(
            "ncm_categorical_4096",
            "ncm-generator-layout-v1",
            crate::generator_layout::PTX,
        )?
        .into_cuda_function();
    logits
        .contiguous()?
        .apply_op2_no_bwd(&uniforms.contiguous()?, &Draw { function })
}
