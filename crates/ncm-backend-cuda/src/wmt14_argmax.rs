//! Explicit WMT14 diagnostic first-maximum selection without full-size temporaries.
use candle_core::cuda_backend::{
    CudaStorageSlice,
    cudarc::driver::{LaunchConfig, PushKernelArg},
};
use candle_core::{CpuStorage, CudaStorage, CustomOp1, Layout, Result, Shape, Tensor};

struct Select {
    function: candle_core::cuda_backend::cudarc::driver::CudaFunction,
}
impl CustomOp1 for Select {
    fn name(&self) -> &'static str {
        "ncm-wmt14-first-max-151680"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("WMT14 diagnostic selection requires CUDA")
    }
    #[allow(unsafe_code)]
    fn cuda_fwd(&self, scores: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        let [batch, length, vocab] = layout.dims() else {
            candle_core::bail!("WMT14 logits must have rank three")
        };
        if !(1..=32).contains(batch) || *length != 64 || *vocab != 151680 {
            candle_core::bail!("unsupported WMT14 selection shape")
        }
        let rows = batch * length;
        let (lo, hi) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("scores must be contiguous".into()))?;
        let source = scores.as_cuda_slice::<f32>()?;
        if hi.checked_sub(lo) != Some(rows * 151680) || hi > source.len() {
            candle_core::bail!("WMT14 selection storage bounds mismatch")
        }
        let source = source.slice(lo..hi);
        let device = &scores.device;
        // SAFETY: checked extents bound every input read and each block writes one result.
        let mut output = unsafe { device.alloc::<u32>(rows) }.map_err(candle_core::Error::wrap)?;
        let stream = device.cuda_stream();
        let mut launch = stream.launch_builder(&self.function);
        launch.arg(&source).arg(&mut output);
        // SAFETY: one 256-thread block per checked row; typed arguments match the kernel.
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

/// Run the explicitly selected diagnostic; unsupported shapes fail without fallback.
pub fn select(backend: &crate::CandleBackend, logits: &Tensor) -> Result<Tensor> {
    if !backend.device().is_cuda() || !logits.device().same_device(backend.device()) {
        candle_core::bail!("explicit matching CUDA device required")
    }
    let device = backend.device().as_cuda_device()?;
    let function = device
        .get_or_load_custom_func(
            "ncm_wmt14_first_max_151680",
            "ncm-generator-layout-v1",
            crate::generator_layout::PTX,
        )?
        .into_cuda_function();
    logits.contiguous()?.apply_op1_no_bwd(&Select { function })
}
