//! Finite 4096-way maxima for the restricted XSum top-1 paths.
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
        "ncm-top1-4096"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("CUDA unique top-1 operation cannot run on CPU")
    }
    #[allow(unsafe_code)]
    fn cuda_fwd(&self, scores: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        let [batch, length, vocab] = layout.dims() else {
            candle_core::bail!("top-1 logits must be rank 3")
        };
        if !(1..=128).contains(batch) || !(1..=128).contains(length) || *vocab != 4096 {
            candle_core::bail!("unsupported top-1 shape")
        }
        let rows = batch * length;
        let (lo, hi) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("scores must be contiguous".into()))?;
        let source = scores.as_cuda_slice::<f32>()?;
        if hi.checked_sub(lo) != Some(rows * 4096) || hi > source.len() {
            candle_core::bail!("top-1 storage bounds mismatch")
        }
        let source = source.slice(lo..hi);
        let device = &scores.device;
        // SAFETY: shapes and storage lengths are checked; each block writes
        // exactly one result, including a sentinel for invalid rows.
        let mut output = unsafe { device.alloc::<u32>(rows) }.map_err(candle_core::Error::wrap)?;
        let stream = device.cuda_stream();
        let mut launch = stream.launch_builder(&self.function);
        launch.arg(&source);
        launch.arg(&mut output);
        // SAFETY: checked extents and typed arguments match the kernel, with
        // one 256-thread block per row and cudarc stream event tracking.
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

pub fn unique_max(backend: &crate::CandleBackend, logits: &Tensor) -> Result<Tensor> {
    select(backend, logits, "ncm_unique_max_4096")
}

pub fn xsum_reference_top1(backend: &crate::CandleBackend, logits: &Tensor) -> Result<Tensor> {
    select(backend, logits, "ncm_xsum_reference_top1_4096")
}

fn select(backend: &crate::CandleBackend, logits: &Tensor, kernel: &'static str) -> Result<Tensor> {
    if !backend.device().is_cuda() || !logits.device().same_device(backend.device()) {
        candle_core::bail!("explicit matching CUDA device required")
    }
    let device = backend.device().as_cuda_device()?;
    let function = device
        .get_or_load_custom_func(
            kernel,
            "ncm-generator-layout-v1",
            crate::generator_layout::PTX,
        )?
        .into_cuda_function();
    logits.contiguous()?.apply_op1_no_bwd(&Select { function })
}
