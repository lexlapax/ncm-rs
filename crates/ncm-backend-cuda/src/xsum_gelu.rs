//! XSum reference-order CUDA operation.
use candle_core::cuda_backend::{
    CudaStorageSlice,
    cudarc::driver::{LaunchConfig, PushKernelArg},
};
use candle_core::{CpuStorage, CudaStorage, CustomOp1, Layout, Result, Shape, Tensor};
struct Gelu {
    function: candle_core::cuda_backend::cudarc::driver::CudaFunction,
}
impl CustomOp1 for Gelu {
    fn name(&self) -> &'static str {
        "xsum-reference-gelu"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("XSum reference operation requires CUDA")
    }
    #[allow(unsafe_code)]
    fn cuda_fwd(&self, x: &CudaStorage, l: &Layout) -> Result<(CudaStorage, Shape)> {
        let n = l.shape().elem_count();
        if n == 0 || n > 128 * 560 * 3072 {
            candle_core::bail!("invalid XSum reference size")
        }
        let (lo, hi) = l
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("noncontiguous XSum reference input".into()))?;
        let xs = x.as_cuda_slice::<f32>()?;
        if hi > xs.len() || hi - lo != n {
            candle_core::bail!("invalid XSum reference bounds")
        }
        let xs = xs.slice(lo..hi);
        let device = &x.device;
        // SAFETY: the checked element count is the exact kernel output extent.
        let mut out = unsafe { device.alloc::<f32>(n) }.map_err(candle_core::Error::wrap)?;
        let stream = device.cuda_stream();
        let mut launch = stream.launch_builder(&self.function);
        let count = n as u32;
        launch.arg(&xs).arg(&mut out).arg(&count);
        // SAFETY: the layouts, device slices, bounds and launch dimensions were validated above.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (count.div_ceil(256), 1, 1),
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
            l.shape().clone(),
        ))
    }
}
pub fn apply(b: &crate::CandleBackend, x: &Tensor) -> Result<Tensor> {
    let d = b.device().as_cuda_device()?;
    let f = d
        .get_or_load_custom_func(
            "ncm_pytorch_gelu",
            "ncm-generator-layout-v1",
            crate::generator_layout::PTX,
        )?
        .into_cuda_function();
    x.contiguous()?.apply_op1_no_bwd(&Gelu { function: f })
}
