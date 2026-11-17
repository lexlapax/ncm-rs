//! Bounded WMT14 initial-cascade reference-order CUDA normalization.
use candle_core::cuda_backend::{
    CudaStorageSlice,
    cudarc::driver::{LaunchConfig, PushKernelArg},
};
use candle_core::{CpuStorage, CudaStorage, CustomOp1, Layout, Result, Shape, Tensor};
struct GroupNorm {
    function: candle_core::cuda_backend::cudarc::driver::CudaFunction,
}
impl CustomOp1 for GroupNorm {
    fn name(&self) -> &'static str {
        "wmt14-reference-groupnorm"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("WMT14 reference operation requires CUDA")
    }
    #[allow(unsafe_code)]
    fn cuda_fwd(&self, x: &CudaStorage, l: &Layout) -> Result<(CudaStorage, Shape)> {
        let dims = l.dims();
        if dims.len() != 3
            || dims[0] == 0
            || dims[0] > 32
            || dims[1] != 384
            || ![4, 16].contains(&dims[2])
        {
            candle_core::bail!("invalid WMT14 reference groupnorm shape")
        };
        let groups = (dims[0] * 8) as u32;
        let n = l.shape().elem_count();
        let (lo, hi) = l
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("noncontiguous WMT14 reference input".into()))?;
        let xs = x.as_cuda_slice::<f32>()?;
        if hi > xs.len() || hi - lo != n {
            candle_core::bail!("invalid WMT14 reference bounds")
        }
        let xs = xs.slice(lo..hi);
        let device = &x.device;
        // SAFETY: the checked element count is the exact kernel output extent.
        let mut out = unsafe { device.alloc::<f32>(n) }.map_err(candle_core::Error::wrap)?;
        let stream = device.cuda_stream();
        let mut launch = stream.launch_builder(&self.function);
        let width = (48 * dims[2]) as i32;
        launch.arg(&xs).arg(&mut out).arg(&width);
        // SAFETY: the layouts, device slices, bounds and launch dimensions were validated above.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (groups, 1, 1),
                block_dim: (if width < 512 { 32 } else { 512 }, 1, 1),
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
            "ncm_wmt14_groupnorm",
            "ncm-generator-layout-v1",
            crate::generator_layout::PTX,
        )?
        .into_cuda_function();
    x.transpose(1, 2)?
        .contiguous()?
        .apply_op1_no_bwd(&GroupNorm { function: f })?
        .transpose(1, 2)?
        .contiguous()
}
