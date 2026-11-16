//! XSum reference-order CUDA adapter for 64-wide RMS normalization.
use candle_core::cuda_backend::{
    CudaStorageSlice,
    cudarc::driver::{LaunchConfig, PushKernelArg},
};
use candle_core::{CpuStorage, CudaStorage, CustomOp2, Layout, Result, Shape, Tensor};

struct RmsNorm {
    function: candle_core::cuda_backend::cudarc::driver::CudaFunction,
}
impl CustomOp2 for RmsNorm {
    fn name(&self) -> &'static str {
        "xsum-reference-rmsnorm-64"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("XSum reference operation requires CUDA")
    }
    #[allow(unsafe_code)]
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        g: &CudaStorage,
        gl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        if xl.dims().last() != Some(&64) || gl.dims() != [64] {
            candle_core::bail!("invalid RMS dimensions")
        }
        let count = xl.shape().elem_count();
        if count == 0 || !count.is_multiple_of(64) || count / 64 > 128 * 12 * 560 {
            candle_core::bail!("invalid RMS row count")
        }
        let (xlo, xhi) = xl
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("x not contiguous".into()))?;
        let (glo, ghi) = gl
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("weight not contiguous".into()))?;
        let xs = x.as_cuda_slice::<f32>()?;
        let gs = g.as_cuda_slice::<f32>()?;
        if xhi > xs.len() || ghi > gs.len() || xhi - xlo != count || ghi - glo != 64 {
            candle_core::bail!("RMS storage bounds differ")
        }
        let xs = xs.slice(xlo..xhi);
        let gs = gs.slice(glo..ghi);
        let rows = (count / 64) as u32;
        let device = &x.device;
        // SAFETY: the checked element count is the exact kernel output extent.
        let mut output = unsafe { device.alloc::<f32>(count) }.map_err(candle_core::Error::wrap)?;
        let stream = device.cuda_stream();
        let mut launch = stream.launch_builder(&self.function);
        launch.arg(&xs).arg(&gs).arg(&mut output).arg(&rows);
        // SAFETY: the layouts, device slices, bounds and launch dimensions were validated above.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (rows.div_ceil(4), 1, 1),
                block_dim: (32, 4, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(candle_core::Error::wrap)?;
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::F32(output),
                device: device.clone(),
            },
            xl.shape().clone(),
        ))
    }
}
pub fn apply(backend: &crate::CandleBackend, x: &Tensor, g: &Tensor) -> Result<Tensor> {
    let device = backend.device().as_cuda_device()?;
    let function = device
        .get_or_load_custom_func(
            "ncm_pytorch_rms64",
            "ncm-generator-layout-v1",
            crate::generator_layout::PTX,
        )?
        .into_cuda_function();
    x.contiguous()?
        .apply_op2_no_bwd(&g.contiguous()?, &RmsNorm { function })
}
