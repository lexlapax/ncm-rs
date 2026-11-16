//! XSum reference-order CUDA adapter for 768-wide layer normalization.
use candle_core::cuda_backend::{
    CudaStorageSlice,
    cudarc::driver::{LaunchConfig, PushKernelArg},
};
use candle_core::{CpuStorage, CudaStorage, CustomOp3, Layout, Result, Shape, Tensor};

struct LayerNorm {
    function: candle_core::cuda_backend::cudarc::driver::CudaFunction,
}

impl CustomOp3 for LayerNorm {
    fn name(&self) -> &'static str {
        "xsum-reference-layernorm-768"
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
        candle_core::bail!("XSum reference operation requires CUDA")
    }
    #[allow(unsafe_code)]
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        g: &CudaStorage,
        gl: &Layout,
        b: &CudaStorage,
        bl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dims = xl.dims();
        if dims.last() != Some(&768) || gl.dims() != [768] || bl.dims() != [768] {
            candle_core::bail!("invalid XSum reference layer norm dimensions")
        }
        let count = xl.shape().elem_count();
        if count == 0 || !count.is_multiple_of(768) || count / 768 > 128 * 560 {
            candle_core::bail!("invalid XSum reference row count")
        }
        let (xlo, xhi) = xl
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("x not contiguous".into()))?;
        let (glo, ghi) = gl
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("gamma not contiguous".into()))?;
        let (blo, bhi) = bl
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("beta not contiguous".into()))?;
        let xs = x.as_cuda_slice::<f32>()?;
        let gs = g.as_cuda_slice::<f32>()?;
        let bs = b.as_cuda_slice::<f32>()?;
        if xhi > xs.len()
            || ghi > gs.len()
            || bhi > bs.len()
            || xhi - xlo != count
            || ghi - glo != 768
            || bhi - blo != 768
        {
            candle_core::bail!("XSum reference storage bounds differ")
        }
        let xs = xs.slice(xlo..xhi);
        let gs = gs.slice(glo..ghi);
        let bs = bs.slice(blo..bhi);
        let device = &x.device;
        // SAFETY: the checked element count is the exact kernel output extent.
        let mut output = unsafe { device.alloc::<f32>(count) }.map_err(candle_core::Error::wrap)?;
        let stream = device.cuda_stream();
        let mut launch = stream.launch_builder(&self.function);
        launch.arg(&xs).arg(&gs).arg(&bs).arg(&mut output);
        // SAFETY: the layouts, device slices, bounds and launch dimensions were validated above.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: ((count / 768) as u32, 1, 1),
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

pub fn apply(backend: &crate::CandleBackend, x: &Tensor, g: &Tensor, b: &Tensor) -> Result<Tensor> {
    let device = backend.device().as_cuda_device()?;
    let function = device
        .get_or_load_custom_func(
            "ncm_pytorch_layernorm_768",
            "ncm-generator-layout-v1",
            crate::generator_layout::PTX,
        )?
        .into_cuda_function();
    x.contiguous()?
        .apply_op3_no_bwd(&g.contiguous()?, &b.contiguous()?, &LayerNorm { function })
}
