//! Explicit bounded LM1B interpolation diagnostic.
use candle_core::cuda_backend::{
    CudaStorageSlice,
    cudarc::driver::{LaunchConfig, PushKernelArg},
};
use candle_core::{CpuStorage, CudaStorage, CustomOp1, Layout, Result, Shape, Tensor};

struct Linear {
    function: candle_core::cuda_backend::cudarc::driver::CudaFunction,
}

impl CustomOp1 for Linear {
    fn name(&self) -> &'static str {
        "lm1b-linear-fma-diagnostic"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("LM1B linear diagnostic requires CUDA")
    }
    #[allow(unsafe_code)]
    fn cuda_fwd(&self, x: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        let (batch, length, channels) = layout.shape().dims3()?;
        if !(1..=32).contains(&batch)
            || channels != 384
            || ![8, 16, 24, 32, 40, 52, 64, 76, 92, 108].contains(&length)
        {
            candle_core::bail!("invalid LM1B interpolation shape")
        }
        let (lo, hi) = layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("noncontiguous LM1B interpolation input".into())
        })?;
        let xs = x.as_cuda_slice::<f32>()?;
        if hi > xs.len() || hi - lo != batch * length * channels {
            candle_core::bail!("invalid LM1B interpolation bounds")
        }
        let xs = xs.slice(lo..hi);
        let count = batch * 128 * channels;
        let device = &x.device;
        // SAFETY: shape validation establishes the exact output extent.
        let mut output = unsafe { device.alloc::<f32>(count) }.map_err(candle_core::Error::wrap)?;
        let stream = device.cuda_stream();
        let mut launch = stream.launch_builder(&self.function);
        let (batch, length, channels, count) =
            (batch as u32, length as u32, channels as u32, count as u32);
        let out_len = 128_u32;
        launch
            .arg(&xs)
            .arg(&mut output)
            .arg(&batch)
            .arg(&length)
            .arg(&out_len)
            .arg(&channels);
        // SAFETY: contiguous device slices and dimensions were checked above.
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
                slice: CudaStorageSlice::F32(output),
                device: device.clone(),
            },
            Shape::from((batch as usize, 128, channels as usize)),
        ))
    }
}

pub fn apply(backend: &crate::CandleBackend, x: &Tensor) -> Result<Tensor> {
    let device = backend.device().as_cuda_device()?;
    let function = device
        .get_or_load_custom_func(
            "ncm_lm1b_linear_fma",
            "ncm-generator-layout-v1",
            crate::generator_layout::PTX,
        )?
        .into_cuda_function();
    x.contiguous()?.apply_op1_no_bwd(&Linear { function })
}
