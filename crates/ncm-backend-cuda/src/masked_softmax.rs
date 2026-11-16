//! CUDA attention scale, additive-mask and warp softmax fusion.
use candle_core::cuda_backend::{
    CudaStorageSlice,
    cudarc::driver::{LaunchConfig, PushKernelArg},
};
use candle_core::{CpuStorage, CudaStorage, CustomOp2, Layout, Result, Shape, Tensor};

struct MaskedSoftmax {
    function: candle_core::cuda_backend::cudarc::driver::CudaFunction,
}
impl CustomOp2 for MaskedSoftmax {
    fn name(&self) -> &'static str {
        "ncm-masked-softmax-64"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("CUDA masked softmax cannot run on CPU")
    }
    #[allow(unsafe_code)]
    fn cuda_fwd(
        &self,
        scores: &CudaStorage,
        layout: &Layout,
        mask: &CudaStorage,
        mask_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let [batch, heads, seq, cols] = layout.dims() else {
            candle_core::bail!("attention scores must have rank 4")
        };
        let [mask_batch, mask_heads, mask_rows, mask_cols] = mask_layout.dims() else {
            candle_core::bail!("attention mask must have rank 4")
        };
        if !(1..=128).contains(batch)
            || !(1..=16).contains(heads)
            || !(512..=768).contains(seq)
            || cols != seq
            || !(*mask_batch == 1 || mask_batch == batch)
            || *mask_heads != 1
            || mask_rows != seq
            || mask_cols != seq
        {
            candle_core::bail!("unsupported masked softmax shape")
        }
        let count = batch * heads * seq * seq;
        let mask_count = mask_batch * seq * seq;
        let (lo, hi) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("scores must be contiguous".into()))?;
        let (mlo, mhi) = mask_layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("mask must be contiguous".into()))?;
        let input = scores.as_cuda_slice::<f32>()?;
        let mask = mask.as_cuda_slice::<f32>()?;
        if hi.checked_sub(lo) != Some(count)
            || hi > input.len()
            || mhi.checked_sub(mlo) != Some(mask_count)
            || mhi > mask.len()
        {
            candle_core::bail!("masked softmax storage bounds mismatch")
        }
        let input = input.slice(lo..hi);
        let mask = mask.slice(mlo..mhi);
        let device = &scores.device;
        // SAFETY: checked shapes bound all reads and each warp writes exactly
        // one complete, disjoint row of the initialized result.
        let mut output = unsafe { device.alloc::<f32>(count) }.map_err(candle_core::Error::wrap)?;
        let (h, s, mb) = (*heads as u32, *seq as u32, *mask_batch as u32);
        let stream = device.cuda_stream();
        let mut launch = stream.launch_builder(&self.function);
        launch
            .arg(&input)
            .arg(&mask)
            .arg(&mut output)
            .arg(&h)
            .arg(&s)
            .arg(&mb);
        // SAFETY: one 32-thread warp per checked score row; all arguments and
        // memory extents match the CUDA kernel, with cudarc event tracking.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: ((batch * heads * seq) as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(candle_core::Error::wrap)?;
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::F32(output),
                device: device.clone(),
            },
            layout.shape().clone(),
        ))
    }
}
pub fn apply(backend: &crate::CandleBackend, scores: &Tensor, mask: &Tensor) -> Result<Tensor> {
    if !backend.device().is_cuda()
        || !scores.device().same_device(backend.device())
        || !mask.device().same_device(backend.device())
    {
        candle_core::bail!("explicit matching CUDA device required")
    }
    let device = backend.device().as_cuda_device()?;
    let function = device
        .get_or_load_custom_func(
            "ncm_masked_softmax_64",
            "ncm-generator-layout-v1",
            crate::generator_layout::PTX,
        )?
        .into_cuda_function();
    scores
        .contiguous()?
        .apply_op2_no_bwd(&mask.contiguous()?, &MaskedSoftmax { function })
}
