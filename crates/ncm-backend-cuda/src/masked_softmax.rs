//! CUDA attention scale, additive-mask and warp softmax fusion.
use candle_core::cuda_backend::{
    CudaStorageSlice,
    cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg},
};
use candle_core::{CpuStorage, CudaStorage, CustomOp2, Layout, Result, Shape, Tensor};

struct MaskedSoftmax {
    function: candle_core::cuda_backend::cudarc::driver::CudaFunction,
    warps_per_block: u32,
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
        let grid_dim = if self.warps_per_block == 4 {
            ((h * s).div_ceil(self.warps_per_block), *batch as u32, 1)
        } else {
            ((*batch as u32 * h * s), 1, 1)
        };
        // SAFETY: one 32-thread warp per checked score row; all arguments and
        // memory extents match the CUDA kernel, with cudarc event tracking.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim,
                block_dim: (32 * self.warps_per_block, 1, 1),
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
    let (name, warps_per_block) = match scores.dims() {
        [_, _, 512..=544, _] => ("ncm_masked_softmax_64_registers_17", 4),
        [_, _, 545..=560, _] => ("ncm_masked_softmax_64_registers_18", 4),
        _ => ("ncm_masked_softmax_64", 1),
    };
    let function = device
        .get_or_load_custom_func(
            name,
            "ncm-generator-layout-v1",
            crate::generator_layout::PTX,
        )?
        .into_cuda_function();
    scores.contiguous()?.apply_op2_no_bwd(
        &mask.contiguous()?,
        &MaskedSoftmax {
            function,
            warps_per_block,
        },
    )
}

struct FusedSoftmax16 {
    function: candle_core::cuda_backend::cudarc::driver::CudaFunction,
    scale: f32,
    has_mask: bool,
}

impl CustomOp2 for FusedSoftmax16 {
    fn name(&self) -> &'static str {
        "ncm-fused-softmax-16"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("CUDA fused low-precision softmax cannot run on CPU")
    }
    #[allow(unsafe_code)]
    fn cuda_fwd(
        &self,
        scores: &CudaStorage,
        layout: &Layout,
        mask: &CudaStorage,
        mask_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let [batch, heads, q_len, k_len] = layout.dims() else {
            candle_core::bail!("attention scores must have rank 4")
        };
        if *batch == 0 || *heads == 0 || *q_len == 0 || *k_len == 0 {
            candle_core::bail!("attention scores must be non-empty")
        }
        let count = batch
            .checked_mul(*heads)
            .and_then(|n| n.checked_mul(*q_len))
            .and_then(|n| n.checked_mul(*k_len))
            .ok_or_else(|| candle_core::Error::Msg("attention score size overflow".into()))?;
        let total_rows = batch
            .checked_mul(*heads)
            .and_then(|n| n.checked_mul(*q_len))
            .ok_or_else(|| candle_core::Error::Msg("attention row count overflow".into()))?;
        let (mask_b, mask_q, mask_count) = if self.has_mask {
            let [mb, mh, mq, mk] = mask_layout.dims() else {
                candle_core::bail!("attention mask must have rank 4")
            };
            if !(*mb == 1 || mb == batch) || *mh != 1 || !(*mq == 1 || mq == q_len) || mk != k_len {
                candle_core::bail!("unsupported low-precision attention mask shape")
            }
            (*mb, *mq, mb * mq * mk)
        } else {
            (1usize, 1usize, 1usize)
        };
        let (lo, hi) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("scores must be contiguous".into()))?;
        let (mlo, mhi) = mask_layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("mask must be contiguous".into()))?;
        if hi.checked_sub(lo) != Some(count)
            || (self.has_mask && mhi.checked_sub(mlo) != Some(mask_count))
        {
            candle_core::bail!("fused softmax storage bounds mismatch")
        }
        let total_rows_u32 = u32::try_from(total_rows).map_err(candle_core::Error::wrap)?;
        let heads_u32 = u32::try_from(*heads).map_err(candle_core::Error::wrap)?;
        let q_len_u32 = u32::try_from(*q_len).map_err(candle_core::Error::wrap)?;
        let k_len_u32 = u32::try_from(*k_len).map_err(candle_core::Error::wrap)?;
        let mask_b_u32 = u32::try_from(mask_b).map_err(candle_core::Error::wrap)?;
        let mask_q_u32 = u32::try_from(mask_q).map_err(candle_core::Error::wrap)?;
        let has_mask_u32 = u32::from(self.has_mask);
        let grid_dim = (total_rows_u32.div_ceil(4), 1, 1);
        let cfg = LaunchConfig {
            grid_dim,
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let device = &scores.device;
        let stream = device.cuda_stream();
        struct FusedParams {
            lo: usize,
            hi: usize,
            mlo: usize,
            mhi: usize,
            count: usize,
            has_mask: bool,
            scale: f32,
            dims: [u32; 7],
            cfg: LaunchConfig,
        }
        #[allow(unsafe_code)]
        fn run_fused<T: DeviceRepr>(
            device: &candle_core::CudaDevice,
            stream: &candle_core::cuda_backend::cudarc::driver::CudaStream,
            function: &candle_core::cuda_backend::cudarc::driver::CudaFunction,
            src: &CudaSlice<T>,
            msk: &CudaSlice<T>,
            p: &FusedParams,
        ) -> Result<CudaSlice<T>> {
            if p.hi > src.len() || (p.has_mask && p.mhi > msk.len()) {
                candle_core::bail!("fused softmax storage out of bounds")
            }
            let src_v = src.slice(p.lo..p.hi);
            let msk_v = msk.slice(p.mlo..p.mhi);
            // SAFETY: checked shapes bound all reads and each warp writes disjoint rows.
            let mut out =
                unsafe { device.alloc::<T>(p.count) }.map_err(candle_core::Error::wrap)?;
            let mut launch = stream.launch_builder(function);
            launch
                .arg(&src_v)
                .arg(&msk_v)
                .arg(&mut out)
                .arg(&p.scale)
                .arg(&p.dims[0])
                .arg(&p.dims[1])
                .arg(&p.dims[2])
                .arg(&p.dims[3])
                .arg(&p.dims[4])
                .arg(&p.dims[5])
                .arg(&p.dims[6]);
            // SAFETY: validated device pointers, extents, and launch dimensions.
            unsafe { launch.launch(p.cfg) }.map_err(candle_core::Error::wrap)?;
            Ok(out)
        }
        let params = FusedParams {
            lo,
            hi,
            mlo,
            mhi,
            count,
            has_mask: self.has_mask,
            scale: self.scale,
            dims: [
                total_rows_u32,
                heads_u32,
                q_len_u32,
                k_len_u32,
                mask_b_u32,
                mask_q_u32,
                has_mask_u32,
            ],
            cfg,
        };
        let slice = match (&scores.slice, &mask.slice) {
            (CudaStorageSlice::BF16(src), CudaStorageSlice::BF16(msk)) => CudaStorageSlice::BF16(
                run_fused(device, &stream, &self.function, src, msk, &params)?,
            ),
            (CudaStorageSlice::F16(src), CudaStorageSlice::F16(msk)) => CudaStorageSlice::F16(
                run_fused(device, &stream, &self.function, src, msk, &params)?,
            ),
            _ => candle_core::bail!(
                "fused low-precision softmax requires matching BF16 or F16 tensors"
            ),
        };
        Ok((
            CudaStorage {
                slice,
                device: device.clone(),
            },
            layout.shape().clone(),
        ))
    }
}

pub(crate) fn apply_low_precision(
    backend: &crate::CandleBackend,
    scores: &Tensor,
    mask: Option<&Tensor>,
    scale: f32,
) -> Result<Tensor> {
    if !backend.device().is_cuda() || !scores.device().same_device(backend.device()) {
        candle_core::bail!("explicit matching CUDA device required")
    }
    let dtype = scores.dtype();
    let kernel_name = match dtype {
        candle_core::DType::BF16 => "ncm_fused_softmax_bf16",
        candle_core::DType::F16 => "ncm_fused_softmax_f16",
        other => candle_core::bail!("unsupported dtype {other:?} for fused low-precision softmax"),
    };
    let device = backend.device().as_cuda_device()?;
    let function = device
        .get_or_load_custom_func(
            kernel_name,
            "ncm-generator-layout-v1",
            crate::generator_layout::PTX,
        )?
        .into_cuda_function();
    let dummy;
    let (mask_tensor, has_mask) = match mask {
        Some(m) => {
            if !m.device().same_device(backend.device()) {
                candle_core::bail!("mask device mismatch")
            }
            (m.contiguous()?, true)
        }
        None => {
            dummy = backend.zero_bias(1, dtype)?;
            (dummy, false)
        }
    };
    scores.contiguous()?.apply_op2_no_bwd(
        &mask_tensor,
        &FusedSoftmax16 {
            function,
            scale,
            has_mask,
        },
    )
}
