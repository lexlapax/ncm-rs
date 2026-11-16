//! Explicit CUDA FP64 implementation of the pinned XSum token penalties.
use candle_core::cuda_backend::{
    CudaStorageSlice,
    cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg},
};
use candle_core::{CpuStorage, CudaStorage, CustomOp2, Layout, Result, Shape, Tensor};

struct Select {
    init: CudaFunction,
    step: CudaFunction,
    finish: CudaFunction,
}
impl CustomOp2 for Select {
    fn name(&self) -> &'static str {
        "ncm-xsum-penalty-fp64"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("CUDA FP64 XSum selection cannot run on CPU")
    }
    #[allow(unsafe_code)]
    fn cuda_fwd(
        &self,
        logits: &CudaStorage,
        layout: &Layout,
        allowance: &CudaStorage,
        allow_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let [batch, time, vocab] = layout.dims() else {
            candle_core::bail!("XSum logits must be rank 3")
        };
        if !(1..=128).contains(batch)
            || !(1..=512).contains(time)
            || !(1..=50304).contains(vocab)
            || allow_layout.dims() != [*vocab]
        {
            candle_core::bail!("unsupported XSum penalty shape")
        }
        let (lo, hi) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("XSum logits must be contiguous".into()))?;
        let (alo, ahi) = allow_layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("XSum allowance must be contiguous".into()))?;
        let input = logits.as_cuda_slice::<f32>()?;
        let allow = allowance.as_cuda_slice::<f32>()?;
        if hi.checked_sub(lo) != Some(batch * time * vocab)
            || hi > input.len()
            || ahi.checked_sub(alo) != Some(*vocab)
            || ahi > allow.len()
        {
            candle_core::bail!("XSum penalty storage bounds mismatch")
        }
        let input = input.slice(lo..hi);
        let allow = allow.slice(alo..ahi);
        let device = &logits.device;
        let tiles = vocab.div_ceil(256);
        let (b, t, v, k) = (*batch as u32, *time as u32, *vocab as u32, tiles as u32);
        // SAFETY: validated dimensions bound every allocation and kernel access;
        // init fills counts and flags, and each step fills every partial/result.
        let (counts, bad, partial, partial_id, output) = unsafe {
            (
                device.alloc::<u32>(batch * vocab),
                device.alloc::<u32>(*batch),
                device.alloc::<f64>(batch * tiles),
                device.alloc::<u32>(batch * tiles),
                device.alloc::<u32>(batch * time),
            )
        };
        let mut counts = counts.map_err(candle_core::Error::wrap)?;
        let mut bad = bad.map_err(candle_core::Error::wrap)?;
        let mut partial = partial.map_err(candle_core::Error::wrap)?;
        let mut partial_id = partial_id.map_err(candle_core::Error::wrap)?;
        let mut output = output.map_err(candle_core::Error::wrap)?;
        let stream = device.cuda_stream();
        {
            let mut launch = stream.launch_builder(&self.init);
            launch.arg(&mut counts).arg(&mut bad).arg(&b).arg(&v);
            // SAFETY: init covers the checked batch*vocab range and flag array.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: ((b * v).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(candle_core::Error::wrap)?;
        }
        for position in 0..t {
            {
                let mut launch = stream.launch_builder(&self.step);
                launch
                    .arg(&input)
                    .arg(&allow)
                    .arg(&counts)
                    .arg(&output)
                    .arg(&mut partial)
                    .arg(&mut partial_id)
                    .arg(&mut bad)
                    .arg(&v)
                    .arg(&t)
                    .arg(&position)
                    .arg(&k);
                // SAFETY: the grid covers checked row/tile bounds; each tile
                // writes one pair and only reads earlier completed positions.
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (k, b, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(candle_core::Error::wrap)?;
            }
            {
                let mut launch = stream.launch_builder(&self.finish);
                launch
                    .arg(&partial)
                    .arg(&partial_id)
                    .arg(&mut counts)
                    .arg(&mut output)
                    .arg(&mut bad)
                    .arg(&v)
                    .arg(&t)
                    .arg(&position)
                    .arg(&k);
                // SAFETY: one block per row reduces the checked partial array
                // and writes one position and at most one bounded count.
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (b, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(candle_core::Error::wrap)?;
            }
        }
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::U32(output),
                device: device.clone(),
            },
            vec![*batch, *time].into(),
        ))
    }
}

pub fn select(
    backend: &crate::CandleBackend,
    logits: &Tensor,
    allowance: &Tensor,
) -> Result<Tensor> {
    if !backend.device().is_cuda()
        || !logits.device().same_device(backend.device())
        || !allowance.device().same_device(backend.device())
    {
        candle_core::bail!("explicit matching CUDA device required")
    }
    let device = backend.device().as_cuda_device()?;
    let load = |name| -> Result<CudaFunction> {
        Ok(device
            .get_or_load_custom_func(
                name,
                "ncm-generator-layout-v1",
                crate::generator_layout::PTX,
            )?
            .into_cuda_function())
    };
    logits.contiguous()?.apply_op2_no_bwd(
        &allowance.contiguous()?,
        &Select {
            init: load("ncm_xsum_penalty_init")?,
            step: load("ncm_xsum_penalty_step")?,
            finish: load("ncm_xsum_penalty_finish")?,
        },
    )
}
