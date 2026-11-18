//! Bounded LM1B-small attention experiment for Ampere CUDA.
use candle_core::{
    CpuStorage, CudaStorage, CustomOp3, Layout, Result, Shape, Storage, Tensor,
    cuda_backend::{
        CudaStorageSlice,
        cudarc::driver::{CudaFunction, CudaView, LaunchConfig, PushKernelArg},
    },
};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/lm1b_attention.ptx"));

struct Attention {
    function: CudaFunction,
    mask: Option<Tensor>,
}

fn input_slice<'a>(
    storage: &'a CudaStorage,
    layout: &Layout,
    count: usize,
) -> Result<CudaView<'a, f32>> {
    let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("LM1B attention requires contiguous input".into())
    })?;
    let input = storage.as_cuda_slice::<f32>()?;
    if end.checked_sub(start) != Some(count) || end > input.len() {
        candle_core::bail!("LM1B attention storage extent differs");
    }
    Ok(input.slice(start..end))
}

impl CustomOp3 for Attention {
    fn name(&self) -> &'static str {
        "lm1b-efficient-attention"
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
        candle_core::bail!("LM1B efficient attention requires Ampere CUDA")
    }

    #[allow(unsafe_code)]
    fn cuda_fwd(
        &self,
        q: &CudaStorage,
        ql: &Layout,
        k: &CudaStorage,
        kl: &Layout,
        v: &CudaStorage,
        vl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (batch, heads, width, depth) = ql.shape().dims4()?;
        if !(1..=32).contains(&batch)
            || ![6, 12].contains(&heads)
            || !(1..=129).contains(&width)
            || depth != 64
            || kl.shape() != ql.shape()
            || vl.shape() != ql.shape()
        {
            candle_core::bail!("unsupported LM1B efficient attention geometry");
        }
        let count = batch * heads * width * depth;
        let query = input_slice(q, ql, count)?;
        let key = input_slice(k, kl, count)?;
        let value = input_slice(v, vl, count)?;
        let mask_storage = self.mask.as_ref().map(Tensor::storage_and_layout);
        let (mask, mask_batches) = match &mask_storage {
            Some((storage, layout)) => {
                let Storage::Cuda(storage) = &**storage else {
                    candle_core::bail!("LM1B efficient attention mask must be on CUDA");
                };
                let (n, h, rows, columns) = layout.shape().dims4()?;
                if !(n == 1 || n == batch) || h != 1 || rows != width || columns != width {
                    candle_core::bail!("unsupported LM1B efficient attention mask");
                }
                (
                    Some(input_slice(storage, layout, n * width * width)?),
                    n as i32,
                )
            }
            None => (None, 0),
        };
        let device = &q.device;
        // SAFETY: dimensions and input extents are validated. The launched
        // kernel covers every element in this output.
        let mut output = unsafe { device.alloc::<f32>(count) }.map_err(candle_core::Error::wrap)?;
        let stream = device.cuda_stream();
        let missing_mask = 0u64;
        let length = width as i32;
        let heads_arg = heads as i32;
        let mut launch = stream.launch_builder(&self.function);
        launch.arg(&query).arg(&key).arg(&value);
        match &mask {
            Some(mask) => launch.arg(mask),
            None => launch.arg(&missing_mask),
        };
        launch
            .arg(&mut output)
            .arg(&length)
            .arg(&heads_arg)
            .arg(&mask_batches);
        // SAFETY: block layout and all pointer extents match the validated
        // BHLD inputs and CUDA stream. Arguments keep device access tracked.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (width.div_ceil(16) as u32, (batch * heads) as u32, 1),
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
            ql.shape().clone(),
        ))
    }
}

pub(crate) fn apply(
    backend: &crate::CandleBackend,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let device = backend.device().as_cuda_device()?;
    let function = device
        .get_or_load_custom_func("ncm_lm1b_efficient_attention", "ncm-lm1b-attention-v1", PTX)?
        .into_cuda_function();
    let op = Attention {
        function,
        mask: mask.map(Tensor::contiguous).transpose()?,
    };
    query
        .contiguous()?
        .apply_op3_no_bwd(&key.contiguous()?, &value.contiguous()?, &op)
}
