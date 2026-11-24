//! Ampere (SM80+) Tensor-Core FlashAttention-2 dispatch for BF16 and FP16 (`head_dim = 64`).
use candle_core::{
    CpuStorage, CudaStorage, CustomOp3, DType, Layout, Result, Shape, Storage, Tensor,
    cuda_backend::{
        CudaStorageSlice,
        cudarc::driver::{
            CudaFunction, CudaSlice, CudaStream, CudaView, DeviceRepr, LaunchConfig, PushKernelArg,
        },
    },
};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/fused_attention.ptx"));

struct FlashAttentionD64 {
    function: CudaFunction,
    mask: Option<Tensor>,
    scale: f32,
}

fn typed_slice<'a, T: DeviceRepr>(
    slice: &'a CudaSlice<T>,
    layout: &Layout,
    count: usize,
    label: &str,
) -> Result<CudaView<'a, T>> {
    let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg(format!("flash attention requires contiguous {label}"))
    })?;
    if end.checked_sub(start) != Some(count) || end > slice.len() {
        candle_core::bail!("flash attention {label} storage extent mismatch");
    }
    Ok(slice.slice(start..end))
}

struct LaunchMeta {
    q_count: usize,
    kv_count: usize,
    mask_count: usize,
    seq_q: i32,
    seq_k: i32,
    heads: i32,
    mask_batches: i32,
    mask_rows: i32,
    scale: f32,
    cfg: LaunchConfig,
}

#[allow(unsafe_code, clippy::too_many_arguments)]
fn run_flash<T: DeviceRepr>(
    device: &candle_core::CudaDevice,
    stream: &CudaStream,
    function: &CudaFunction,
    q: &CudaSlice<T>,
    ql: &Layout,
    k: &CudaSlice<T>,
    kl: &Layout,
    v: &CudaSlice<T>,
    vl: &Layout,
    mask: Option<(&CudaSlice<T>, &Layout)>,
    meta: &LaunchMeta,
) -> Result<CudaSlice<T>> {
    let qv = typed_slice(q, ql, meta.q_count, "q")?;
    let kv = typed_slice(k, kl, meta.kv_count, "k")?;
    let vv = typed_slice(v, vl, meta.kv_count, "v")?;
    let mv = match mask {
        Some((ms, ml)) => Some(typed_slice(ms, ml, meta.mask_count, "mask")?),
        None => None,
    };
    let null_mask = 0u64;
    // SAFETY: all input/output shapes and extents are validated; each threadblock writes
    // every element of its assigned query-row tile in disjoint memory.
    let mut output =
        unsafe { device.alloc::<T>(meta.q_count) }.map_err(candle_core::Error::wrap)?;
    let mut launch = stream.launch_builder(function);
    launch.arg(&qv).arg(&kv).arg(&vv);
    match &mv {
        Some(m) => launch.arg(m),
        None => launch.arg(&null_mask),
    };
    launch
        .arg(&mut output)
        .arg(&meta.seq_q)
        .arg(&meta.seq_k)
        .arg(&meta.heads)
        .arg(&meta.mask_batches)
        .arg(&meta.mask_rows)
        .arg(&meta.scale);
    // SAFETY: SM80+ PTX kernel arguments and bounds match validated device slices.
    unsafe { launch.launch(meta.cfg) }.map_err(candle_core::Error::wrap)?;
    Ok(output)
}

impl CustomOp3 for FlashAttentionD64 {
    fn name(&self) -> &'static str {
        "ncm-flash-attention-d64"
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
        candle_core::bail!("CUDA flash attention cannot run on CPU")
    }

    fn cuda_fwd(
        &self,
        q: &CudaStorage,
        ql: &Layout,
        k: &CudaStorage,
        kl: &Layout,
        v: &CudaStorage,
        vl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (batch, heads, seq_q, head_dim) = ql.shape().dims4()?;
        let (kb, kh, seq_k, kd) = kl.shape().dims4()?;
        let (vb, vh, sv, vd) = vl.shape().dims4()?;
        if head_dim != 64
            || batch == 0
            || heads == 0
            || seq_q == 0
            || seq_k == 0
            || kb != batch
            || kh != heads
            || kd != 64
            || vb != batch
            || vh != heads
            || sv != seq_k
            || vd != 64
        {
            candle_core::bail!("unsupported flash attention geometry")
        }

        let mask_guard = self.mask.as_ref().map(Tensor::storage_and_layout);
        let (mask_batches, mask_rows): (i32, i32) = match &mask_guard {
            Some((storage, mlayout)) => {
                if !matches!(&**storage, Storage::Cuda(_)) {
                    candle_core::bail!("flash attention mask must reside on CUDA");
                }
                let (mb, mh, mr, mc) = mlayout.shape().dims4()?;
                if !(mb == 1 || mb == batch) || mh != 1 || !(mr == 1 || mr == seq_q) || mc != seq_k
                {
                    candle_core::bail!("unsupported flash attention mask shape");
                }
                (mb as i32, mr as i32)
            }
            None => (0, 0),
        };

        let meta = LaunchMeta {
            q_count: batch * heads * seq_q * head_dim,
            kv_count: batch * heads * seq_k * head_dim,
            mask_count: (mask_batches as usize) * (mask_rows as usize) * seq_k,
            seq_q: i32::try_from(seq_q)
                .map_err(|_| candle_core::Error::Msg("seq_q overflow".into()))?,
            seq_k: i32::try_from(seq_k)
                .map_err(|_| candle_core::Error::Msg("seq_k overflow".into()))?,
            heads: i32::try_from(heads)
                .map_err(|_| candle_core::Error::Msg("heads overflow".into()))?,
            mask_batches,
            mask_rows,
            scale: self.scale,
            cfg: LaunchConfig {
                grid_dim: (seq_q.div_ceil(64) as u32, (batch * heads) as u32, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            },
        };

        let device = &q.device;
        let stream = device.cuda_stream();
        let out_slice = match (&q.slice, &k.slice, &v.slice) {
            (
                CudaStorageSlice::BF16(qs),
                CudaStorageSlice::BF16(ks),
                CudaStorageSlice::BF16(vs),
            ) => {
                let mpair = match &mask_guard {
                    Some((storage, ml)) => {
                        let Storage::Cuda(cs) = &**storage else {
                            unreachable!()
                        };
                        let CudaStorageSlice::BF16(ms) = &cs.slice else {
                            candle_core::bail!(
                                "flash attention mask dtype mismatch (expected BF16)"
                            )
                        };
                        Some((ms, *ml))
                    }
                    None => None,
                };
                CudaStorageSlice::BF16(run_flash(
                    device,
                    &stream,
                    &self.function,
                    qs,
                    ql,
                    ks,
                    kl,
                    vs,
                    vl,
                    mpair,
                    &meta,
                )?)
            }
            (CudaStorageSlice::F16(qs), CudaStorageSlice::F16(ks), CudaStorageSlice::F16(vs)) => {
                let mpair = match &mask_guard {
                    Some((storage, ml)) => {
                        let Storage::Cuda(cs) = &**storage else {
                            unreachable!()
                        };
                        let CudaStorageSlice::F16(ms) = &cs.slice else {
                            candle_core::bail!("flash attention mask dtype mismatch (expected F16)")
                        };
                        Some((ms, *ml))
                    }
                    None => None,
                };
                CudaStorageSlice::F16(run_flash(
                    device,
                    &stream,
                    &self.function,
                    qs,
                    ql,
                    ks,
                    kl,
                    vs,
                    vl,
                    mpair,
                    &meta,
                )?)
            }
            _ => candle_core::bail!("flash attention requires matching BF16 or F16 q/k/v slices"),
        };

        Ok((
            CudaStorage {
                slice: out_slice,
                device: device.clone(),
            },
            ql.shape().clone(),
        ))
    }
}

/// Returns true if `(q, k, v, mask)` matches the Ampere FlashAttention-2 `head_dim=64` fast path.
pub fn is_supported(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> bool {
    let dtype = q.dtype();
    if !matches!(dtype, DType::BF16 | DType::F16) || k.dtype() != dtype || v.dtype() != dtype {
        return false;
    }
    let Ok((batch, heads, seq_q, head_dim)) = q.dims4() else {
        return false;
    };
    let Ok((kb, kh, seq_k, kd)) = k.dims4() else {
        return false;
    };
    let Ok((vb, vh, sv, vd)) = v.dims4() else {
        return false;
    };
    if head_dim != 64
        || batch == 0
        || heads == 0
        || seq_q == 0
        || seq_k == 0
        || kb != batch
        || kh != heads
        || kd != 64
        || vb != batch
        || vh != heads
        || sv != seq_k
        || vd != 64
    {
        return false;
    }
    if let Some(m) = mask {
        if m.dtype() != dtype {
            return false;
        }
        let Ok((mb, mh, mr, mc)) = m.dims4() else {
            return false;
        };
        if !(mb == 1 || mb == batch) || mh != 1 || !(mr == 1 || mr == seq_q) || mc != seq_k {
            return false;
        }
    }
    true
}

/// Execute fused Ampere Tensor-Core FlashAttention-2 (`head_dim = 64`, `BF16` or `F16`).
pub fn apply(
    backend: &crate::CandleBackend,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let device = backend.device().as_cuda_device()?;
    let dtype = q.dtype();
    let kernel_name = match dtype {
        DType::BF16 => "ncm_flash_attention_bf16_d64",
        DType::F16 => "ncm_flash_attention_f16_d64",
        other => candle_core::bail!("flash attention requires BF16 or F16, got {other:?}"),
    };
    let function = device
        .get_or_load_custom_func(kernel_name, "ncm-flash-attention-v1", PTX)?
        .into_cuda_function();
    let qc = q.contiguous()?;
    let kc = k.contiguous()?;
    let vc = v.contiguous()?;
    let mc = mask.map(Tensor::contiguous).transpose()?;
    let op = FlashAttentionD64 {
        function,
        mask: mc,
        scale: 0.125f32,
    };
    qc.apply_op3_no_bwd(&kc, &vc, &op)
}
