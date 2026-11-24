//! Native Candle CPU & CUDA training backend (reverse-mode autodiff + AdamW + RVQ/EMA).
#![cfg(feature = "cpu")]

use crate::{
    Result, StepMetrics, TrainConfig, TrainError, TrainStage,
    artifact::{CheckpointMeta, LoadedCheckpoint, save_checkpoint},
    data::TokenDataset,
    ema::EmaCodebookState,
    lora,
    schedule::{
        TensorMap, area1d_matrix, level_attention_mask, linear1d_matrix, lr_at_step, rope_tables,
        splitmix64,
    },
};
use candle_core::{D, DType, Device, Tensor, Var};
use ncm_core::{CodecConfig, LocalGeneratorConfig, LocalHeadKind, LocalNcmConfig};
use std::{collections::BTreeMap, path::Path, time::Instant};

fn c_err(err: candle_core::Error) -> TrainError {
    TrainError::Backend(err.to_string())
}

/// Select a Candle execution device (`"cpu"` or `"cuda"` / `"cuda-cudnn"`).
pub fn resolve_device(device: &str) -> Result<Device> {
    match device {
        "cpu" => Ok(Device::Cpu),
        #[cfg(feature = "cuda")]
        "cuda" | "cuda-cudnn" => Device::new_cuda(0).map_err(c_err),
        other => Err(TrainError::Backend(format!(
            "Candle training device `{other}` is unavailable in this build"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Differentiable FP32 building blocks (matching ncm-model / ncm-backend-cuda)
// ---------------------------------------------------------------------------

fn linear_3d(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> candle_core::Result<Tensor> {
    linear_3d_impl(x, w, b, false)
}

fn linear_3d_impl(
    x: &Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
    frozen_weight: bool,
) -> candle_core::Result<Tensor> {
    let (bsz, seq, d_in) = x.dims3()?;
    let (d_out, _) = w.dims2()?;
    let x2 = x.reshape((bsz * seq, d_in))?;
    let wt = if frozen_weight {
        w.t()?.detach()
    } else {
        w.t()?
    };
    let y2 = x2.matmul(&wt)?;
    let y = y2.reshape((bsz, seq, d_out))?;
    match b {
        Some(bias) => y.broadcast_add(bias),
        None => Ok(y),
    }
}

fn linear_3d_maybe_lora(
    x: &Tensor,
    weight_name: &str,
    w: &Tensor,
    b: Option<&Tensor>,
    lora_ctx: Option<(&ParamStore, f64)>,
) -> candle_core::Result<Tensor> {
    let base = linear_3d_impl(x, w, b, lora_ctx.is_some())?;
    if let Some((lora_store, scale)) = lora_ctx
        && lora::is_lora_target(weight_name)
    {
        let (bsz, seq, d_in) = x.dims3()?;
        let (d_out, _) = w.dims2()?;
        let a = lora_store.tensor(&lora::lora_a_name(weight_name))?;
        let b_mat = lora_store.tensor(&lora::lora_b_name(weight_name))?;
        let x2 = x.reshape((bsz * seq, d_in))?;
        let delta = x2
            .matmul(&a.t()?)?
            .matmul(&b_mat.t()?)?
            .reshape((bsz, seq, d_out))?;
        let scaled = (delta * scale)?;
        return &base + scaled;
    }
    Ok(base)
}

fn layer_norm(x: &Tensor, w: &Tensor, b: Option<&Tensor>, eps: f64) -> candle_core::Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let inv = (var + eps)?.sqrt()?;
    let normed = xc.broadcast_div(&inv)?.broadcast_mul(w)?;
    match b {
        Some(bias) => normed.broadcast_add(bias),
        None => Ok(normed),
    }
}

fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let ms = x.sqr()?.mean_keepdim(D::Minus1)?;
    let inv = (ms + eps)?.sqrt()?;
    x.broadcast_div(&inv)?.broadcast_mul(w)
}

fn group_norm_nlc(x: &Tensor, groups: usize, eps: f64) -> candle_core::Result<Tensor> {
    let (bsz, seq, c) = x.dims3()?;
    let cpg = c / groups;
    let g = x
        .reshape((bsz, seq, groups, cpg))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((bsz, groups, seq * cpg))?;
    let mean = g.mean_keepdim(D::Minus1)?;
    let gc = g.broadcast_sub(&mean)?;
    let var = gc.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = gc.broadcast_div(&(var + eps)?.sqrt()?)?;
    normed
        .reshape((bsz, groups, seq, cpg))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((bsz, seq, c))
}

fn resample_mat(x: &Tensor, mat: &Tensor) -> candle_core::Result<Tensor> {
    let (bsz, t_in, d) = x.dims3()?;
    let (t_out, _) = mat.dims2()?;
    // x: [B, T_in, D] -> [B, D, T_in] -> [B*D, T_in] @ [T_in, T_out] -> [B, D, T_out] -> [B, T_out, D]
    let xt = x.transpose(1, 2)?.contiguous()?.reshape((bsz * d, t_in))?;
    let yt = xt.matmul(&mat.t()?)?;
    yt.reshape((bsz, d, t_out))?.transpose(1, 2)?.contiguous()
}

fn conv1d_stride_eq_k(
    x: &Tensor,
    w: &Tensor,
    b: &Tensor,
    stride: usize,
) -> candle_core::Result<Tensor> {
    let (bsz, seq, c_in) = x.dims3()?;
    let (c_out, _, k) = w.dims3()?;
    debug_assert_eq!(k, stride);
    let out_len = seq / stride;
    let x_col = x
        .reshape((bsz, out_len, stride, c_in))?
        .transpose(2, 3)?
        .contiguous()?
        .reshape((bsz * out_len, c_in * stride))?;
    let w_col = w.reshape((c_out, c_in * stride))?;
    let y = x_col.matmul(&w_col.t()?)?.reshape((bsz, out_len, c_out))?;
    y.broadcast_add(b)
}

fn conv_transpose1d_stride_eq_k(
    x: &Tensor,
    w: &Tensor,
    b: &Tensor,
    stride: usize,
) -> candle_core::Result<Tensor> {
    let (bsz, seq, c_in) = x.dims3()?;
    let (_, c_out, k) = w.dims3()?;
    debug_assert_eq!(k, stride);
    let x2 = x.reshape((bsz * seq, c_in))?;
    let w2 = w
        .transpose(1, 2)?
        .contiguous()?
        .reshape((c_in, stride * c_out))?;
    let y = x2.matmul(&w2)?.reshape((bsz, seq * stride, c_out))?;
    y.broadcast_add(b)
}

fn conv1d_k3_p1(x: &Tensor, w: &Tensor, b: &Tensor) -> candle_core::Result<Tensor> {
    let (bsz, seq, c_in) = x.dims3()?;
    let (c_out, _, _) = w.dims3()?;
    let zeros = Tensor::zeros((bsz, 1, c_in), DType::F32, x.device())?;
    let x_left = if seq > 1 {
        Tensor::cat(&[&zeros, &x.narrow(1, 0, seq - 1)?], 1)?
    } else {
        zeros.clone()
    };
    let x_right = if seq > 1 {
        Tensor::cat(&[&x.narrow(1, 1, seq - 1)?, &zeros], 1)?
    } else {
        zeros
    };
    let x_cat = Tensor::cat(&[&x_left, x, &x_right], 2)?.reshape((bsz * seq, 3 * c_in))?;
    let w_col = w
        .transpose(1, 2)?
        .contiguous()?
        .reshape((c_out, 3 * c_in))?;
    let y = x_cat.matmul(&w_col.t()?)?.reshape((bsz, seq, c_out))?;
    y.broadcast_add(b)
}

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
    let (_, _, _, head_dim) = x.dims4()?;
    let half = head_dim / 2;
    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;
    let c = cos.unsqueeze(0)?.unsqueeze(0)?;
    let s = sin.unsqueeze(0)?.unsqueeze(0)?;
    let o1 = (x1.broadcast_mul(&c)? - x2.broadcast_mul(&s)?)?;
    let o2 = (x2.broadcast_mul(&c)? + x1.broadcast_mul(&s)?)?;
    Tensor::cat(&[&o1, &o2], 3)
}

fn softmax_differentiable(x: &Tensor) -> candle_core::Result<Tensor> {
    let max = x.max_keepdim(D::Minus1)?.detach();
    let shifted = x.broadcast_sub(&max)?;
    let exps = shifted.exp()?;
    let denom = exps.sum_keepdim(D::Minus1)?;
    exps.broadcast_div(&denom)
}

fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> candle_core::Result<Tensor> {
    let (_, _, _, head_dim) = q.dims4()?;
    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
    if let Some(m) = mask {
        scores = scores.broadcast_add(m)?;
    }
    let probs = softmax_differentiable(&scores)?;
    probs.matmul(&v.contiguous()?)
}

fn embedding_gather(table: &Tensor, ids: &Tensor) -> candle_core::Result<Tensor> {
    let dims = ids.dims();
    let flat = ids.flatten_all()?;
    let gathered = table.index_select(&flat, 0)?;
    let d = table.dim(1)?;
    let mut out_shape = dims.to_vec();
    out_shape.push(d);
    gathered.reshape(out_shape)
}

fn squared_distances(x: &Tensor, book: &Tensor) -> candle_core::Result<Tensor> {
    let xx = x.sqr()?.sum_keepdim(1)?;
    let ee = book.sqr()?.sum_keepdim(1)?.t()?;
    let dot = (x.matmul(&book.t()?)? * 2.0)?;
    xx.broadcast_add(&ee)?.broadcast_sub(&dot)
}

// ---------------------------------------------------------------------------
// Weight container helpers
// ---------------------------------------------------------------------------

struct ParamStore {
    vars: BTreeMap<String, Var>,
    m: BTreeMap<String, Tensor>,
    v: BTreeMap<String, Tensor>,
}

impl ParamStore {
    fn from_host(
        weights: &TensorMap,
        optim_m: Option<&TensorMap>,
        optim_v: Option<&TensorMap>,
        device: &Device,
    ) -> Result<Self> {
        let mut vars = BTreeMap::new();
        let mut m = BTreeMap::new();
        let mut v = BTreeMap::new();
        for (name, (shape, vals)) in weights {
            let t = Tensor::from_slice(vals, shape.as_slice(), device).map_err(c_err)?;
            let var = Var::from_tensor(&t).map_err(c_err)?;
            let m_t = if let Some(mm) = optim_m
                && let Some((ms, mv)) = mm.get(name)
            {
                Tensor::from_slice(mv, ms.as_slice(), device).map_err(c_err)?
            } else {
                Tensor::zeros(shape.as_slice(), DType::F32, device).map_err(c_err)?
            };
            let v_t = if let Some(vv) = optim_v
                && let Some((vs, val)) = vv.get(name)
            {
                Tensor::from_slice(val, vs.as_slice(), device).map_err(c_err)?
            } else {
                Tensor::zeros(shape.as_slice(), DType::F32, device).map_err(c_err)?
            };
            vars.insert(name.clone(), var);
            m.insert(name.clone(), m_t);
            v.insert(name.clone(), v_t);
        }
        Ok(Self { vars, m, v })
    }

    fn tensor(&self, name: &str) -> candle_core::Result<Tensor> {
        self.vars
            .get(name)
            .map(Var::as_tensor)
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg(format!("missing weight `{name}`")))
    }

    fn export_weights(&self) -> Result<TensorMap> {
        let mut map = TensorMap::new();
        for (name, var) in &self.vars {
            let t = var.as_tensor();
            let shape = t.dims().to_vec();
            let vals = t
                .flatten_all()
                .map_err(c_err)?
                .to_vec1::<f32>()
                .map_err(c_err)?;
            map.insert(name.clone(), (shape, vals));
        }
        Ok(map)
    }

    fn export_moments(&self) -> Result<(TensorMap, TensorMap)> {
        let mut mm = TensorMap::new();
        let mut vv = TensorMap::new();
        for (name, t) in &self.m {
            let shape = t.dims().to_vec();
            let vals = t
                .flatten_all()
                .map_err(c_err)?
                .to_vec1::<f32>()
                .map_err(c_err)?;
            mm.insert(name.clone(), (shape, vals));
        }
        for (name, t) in &self.v {
            let shape = t.dims().to_vec();
            let vals = t
                .flatten_all()
                .map_err(c_err)?
                .to_vec1::<f32>()
                .map_err(c_err)?;
            vv.insert(name.clone(), (shape, vals));
        }
        Ok((mm, vv))
    }

    /// Decoupled AdamW + global L2 gradient clipping from a `GradStore`.
    #[allow(clippy::too_many_arguments)]
    fn adamw_step(
        &mut self,
        grads: &candle_core::backprop::GradStore,
        skip_prefix: Option<&str>,
        step_one_based: usize,
        lr: f64,
        weight_decay: f64,
        grad_clip_norm: f64,
    ) -> Result<f64> {
        let mut grad_map = BTreeMap::new();
        for (name, var) in &self.vars {
            if let Some(pfx) = skip_prefix
                && name.starts_with(pfx)
            {
                continue;
            }
            if let Some(g) = grads.get(var.as_tensor()) {
                grad_map.insert(name.clone(), g.clone());
            }
        }
        self.adamw_step_from_map(&grad_map, step_one_based, lr, weight_decay, grad_clip_norm)
    }

    /// Accumulate gradients from a `GradStore` into `accum` (for per-level backward and grad accum).
    fn accumulate_grads(
        &self,
        grads: &candle_core::backprop::GradStore,
        accum: &mut BTreeMap<String, Tensor>,
    ) -> Result<()> {
        for (name, var) in &self.vars {
            if let Some(g) = grads.get(var.as_tensor()) {
                let g_det = g.detach();
                if let Some(prev) = accum.get(name) {
                    let sum = (prev + &g_det).map_err(c_err)?;
                    accum.insert(name.clone(), sum);
                } else {
                    accum.insert(name.clone(), g_det);
                }
            }
        }
        Ok(())
    }

    /// Decoupled AdamW + global L2 gradient clipping from an accumulated gradient map.
    fn adamw_step_from_map(
        &mut self,
        grad_map: &BTreeMap<String, Tensor>,
        step_one_based: usize,
        lr: f64,
        weight_decay: f64,
        grad_clip_norm: f64,
    ) -> Result<f64> {
        let beta1 = 0.9_f64;
        let beta2 = 0.999_f64;
        let eps = 1e-8_f64;

        let mut active: Vec<(String, Tensor)> = Vec::with_capacity(grad_map.len());
        let mut sq_sum = 0.0_f64;
        for (name, g) in grad_map {
            let norm_sq = g
                .sqr()
                .and_then(|x| x.sum_all())
                .and_then(|x| x.to_scalar::<f32>())
                .map_err(c_err)? as f64;
            sq_sum += norm_sq;
            active.push((name.clone(), g.clone()));
        }
        let grad_norm = sq_sum.sqrt();
        let clip_scale = if grad_clip_norm > 0.0 && grad_norm > grad_clip_norm {
            grad_clip_norm / (grad_norm + 1e-6)
        } else {
            1.0
        };

        let bc1 = 1.0 - beta1.powi(step_one_based as i32);
        let bc2 = 1.0 - beta2.powi(step_one_based as i32);

        for (name, g_raw) in active {
            let g = if clip_scale < 1.0 {
                (g_raw * clip_scale).map_err(c_err)?
            } else {
                g_raw
            };
            let m_prev = self.m.get(&name).unwrap();
            let v_prev = self.v.get(&name).unwrap();
            let m_new = ((m_prev * beta1).map_err(c_err)? + (&g * (1.0 - beta1)).map_err(c_err)?)
                .map_err(c_err)?;
            let v_new = ((v_prev * beta2).map_err(c_err)?
                + (g.sqr().map_err(c_err)? * (1.0 - beta2)).map_err(c_err)?)
            .map_err(c_err)?;

            let m_hat = (&m_new / bc1).map_err(c_err)?;
            let v_hat = (&v_new / bc2).map_err(c_err)?;
            let denom = (v_hat.sqrt().map_err(c_err)? + eps).map_err(c_err)?;
            let update = (m_hat.broadcast_div(&denom).map_err(c_err)? * lr).map_err(c_err)?;

            let var = self.vars.get(&name).unwrap();
            let w_cur = var.as_tensor();
            let apply_wd = weight_decay > 0.0
                && w_cur.rank() >= 2
                && !name.contains("ln_")
                && !name.contains("q_norm")
                && !name.contains("k_norm");
            let w_base = if apply_wd {
                (w_cur * (1.0 - lr * weight_decay)).map_err(c_err)?
            } else {
                w_cur.clone()
            };
            let w_next = (w_base - update).map_err(c_err)?;
            var.set(&w_next).map_err(c_err)?;
            self.m.insert(name.clone(), m_new);
            self.v.insert(name, v_new);
        }
        Ok(grad_norm)
    }
}

// ---------------------------------------------------------------------------
// Codec forward (differentiable STE + EMA capture, OR frozen inference encode)
// ---------------------------------------------------------------------------

struct CodecPrecomputed {
    area_mats: Vec<Option<Tensor>>,
    linear_mats: Vec<Option<Tensor>>,
}

impl CodecPrecomputed {
    fn new(codec_cfg: &CodecConfig, device: &Device) -> Result<Self> {
        let ctx = codec_cfg.raw().context_length;
        let levels = codec_cfg.levels();
        let mut area_mats = Vec::with_capacity(levels.len());
        let mut linear_mats = Vec::with_capacity(levels.len());
        for (idx, &len) in levels.iter().enumerate() {
            let terminal = idx + 1 == levels.len();
            if idx > 0 && !terminal {
                let am = area1d_matrix(ctx, len);
                let lm = linear1d_matrix(len, ctx);
                area_mats.push(Some(
                    Tensor::from_slice(&am, (len, ctx), device).map_err(c_err)?,
                ));
                linear_mats.push(Some(
                    Tensor::from_slice(&lm, (ctx, len), device).map_err(c_err)?,
                ));
            } else {
                area_mats.push(None);
                linear_mats.push(None);
            }
        }
        Ok(Self {
            area_mats,
            linear_mats,
        })
    }
}

fn codec_cascade(
    codec_cfg: &CodecConfig,
    weights: &impl Fn(&str) -> candle_core::Result<Tensor>,
    x: &Tensor,
    up: bool,
) -> candle_core::Result<Tensor> {
    let q = &codec_cfg.raw().vector_quantizer_config;
    let mut strides = codec_cfg.strides().to_vec();
    if up {
        strides.reverse();
    }
    let mut val = x.clone();
    for (i, stride) in strides.into_iter().enumerate() {
        if i > 0 && q.learned_l1_cascade_norm {
            val = group_norm_nlc(&val, 8, 1e-5)?;
        }
        let index = if q.learned_l1_cascade_norm { i * 2 } else { i };
        let name = if q.learned_l1_cascade {
            format!("vq.learned_{}_l1.{index}", if up { "up" } else { "down" })
        } else {
            format!("vq.learned_{}_l1", if up { "up" } else { "down" })
        };
        let w = weights(&format!("{name}.weight"))?;
        let b = weights(&format!("{name}.bias"))?;
        val = if up {
            conv_transpose1d_stride_eq_k(&val, &w, &b, stride)?
        } else {
            conv1d_stride_eq_k(&val, &w, &b, stride)?
        };
    }
    if !up && q.learned_l1_cascade_norm_final {
        val = group_norm_nlc(&val, 8, 1e-5)?;
    }
    Ok(val)
}

fn codec_refine(
    codec_cfg: &CodecConfig,
    weights: &impl Fn(&str) -> candle_core::Result<Tensor>,
    x: &Tensor,
    level: usize,
) -> candle_core::Result<Tensor> {
    let ratio = codec_cfg.raw().vector_quantizer_config.quant_resi.ratio;
    let name = format!("vq.quant_resi.{level}");
    let w = weights(&format!("{name}.weight"))?;
    let b = weights(&format!("{name}.bias"))?;
    let conv = conv1d_k3_p1(x, &w, &b)?;
    (x * (1.0 - ratio))? + (conv * ratio)?
}

fn codec_decoder_layer(
    codec_cfg: &CodecConfig,
    weights: &impl Fn(&str) -> candle_core::Result<Tensor>,
    x: &Tensor,
    name: &str,
) -> candle_core::Result<Tensor> {
    let (bsz, seq, width) = x.dims3()?;
    let heads = codec_cfg.raw().n_layers;
    let head_dim = width / heads;

    let norm = layer_norm(
        x,
        &weights(&format!("{name}.ln_1.weight"))?,
        Some(&weights(&format!("{name}.ln_1.bias"))?),
        1e-6,
    )?;
    let qkv = linear_3d(
        &norm,
        &weights(&format!("{name}.attn.c_attn.weight"))?,
        Some(&weights(&format!("{name}.attn.c_attn.bias"))?),
    )?;
    let q = qkv
        .narrow(2, 0, width)?
        .reshape((bsz, seq, heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()?;
    let k = qkv
        .narrow(2, width, width)?
        .reshape((bsz, seq, heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()?;
    let v = qkv
        .narrow(2, 2 * width, width)?
        .reshape((bsz, seq, heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()?;
    let q = rms_norm(&q, &weights(&format!("{name}.attn.q_norm.weight"))?, 1e-6)?;
    let k = rms_norm(&k, &weights(&format!("{name}.attn.k_norm.weight"))?, 1e-6)?;
    let attn = attention(&q, &k, &v, None)?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((bsz, seq, width))?;
    let attn_out = linear_3d(
        &attn,
        &weights(&format!("{name}.attn.c_proj.weight"))?,
        Some(&weights(&format!("{name}.attn.c_proj.bias"))?),
    )?;
    let residual = (x + attn_out)?;

    let norm2 = layer_norm(
        &residual,
        &weights(&format!("{name}.ln_2.weight"))?,
        Some(&weights(&format!("{name}.ln_2.bias"))?),
        1e-6,
    )?;
    let h = linear_3d(
        &norm2,
        &weights(&format!("{name}.mlp.c_fc.weight"))?,
        Some(&weights(&format!("{name}.mlp.c_fc.bias"))?),
    )?
    .gelu_erf()?;
    let mlp = linear_3d(
        &h,
        &weights(&format!("{name}.mlp.c_proj.weight"))?,
        Some(&weights(&format!("{name}.mlp.c_proj.bias"))?),
    )?;
    &residual + mlp
}

struct CodecForwardOutput {
    total_loss: Tensor,
    recon_loss: f64,
    commit_loss: f64,
    ema_captures: Vec<(Vec<f32>, Vec<u32>)>,
}

#[allow(clippy::too_many_arguments)]
fn codec_forward_train(
    codec_cfg: &CodecConfig,
    pre: &CodecPrecomputed,
    store: &ParamStore,
    tokens: &[u32],
    batch: usize,
    commitment_weight: f64,
    capture_ema: bool,
    device: &Device,
) -> Result<CodecForwardOutput> {
    let raw = codec_cfg.raw();
    let ctx = raw.context_length;
    let width = raw.embed_dim;
    let ids = Tensor::from_slice(tokens, (batch, ctx), device).map_err(c_err)?;
    let get = |name: &str| store.tensor(name);

    let mut enc =
        embedding_gather(&get("encoder.wte.weight").map_err(c_err)?, &ids).map_err(c_err)?;
    if raw.enc_positional {
        let wpe = get("encoder.wpe.weight").map_err(c_err)?;
        enc = enc.broadcast_add(&wpe).map_err(c_err)?;
    }
    let mut residual = group_norm_nlc(&enc, raw.pre_quant_groupnorm, 1e-5).map_err(c_err)?;

    let mut cumulative: Option<Tensor> = None;
    let mut total_commit: Option<Tensor> = None;
    let mut ema_captures = Vec::new();
    let levels = codec_cfg.levels();

    for (level, &level_len) in levels.iter().enumerate() {
        let terminal = level + 1 == levels.len();
        let down = if terminal {
            residual.clone()
        } else if level == 0 {
            codec_cascade(codec_cfg, &get, &residual, false).map_err(c_err)?
        } else {
            resample_mat(&residual, pre.area_mats[level].as_ref().unwrap()).map_err(c_err)?
        };
        let flat_ze = down.reshape((batch * level_len, width)).map_err(c_err)?;
        let book = get(&format!("vq.embeddings.{level}.weight")).map_err(c_err)?;
        let book_det = book.detach();

        let dists = squared_distances(&flat_ze.detach(), &book_det).map_err(c_err)?;
        let code_ids = dists.argmin(1).map_err(c_err)?;
        let z_q_flat = book_det.index_select(&code_ids, 0).map_err(c_err)?;
        let z_q = z_q_flat.reshape((batch, level_len, width)).map_err(c_err)?;

        if capture_ema {
            let ze_host = flat_ze
                .detach()
                .flatten_all()
                .map_err(c_err)?
                .to_vec1::<f32>()
                .map_err(c_err)?;
            let ids_host = code_ids.to_vec1::<u32>().map_err(c_err)?;
            ema_captures.push((ze_host, ids_host));
        }

        // Commitment loss: ||z_e - sg[z_q]||^2
        let commit = (&down - &z_q)
            .map_err(c_err)?
            .sqr()
            .map_err(c_err)?
            .mean_all()
            .map_err(c_err)?;
        total_commit = Some(match total_commit {
            Some(prev) => (prev + commit).map_err(c_err)?,
            None => commit,
        });

        // Straight-through estimator: forward = z_q, backward = dz_e
        let z_st = (&down + (&z_q - &down).map_err(c_err)?.detach()).map_err(c_err)?;

        let up = if level == 0 {
            codec_cascade(codec_cfg, &get, &z_st, true).map_err(c_err)?
        } else if terminal {
            z_st
        } else {
            resample_mat(&z_st, pre.linear_mats[level].as_ref().unwrap()).map_err(c_err)?
        };
        let contrib = codec_refine(codec_cfg, &get, &up, level).map_err(c_err)?;
        let sum = match cumulative {
            Some(ref prev) => (prev + &contrib).map_err(c_err)?,
            None => contrib.clone(),
        };
        residual = (&residual - &contrib).map_err(c_err)?;
        cumulative = Some(sum);
    }

    let mut hidden = cumulative.expect("nonempty codec levels");
    if raw.dec_positional {
        let wpe = get("decoder.wpe.weight").map_err(c_err)?;
        hidden = hidden.broadcast_add(&wpe).map_err(c_err)?;
    }
    hidden =
        codec_decoder_layer(codec_cfg, &get, &hidden, "decoder.middle_block").map_err(c_err)?;
    for idx in 0..raw.dec_layers - 1 {
        hidden = codec_decoder_layer(
            codec_cfg,
            &get,
            &hidden,
            &format!("decoder.extra_blocks.{idx}"),
        )
        .map_err(c_err)?;
    }
    let ln_f = layer_norm(
        &hidden,
        &get("decoder.ln_f.weight").map_err(c_err)?,
        None,
        1e-6,
    )
    .map_err(c_err)?;
    let logits = linear_3d(
        &ln_f,
        &get("decoder.output_layer.weight").map_err(c_err)?,
        None,
    )
    .map_err(c_err)?;

    let logits_2d = logits
        .reshape((batch * ctx, raw.in_vocab_size))
        .map_err(c_err)?;
    let targets_1d = ids.flatten_all().map_err(c_err)?;
    let recon = candle_nn::loss::cross_entropy(&logits_2d, &targets_1d).map_err(c_err)?;
    let commit = total_commit.expect("nonempty levels");
    let total_loss = (&recon + (&commit * commitment_weight).map_err(c_err)?).map_err(c_err)?;

    let recon_loss = recon.to_scalar::<f32>().map_err(c_err)? as f64;
    let commit_loss = commit.to_scalar::<f32>().map_err(c_err)? as f64;
    Ok(CodecForwardOutput {
        total_loss,
        recon_loss,
        commit_loss,
        ema_captures,
    })
}

// ---------------------------------------------------------------------------
// Frozen-codec encoder + Differentiable Generator forward
// ---------------------------------------------------------------------------

struct GeneratorPrecomputed {
    area_mats: Vec<Option<Tensor>>,
    linear_mats: Vec<Option<Tensor>>,
    gen_area_mats: Vec<Option<Tensor>>,
    cos: Vec<Tensor>,
    sin: Vec<Tensor>,
    masks: Vec<Tensor>,
}

impl GeneratorPrecomputed {
    fn new(config: &LocalNcmConfig, device: &Device) -> Result<Self> {
        let codec_cfg = config.codec();
        let gen_cfg = config.model();
        let ctx = codec_cfg.raw().context_length;
        let levels = codec_cfg.levels();
        let head_dim = gen_cfg.n_embd / gen_cfg.n_head;

        let mut area_mats = Vec::with_capacity(levels.len());
        let mut linear_mats = Vec::with_capacity(levels.len());
        let mut gen_area_mats = Vec::with_capacity(levels.len());
        let mut cos = Vec::with_capacity(levels.len());
        let mut sin = Vec::with_capacity(levels.len());
        let mut masks = Vec::with_capacity(levels.len());

        for (idx, &len) in levels.iter().enumerate() {
            let terminal = idx + 1 == levels.len();
            if idx > 0 && !terminal {
                area_mats.push(Some(
                    Tensor::from_slice(&area1d_matrix(ctx, len), (len, ctx), device)
                        .map_err(c_err)?,
                ));
                linear_mats.push(Some(
                    Tensor::from_slice(&linear1d_matrix(len, ctx), (ctx, len), device)
                        .map_err(c_err)?,
                ));
            } else {
                area_mats.push(None);
                linear_mats.push(None);
            }
            if idx > 0 {
                gen_area_mats.push(Some(
                    Tensor::from_slice(&area1d_matrix(ctx, len), (len, ctx), device)
                        .map_err(c_err)?,
                ));
            } else {
                gen_area_mats.push(None);
            }
            let seq = gen_cfg.prefix_len + len;
            let (c, s) = rope_tables(gen_cfg.prefix_len, len, head_dim, gen_cfg.rope_base);
            let m = level_attention_mask(gen_cfg.prefix_len, len);
            cos.push(Tensor::from_slice(&c, (seq, head_dim / 2), device).map_err(c_err)?);
            sin.push(Tensor::from_slice(&s, (seq, head_dim / 2), device).map_err(c_err)?);
            masks.push(Tensor::from_slice(&m, (1, 1, seq, seq), device).map_err(c_err)?);
        }
        Ok(Self {
            area_mats,
            linear_mats,
            gen_area_mats,
            cos,
            sin,
            masks,
        })
    }
}

/// Encode target token sequences through frozen codec weights to produce ground-truth
/// RVQ code targets `gt_codes[level]` and teacher-forced conditioning latents `cond_latents[level]`
/// (`None` for level 0, which uses `l0_queries`).
fn frozen_codec_encode_for_generator(
    codec_cfg: &CodecConfig,
    pre: &GeneratorPrecomputed,
    frozen_codec: &BTreeMap<String, Tensor>,
    targets: &[u32],
    batch: usize,
    device: &Device,
) -> Result<(Vec<Tensor>, Vec<Option<Tensor>>)> {
    let raw = codec_cfg.raw();
    let ctx = raw.context_length;
    let width = raw.embed_dim;
    let get = |name: &str| {
        frozen_codec
            .get(name)
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg(format!("missing frozen codec `{name}`")))
    };

    let ids = Tensor::from_slice(targets, (batch, ctx), device).map_err(c_err)?;
    let mut enc =
        embedding_gather(&get("encoder.wte.weight").map_err(c_err)?, &ids).map_err(c_err)?;
    if raw.enc_positional {
        enc = enc
            .broadcast_add(&get("encoder.wpe.weight").map_err(c_err)?)
            .map_err(c_err)?;
    }
    let mut residual = group_norm_nlc(&enc, raw.pre_quant_groupnorm, 1e-5).map_err(c_err)?;
    let levels = codec_cfg.levels();
    let mut cumulative: Option<Tensor> = None;
    let mut gt_codes = Vec::with_capacity(levels.len());
    let mut cond_latents = Vec::with_capacity(levels.len());

    for (level, &level_len) in levels.iter().enumerate() {
        let terminal = level + 1 == levels.len();
        if level == 0 {
            cond_latents.push(None);
        } else {
            let cum = cumulative.as_ref().expect("preceding cumulative latent");
            let down_cond = if level_len == ctx {
                cum.clone()
            } else {
                resample_mat(cum, pre.gen_area_mats[level].as_ref().unwrap()).map_err(c_err)?
            };
            cond_latents.push(Some(down_cond.detach()));
        }

        let down = if terminal {
            residual.clone()
        } else if level == 0 {
            codec_cascade(codec_cfg, &get, &residual, false).map_err(c_err)?
        } else {
            resample_mat(&residual, pre.area_mats[level].as_ref().unwrap()).map_err(c_err)?
        };
        let flat = down.reshape((batch * level_len, width)).map_err(c_err)?;
        let book = get(&format!("vq.embeddings.{level}.weight")).map_err(c_err)?;
        let code_ids = squared_distances(&flat, &book)
            .map_err(c_err)?
            .argmin(1)
            .map_err(c_err)?;
        let z_q = book
            .index_select(&code_ids, 0)
            .map_err(c_err)?
            .reshape((batch, level_len, width))
            .map_err(c_err)?;
        let up = if level == 0 {
            codec_cascade(codec_cfg, &get, &z_q, true).map_err(c_err)?
        } else if terminal {
            z_q
        } else {
            resample_mat(&z_q, pre.linear_mats[level].as_ref().unwrap()).map_err(c_err)?
        };
        let contrib = codec_refine(codec_cfg, &get, &up, level).map_err(c_err)?;
        let sum = match cumulative {
            Some(ref prev) => (prev + &contrib).map_err(c_err)?,
            None => contrib.clone(),
        };
        residual = (&residual - &contrib).map_err(c_err)?;
        cumulative = Some(sum);
        gt_codes.push(code_ids.detach());
    }
    Ok((gt_codes, cond_latents))
}

fn build_combined_mask(
    base_mask: &Tensor,
    prefixes: &[u32],
    batch: usize,
    prefix_len: usize,
    level_len: usize,
    pad_id: Option<u32>,
    device: &Device,
) -> Result<Tensor> {
    let Some(pad) = pad_id else {
        return Ok(base_mask.clone());
    };
    let seq = prefix_len + level_len;
    let mut mask = vec![0.0f32; batch * seq * seq];
    for b in 0..batch {
        let row_slice = &prefixes[b * prefix_len..(b + 1) * prefix_len];
        for r in 0..seq {
            for c in 0..seq {
                let mut val = 0.0f32;
                if r < prefix_len && c >= prefix_len {
                    val = f32::NEG_INFINITY;
                }
                if c < prefix_len && row_slice[c] == pad {
                    val = f32::NEG_INFINITY;
                }
                mask[(b * seq + r) * seq + c] = val;
            }
        }
    }
    Tensor::from_slice(&mask, (batch, 1, seq, seq), device).map_err(c_err)
}

#[allow(clippy::too_many_arguments)]
fn generator_block(
    gen_cfg: &LocalGeneratorConfig,
    get: &impl Fn(&str) -> candle_core::Result<Tensor>,
    lora_ctx: Option<(&ParamStore, f64)>,
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mask: &Tensor,
    layer: usize,
) -> candle_core::Result<Tensor> {
    let (bsz, seq, width) = x.dims3()?;
    let heads = gen_cfg.n_head;
    let head_dim = width / heads;
    let p = format!("transformer.blocks.{layer}");

    let norm1 = layer_norm(
        x,
        &get(&format!("{p}.ln_1.weight"))?,
        Some(&get(&format!("{p}.ln_1.bias"))?),
        1e-6,
    )?;
    let c_attn_w_name = format!("{p}.attn.c_attn.weight");
    let qkv = linear_3d_maybe_lora(
        &norm1,
        &c_attn_w_name,
        &get(&c_attn_w_name)?,
        Some(&get(&format!("{p}.attn.c_attn.bias"))?),
        lora_ctx,
    )?;
    let q = qkv
        .narrow(2, 0, width)?
        .reshape((bsz, seq, heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()?;
    let k = qkv
        .narrow(2, width, width)?
        .reshape((bsz, seq, heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()?;
    let v = qkv
        .narrow(2, 2 * width, width)?
        .reshape((bsz, seq, heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()?;

    let q = rms_norm(&q, &get(&format!("{p}.attn.q_norm.weight"))?, 1e-6)?;
    let k = rms_norm(&k, &get(&format!("{p}.attn.k_norm.weight"))?, 1e-6)?;
    let q = apply_rope(&q, cos, sin)?;
    let k = apply_rope(&k, cos, sin)?;

    let attn = attention(&q, &k, &v, Some(mask))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((bsz, seq, width))?;
    let c_proj_w_name = format!("{p}.attn.c_proj.weight");
    let attn_out = linear_3d_maybe_lora(
        &attn,
        &c_proj_w_name,
        &get(&c_proj_w_name)?,
        Some(&get(&format!("{p}.attn.c_proj.bias"))?),
        lora_ctx,
    )?;
    let res = (x + attn_out)?;

    let norm2 = layer_norm(
        &res,
        &get(&format!("{p}.ln_2.weight"))?,
        Some(&get(&format!("{p}.ln_2.bias"))?),
        1e-6,
    )?;
    let c_fc_w_name = format!("{p}.mlp.c_fc.weight");
    let h = linear_3d_maybe_lora(
        &norm2,
        &c_fc_w_name,
        &get(&c_fc_w_name)?,
        Some(&get(&format!("{p}.mlp.c_fc.bias"))?),
        lora_ctx,
    )?
    .gelu_erf()?;
    let mlp_proj_w_name = format!("{p}.mlp.c_proj.weight");
    let mlp = linear_3d_maybe_lora(
        &h,
        &mlp_proj_w_name,
        &get(&mlp_proj_w_name)?,
        Some(&get(&format!("{p}.mlp.c_proj.bias"))?),
        lora_ctx,
    )?;
    &res + mlp
}

#[allow(clippy::too_many_arguments)]
fn generator_level_loss(
    config: &LocalNcmConfig,
    pre: &GeneratorPrecomputed,
    frozen_wte: &Tensor,
    get: &impl Fn(&str) -> candle_core::Result<Tensor>,
    lora_ctx: Option<(&ParamStore, f64)>,
    prefixes: &[u32],
    gt_code: &Tensor,
    cond_latent: Option<&Tensor>,
    level: usize,
    batch: usize,
    device: &Device,
) -> Result<Tensor> {
    let codec_cfg = config.codec();
    let gen_cfg = config.model();
    let level_len = codec_cfg.levels()[level];

    let frozen_base = lora_ctx.is_some();
    let prefix_ids =
        Tensor::from_slice(prefixes, (batch, gen_cfg.prefix_len), device).map_err(c_err)?;
    let prefix_raw = embedding_gather(frozen_wte, &prefix_ids).map_err(c_err)?;
    let wte_w = get("transformer.wte.weight").map_err(c_err)?;
    let wte_b = get("transformer.wte.bias").map_err(c_err)?;
    let prefix_h = linear_3d_impl(&prefix_raw, &wte_w, Some(&wte_b), frozen_base).map_err(c_err)?;

    let wle = get("transformer.wle.weight").map_err(c_err)?;
    let ln_f_w = get("transformer.ln_f.weight").map_err(c_err)?;
    let ln_f_b = get("transformer.ln_f.bias").map_err(c_err)?;

    let latent = if level == 0 {
        let l0_queries = get("l0_queries").map_err(c_err)?;
        let zeros = Tensor::zeros(
            (batch, level_len, codec_cfg.raw().embed_dim),
            DType::F32,
            device,
        )
        .map_err(c_err)?;
        l0_queries.broadcast_add(&zeros).map_err(c_err)?
    } else {
        cond_latent.expect("cond latent for level > 0").clone()
    };
    let lvl_emb = wle.narrow(0, level, 1).map_err(c_err)?;
    let target_h = linear_3d_impl(&latent, &wte_w, Some(&wte_b), frozen_base)
        .map_err(c_err)?
        .broadcast_add(&lvl_emb)
        .map_err(c_err)?;

    let mut x = Tensor::cat(&[&prefix_h, &target_h], 1).map_err(c_err)?;
    let mask = build_combined_mask(
        &pre.masks[level],
        prefixes,
        batch,
        gen_cfg.prefix_len,
        level_len,
        gen_cfg.pad_token_id,
        device,
    )?;
    for layer in 0..gen_cfg.n_layer {
        x = generator_block(
            gen_cfg,
            get,
            lora_ctx,
            &x,
            &pre.cos[level],
            &pre.sin[level],
            &mask,
            layer,
        )
        .map_err(c_err)?;
    }
    let target_out = x.narrow(1, gen_cfg.prefix_len, level_len).map_err(c_err)?;
    let normed = layer_norm(&target_out, &ln_f_w, Some(&ln_f_b), 1e-6).map_err(c_err)?;
    let logits = match gen_cfg.head_kind {
        LocalHeadKind::UntiedAffine => linear_3d_impl(
            &normed,
            &get(&format!("lm_heads.{level}.weight")).map_err(c_err)?,
            Some(&get(&format!("lm_heads.{level}.bias")).map_err(c_err)?),
            frozen_base,
        )
        .map_err(c_err)?,
        LocalHeadKind::UntiedLinear => linear_3d_impl(
            &normed,
            &get(&format!("lm_heads.{level}.weight")).map_err(c_err)?,
            None,
            frozen_base,
        )
        .map_err(c_err)?,
        LocalHeadKind::SharedHeadBiases => linear_3d_impl(
            &normed,
            &get("lm_head.weight").map_err(c_err)?,
            Some(&get(&format!("head_biases.{level}")).map_err(c_err)?),
            frozen_base,
        )
        .map_err(c_err)?,
    };
    let vocab_l = gen_cfg.vocab_size[level];
    let logits_2d = logits
        .reshape((batch * level_len, vocab_l))
        .map_err(c_err)?;
    candle_nn::loss::cross_entropy(&logits_2d, gt_code).map_err(c_err)
}

#[allow(clippy::too_many_arguments)]
fn generator_forward_loss(
    config: &LocalNcmConfig,
    pre: &GeneratorPrecomputed,
    frozen_wte: &Tensor,
    get: &impl Fn(&str) -> candle_core::Result<Tensor>,
    lora_ctx: Option<(&ParamStore, f64)>,
    prefixes: &[u32],
    gt_codes: &[Tensor],
    cond_latents: &[Option<Tensor>],
    batch: usize,
    device: &Device,
) -> Result<Tensor> {
    let n_levels = config.codec().levels().len();
    let mut total_loss: Option<Tensor> = None;
    for level in 0..n_levels {
        let loss_l = generator_level_loss(
            config,
            pre,
            frozen_wte,
            get,
            lora_ctx,
            prefixes,
            &gt_codes[level],
            cond_latents[level].as_ref(),
            level,
            batch,
            device,
        )?;
        total_loss = Some(match total_loss {
            Some(prev) => (prev + loss_l).map_err(c_err)?,
            None => loss_l,
        });
    }
    (total_loss.expect("nonempty levels") / (n_levels as f64)).map_err(c_err)
}

// ---------------------------------------------------------------------------
// Public stage runners + finite-difference verification helpers
// ---------------------------------------------------------------------------

/// Train a `local-ncm-v1` codec (RVQ + straight-through + EMA codebooks) on CPU or CUDA.
#[allow(clippy::too_many_arguments)]
pub fn train_codec(
    config: &LocalNcmConfig,
    dataset: &TokenDataset,
    init_weights: &TensorMap,
    resume: Option<&LoadedCheckpoint>,
    train_cfg: &TrainConfig,
    device_name: &str,
    checkpoint_dir: Option<&Path>,
) -> Result<(TensorMap, Vec<StepMetrics>)> {
    let device = resolve_device(device_name)?;
    let codec_cfg = config.codec();
    let pre = CodecPrecomputed::new(codec_cfg, &device)?;

    let (start_step, mut rng_state, mut store, mut ema) = if let Some(ckpt) = resume {
        let store = ParamStore::from_host(
            &ckpt.weights,
            Some(&ckpt.optim_m),
            Some(&ckpt.optim_v),
            &device,
        )?;
        let ema = ckpt
            .ema
            .clone()
            .unwrap_or_else(|| EmaCodebookState::from_codebooks(codec_cfg, &ckpt.weights));
        (ckpt.meta.step, ckpt.meta.rng_state, store, ema)
    } else {
        let store = ParamStore::from_host(init_weights, None, None, &device)?;
        let ema = EmaCodebookState::from_codebooks(codec_cfg, init_weights);
        (0usize, splitmix64(train_cfg.seed ^ 0xE4A0_0001), store, ema)
    };

    let mut history = Vec::new();
    for step in start_step..train_cfg.steps {
        let t0 = Instant::now();
        let lr = lr_at_step(
            step,
            train_cfg.steps,
            train_cfg.learning_rate,
            train_cfg.min_lr,
            train_cfg.warmup_steps,
        );
        let (_prefixes, targets) = dataset.sample_batch(train_cfg.batch_size, step, train_cfg.seed);
        let fwd = codec_forward_train(
            codec_cfg,
            &pre,
            &store,
            &targets,
            train_cfg.batch_size,
            train_cfg.commitment_weight,
            true,
            &device,
        )?;
        let loss_val = fwd.total_loss.to_scalar::<f32>().map_err(c_err)? as f64;
        if !loss_val.is_finite() {
            return Err(TrainError::Backend(format!(
                "non-finite codec loss at step {step}"
            )));
        }
        let grads = fwd.total_loss.backward().map_err(c_err)?;
        let grad_norm = store.adamw_step(
            &grads,
            Some("vq.embeddings."),
            step + 1,
            lr,
            train_cfg.weight_decay,
            train_cfg.grad_clip_norm,
        )?;

        // Update codebooks via EMA + Laplace smoothing + dead-code restarts.
        let mut total_restarted = 0usize;
        for (level, (ze_host, ids_host)) in fwd.ema_captures.iter().enumerate() {
            let name = format!("vq.embeddings.{level}.weight");
            let var = store.vars.get(&name).unwrap();
            let shape = var.as_tensor().dims().to_vec();
            let mut book = var
                .as_tensor()
                .flatten_all()
                .map_err(c_err)?
                .to_vec1::<f32>()
                .map_err(c_err)?;
            let stats = ema.update_level(
                level,
                ze_host,
                ids_host,
                &mut book,
                train_cfg.ema_decay as f32,
                train_cfg.ema_eps as f32,
                train_cfg.dead_code_threshold as f32,
                &mut rng_state,
            );
            total_restarted += stats.restarted_codes;
            let updated = Tensor::from_slice(&book, shape.as_slice(), &device).map_err(c_err)?;
            var.set(&updated).map_err(c_err)?;
        }

        let step_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let metric = StepMetrics {
            stage: TrainStage::Codec,
            step: step + 1,
            loss: loss_val,
            recon_loss: Some(fwd.recon_loss),
            commit_loss: Some(fwd.commit_loss),
            grad_norm,
            learning_rate: lr,
            restarted_codes: total_restarted,
            step_ms,
        };
        history.push(metric);

        if let Some(dir) = checkpoint_dir
            && train_cfg.checkpoint_every > 0
            && (step + 1).is_multiple_of(train_cfg.checkpoint_every)
        {
            let weights = store.export_weights()?;
            let (m, v) = store.export_moments()?;
            let meta = CheckpointMeta {
                schema: "ncm-train-checkpoint-v1".into(),
                stage: TrainStage::Codec,
                step: step + 1,
                total_steps: train_cfg.steps,
                rng_state,
                last_loss: loss_val,
                lora_rank: 0,
                lora_alpha: 0.0,
            };
            save_checkpoint(dir, &meta, &weights, &m, &v, Some(&ema))?;
        }
    }

    let final_weights = store.export_weights()?;
    Ok((final_weights, history))
}

/// Train or fine-tune a `local-ncm-v1` generator (full or LoRA) against a frozen codec on CPU or CUDA.
#[allow(clippy::too_many_arguments)]
pub fn train_generator(
    config: &LocalNcmConfig,
    dataset: &TokenDataset,
    frozen_codec_weights: &TensorMap,
    init_gen_weights: &TensorMap,
    resume: Option<&LoadedCheckpoint>,
    train_cfg: &TrainConfig,
    device_name: &str,
    checkpoint_dir: Option<&Path>,
) -> Result<(TensorMap, Vec<StepMetrics>)> {
    let device = resolve_device(device_name)?;
    let pre = GeneratorPrecomputed::new(config, &device)?;
    let n_levels = config.codec().levels().len();
    let inv_levels = 1.0 / (n_levels as f64);

    let mut frozen_codec = BTreeMap::new();
    for (name, (shape, vals)) in frozen_codec_weights {
        let t = Tensor::from_slice(vals, shape.as_slice(), &device).map_err(c_err)?;
        frozen_codec.insert(name.clone(), t);
    }
    let frozen_wte = frozen_codec
        .get("encoder.wte.weight")
        .ok_or_else(|| TrainError::Artifact("missing frozen encoder.wte.weight".into()))?
        .clone();

    let lora_rank = train_cfg.lora_rank;
    let lora_alpha = train_cfg.effective_lora_alpha();

    let (start_step, rng_state, frozen_gen, mut store) = if lora_rank > 0 {
        let mut frozen = BTreeMap::new();
        for (name, (shape, vals)) in init_gen_weights {
            let t = Tensor::from_slice(vals, shape.as_slice(), &device).map_err(c_err)?;
            frozen.insert(name.clone(), t);
        }
        if let Some(ckpt) = resume {
            if ckpt.meta.lora_rank != lora_rank || (ckpt.meta.lora_alpha - lora_alpha).abs() > 1e-9
            {
                return Err(TrainError::Artifact(format!(
                    "checkpoint LoRA config (rank={}, alpha={}) does not match current (rank={}, alpha={})",
                    ckpt.meta.lora_rank, ckpt.meta.lora_alpha, lora_rank, lora_alpha
                )));
            }
            lora::validate_lora_adapters(init_gen_weights, &ckpt.weights, lora_rank)?;
            let store = ParamStore::from_host(
                &ckpt.weights,
                Some(&ckpt.optim_m),
                Some(&ckpt.optim_v),
                &device,
            )?;
            (ckpt.meta.step, ckpt.meta.rng_state, Some(frozen), store)
        } else {
            let adapters = lora::init_lora_adapters(init_gen_weights, lora_rank, train_cfg.seed)?;
            let store = ParamStore::from_host(&adapters, None, None, &device)?;
            (
                0usize,
                splitmix64(train_cfg.seed ^ 0x6E4E_0001),
                Some(frozen),
                store,
            )
        }
    } else if let Some(ckpt) = resume {
        if ckpt.meta.lora_rank != 0 {
            return Err(TrainError::Artifact(format!(
                "cannot resume LoRA checkpoint (rank={}) without --lora-rank {}",
                ckpt.meta.lora_rank, ckpt.meta.lora_rank
            )));
        }
        let store = ParamStore::from_host(
            &ckpt.weights,
            Some(&ckpt.optim_m),
            Some(&ckpt.optim_v),
            &device,
        )?;
        (ckpt.meta.step, ckpt.meta.rng_state, None, store)
    } else {
        let store = ParamStore::from_host(init_gen_weights, None, None, &device)?;
        (
            0usize,
            splitmix64(train_cfg.seed ^ 0x6E4E_0001),
            None,
            store,
        )
    };

    let mut history = Vec::new();
    for step in start_step..train_cfg.steps {
        let t0 = Instant::now();
        let lr = lr_at_step(
            step,
            train_cfg.steps,
            train_cfg.learning_rate,
            train_cfg.min_lr,
            train_cfg.warmup_steps,
        );
        let (prefixes, targets) = dataset.sample_batch(train_cfg.batch_size, step, train_cfg.seed);
        let (gt_codes, cond_latents) = frozen_codec_encode_for_generator(
            config.codec(),
            &pre,
            &frozen_codec,
            &targets,
            train_cfg.batch_size,
            &device,
        )?;

        // Accumulate gradients one level at a time so peak activation memory stays
        // bounded to a single level even for 12-level 89M-param TinyStories.
        let mut accum_grads = BTreeMap::<String, Tensor>::new();
        let mut loss_val = 0.0_f64;
        for level in 0..n_levels {
            let loss_l = if let Some(ref frozen) = frozen_gen {
                let get = |name: &str| {
                    frozen
                        .get(name)
                        .cloned()
                        .ok_or_else(|| candle_core::Error::Msg(format!("missing weight `{name}`")))
                };
                let scale = lora::lora_scale(lora_rank, lora_alpha);
                generator_level_loss(
                    config,
                    &pre,
                    &frozen_wte,
                    &get,
                    Some((&store, scale)),
                    &prefixes,
                    &gt_codes[level],
                    cond_latents[level].as_ref(),
                    level,
                    train_cfg.batch_size,
                    &device,
                )?
            } else {
                let get = |name: &str| store.tensor(name);
                generator_level_loss(
                    config,
                    &pre,
                    &frozen_wte,
                    &get,
                    None,
                    &prefixes,
                    &gt_codes[level],
                    cond_latents[level].as_ref(),
                    level,
                    train_cfg.batch_size,
                    &device,
                )?
            };
            let scaled_loss = (&loss_l * inv_levels).map_err(c_err)?;
            let lv = scaled_loss.to_scalar::<f32>().map_err(c_err)? as f64;
            if !lv.is_finite() {
                return Err(TrainError::Backend(format!(
                    "non-finite generator loss at step {step} level {level}"
                )));
            }
            loss_val += lv;
            let grads = scaled_loss.backward().map_err(c_err)?;
            store.accumulate_grads(&grads, &mut accum_grads)?;
        }

        let grad_norm = store.adamw_step_from_map(
            &accum_grads,
            step + 1,
            lr,
            train_cfg.weight_decay,
            train_cfg.grad_clip_norm,
        )?;
        let step_ms = t0.elapsed().as_secs_f64() * 1000.0;
        history.push(StepMetrics {
            stage: TrainStage::Generator,
            step: step + 1,
            loss: loss_val,
            recon_loss: None,
            commit_loss: None,
            grad_norm,
            learning_rate: lr,
            restarted_codes: 0,
            step_ms,
        });

        if let Some(dir) = checkpoint_dir
            && train_cfg.checkpoint_every > 0
            && (step + 1).is_multiple_of(train_cfg.checkpoint_every)
        {
            let weights = store.export_weights()?;
            let (m, v) = store.export_moments()?;
            let meta = CheckpointMeta {
                schema: "ncm-train-checkpoint-v1".into(),
                stage: TrainStage::Generator,
                step: step + 1,
                total_steps: train_cfg.steps,
                rng_state,
                last_loss: loss_val,
                lora_rank,
                lora_alpha,
            };
            save_checkpoint(dir, &meta, &weights, &m, &v, None)?;
        }
    }

    let final_weights = if lora_rank > 0 {
        let trained_adapters = store.export_weights()?;
        lora::merge_lora_weights(init_gen_weights, &trained_adapters, lora_rank, lora_alpha)?
    } else {
        store.export_weights()?
    };
    Ok((final_weights, history))
}

/// Finite-difference vs analytic gradient check for one named generator parameter element.
/// Returns `(analytic_grad, numeric_grad, abs_diff)`.
#[allow(clippy::too_many_arguments)]
pub fn finite_diff_check_generator(
    config: &LocalNcmConfig,
    frozen_codec_weights: &TensorMap,
    gen_weights: &TensorMap,
    prefixes: &[u32],
    targets: &[u32],
    batch: usize,
    param_name: &str,
    flat_index: usize,
    eps: f32,
) -> Result<(f64, f64, f64)> {
    let device = Device::Cpu;
    let pre = GeneratorPrecomputed::new(config, &device)?;
    let mut frozen_codec = BTreeMap::new();
    for (name, (shape, vals)) in frozen_codec_weights {
        frozen_codec.insert(
            name.clone(),
            Tensor::from_slice(vals, shape.as_slice(), &device).map_err(c_err)?,
        );
    }
    let frozen_wte = frozen_codec.get("encoder.wte.weight").unwrap().clone();
    let (gt_codes, cond_latents) = frozen_codec_encode_for_generator(
        config.codec(),
        &pre,
        &frozen_codec,
        targets,
        batch,
        &device,
    )?;

    // Analytic gradient
    let store = ParamStore::from_host(gen_weights, None, None, &device)?;
    let get = |name: &str| store.tensor(name);
    let loss = generator_forward_loss(
        config,
        &pre,
        &frozen_wte,
        &get,
        None,
        prefixes,
        &gt_codes,
        &cond_latents,
        batch,
        &device,
    )?;
    let grads = loss.backward().map_err(c_err)?;
    let var = store
        .vars
        .get(param_name)
        .ok_or_else(|| TrainError::Config(format!("unknown param `{param_name}`")))?;
    let g_vec = grads
        .get(var.as_tensor())
        .ok_or_else(|| TrainError::Backend(format!("missing grad for `{param_name}`")))?
        .flatten_all()
        .map_err(c_err)?
        .to_vec1::<f32>()
        .map_err(c_err)?;
    let analytic = g_vec[flat_index] as f64;

    // Central finite difference
    let eval_with = |delta: f32| -> Result<f64> {
        let mut perturbed = gen_weights.clone();
        perturbed.get_mut(param_name).unwrap().1[flat_index] += delta;
        let s = ParamStore::from_host(&perturbed, None, None, &device)?;
        let g = |name: &str| s.tensor(name);
        let l = generator_forward_loss(
            config,
            &pre,
            &frozen_wte,
            &g,
            None,
            prefixes,
            &gt_codes,
            &cond_latents,
            batch,
            &device,
        )?;
        Ok(l.to_scalar::<f32>().map_err(c_err)? as f64)
    };
    let l_plus = eval_with(eps)?;
    let l_minus = eval_with(-eps)?;
    let numeric = (l_plus - l_minus) / (2.0 * eps as f64);
    Ok((analytic, numeric, (analytic - numeric).abs()))
}

/// Central finite-difference gradient check for one LoRA adapter (`lora_a` or `lora_b`) element.
/// Returns `(analytic_grad, numeric_grad, abs_diff)`.
#[allow(clippy::too_many_arguments)]
pub fn finite_diff_check_generator_lora(
    config: &LocalNcmConfig,
    frozen_codec_weights: &TensorMap,
    base_gen_weights: &TensorMap,
    lora_adapters: &TensorMap,
    lora_rank: usize,
    lora_alpha: f64,
    prefixes: &[u32],
    targets: &[u32],
    batch: usize,
    adapter_name: &str,
    flat_index: usize,
    eps: f32,
) -> Result<(f64, f64, f64)> {
    let device = Device::Cpu;
    let pre = GeneratorPrecomputed::new(config, &device)?;
    let mut frozen_codec = BTreeMap::new();
    for (name, (shape, vals)) in frozen_codec_weights {
        frozen_codec.insert(
            name.clone(),
            Tensor::from_slice(vals, shape.as_slice(), &device).map_err(c_err)?,
        );
    }
    let frozen_wte = frozen_codec.get("encoder.wte.weight").unwrap().clone();
    let (gt_codes, cond_latents) = frozen_codec_encode_for_generator(
        config.codec(),
        &pre,
        &frozen_codec,
        targets,
        batch,
        &device,
    )?;

    let mut frozen_gen = BTreeMap::new();
    for (name, (shape, vals)) in base_gen_weights {
        frozen_gen.insert(
            name.clone(),
            Tensor::from_slice(vals, shape.as_slice(), &device).map_err(c_err)?,
        );
    }
    let get_base = |name: &str| {
        frozen_gen
            .get(name)
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg(format!("missing weight `{name}`")))
    };
    let scale = lora::lora_scale(lora_rank, lora_alpha);

    // Analytic gradient via per-level accumulation (exact same path as train_generator).
    let store = ParamStore::from_host(lora_adapters, None, None, &device)?;
    let n_levels = config.codec().levels().len();
    let inv_levels = 1.0 / (n_levels as f64);
    let mut accum = BTreeMap::<String, Tensor>::new();
    for level in 0..n_levels {
        let loss_l = generator_level_loss(
            config,
            &pre,
            &frozen_wte,
            &get_base,
            Some((&store, scale)),
            prefixes,
            &gt_codes[level],
            cond_latents[level].as_ref(),
            level,
            batch,
            &device,
        )?;
        let scaled = (&loss_l * inv_levels).map_err(c_err)?;
        let grads = scaled.backward().map_err(c_err)?;
        store.accumulate_grads(&grads, &mut accum)?;
    }
    let g_vec = accum
        .get(adapter_name)
        .ok_or_else(|| TrainError::Backend(format!("missing LoRA grad for `{adapter_name}`")))?
        .flatten_all()
        .map_err(c_err)?
        .to_vec1::<f32>()
        .map_err(c_err)?;
    let analytic = g_vec[flat_index] as f64;

    let eval_with = |delta: f32| -> Result<f64> {
        let mut perturbed = lora_adapters.clone();
        perturbed.get_mut(adapter_name).unwrap().1[flat_index] += delta;
        let s = ParamStore::from_host(&perturbed, None, None, &device)?;
        let l = generator_forward_loss(
            config,
            &pre,
            &frozen_wte,
            &get_base,
            Some((&s, scale)),
            prefixes,
            &gt_codes,
            &cond_latents,
            batch,
            &device,
        )?;
        Ok(l.to_scalar::<f32>().map_err(c_err)? as f64)
    };
    let l_plus = eval_with(eps)?;
    let l_minus = eval_with(-eps)?;
    let numeric = (l_plus - l_minus) / (2.0 * eps as f64);
    Ok((analytic, numeric, (analytic - numeric).abs()))
}

/// Evaluate generator forward loss with optional active LoRA adapters (used to verify
/// exact LoRA-vs-merged forward equivalence).
#[allow(clippy::too_many_arguments)]
pub fn eval_generator_loss(
    config: &LocalNcmConfig,
    frozen_codec_weights: &TensorMap,
    gen_weights: &TensorMap,
    lora_adapters: Option<(&TensorMap, usize, f64)>,
    prefixes: &[u32],
    targets: &[u32],
    batch: usize,
    device_name: &str,
) -> Result<f64> {
    let device = resolve_device(device_name)?;
    let pre = GeneratorPrecomputed::new(config, &device)?;
    let mut frozen_codec = BTreeMap::new();
    for (name, (shape, vals)) in frozen_codec_weights {
        frozen_codec.insert(
            name.clone(),
            Tensor::from_slice(vals, shape.as_slice(), &device).map_err(c_err)?,
        );
    }
    let frozen_wte = frozen_codec.get("encoder.wte.weight").unwrap().clone();
    let (gt_codes, cond_latents) = frozen_codec_encode_for_generator(
        config.codec(),
        &pre,
        &frozen_codec,
        targets,
        batch,
        &device,
    )?;
    let mut base_map = BTreeMap::new();
    for (name, (shape, vals)) in gen_weights {
        base_map.insert(
            name.clone(),
            Tensor::from_slice(vals, shape.as_slice(), &device).map_err(c_err)?,
        );
    }
    let get = |name: &str| {
        base_map
            .get(name)
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg(format!("missing weight `{name}`")))
    };
    let lora_store = match lora_adapters {
        Some((adapters, rank, alpha)) => Some((
            ParamStore::from_host(adapters, None, None, &device)?,
            lora::lora_scale(rank, alpha),
        )),
        None => None,
    };
    let lora_ctx = lora_store.as_ref().map(|(s, sc)| (s, *sc));
    let loss = generator_forward_loss(
        config,
        &pre,
        &frozen_wte,
        &get,
        lora_ctx,
        prefixes,
        &gt_codes,
        &cond_latents,
        batch,
        &device,
    )?;
    Ok(loss.to_scalar::<f32>().map_err(c_err)? as f64)
}
