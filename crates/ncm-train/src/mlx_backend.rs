//! Native Apple Silicon Metal (MLX) training backend (`value_and_grad` + AdamW + RVQ/EMA).
#![cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]

use crate::{
    Result, StepMetrics, TrainConfig, TrainError, TrainPrecision, TrainStage,
    artifact::{CheckpointMeta, LoadedCheckpoint, save_checkpoint},
    data::TokenDataset,
    ema::EmaCodebookState,
    lora,
    schedule::{
        TensorMap, area1d_matrix, level_attention_mask, linear1d_matrix, lr_at_step, rope_tables,
        splitmix64,
    },
    should_eval_val,
};
use mlx_rs::{
    Array, Dtype, Stream,
    error::Exception,
    ops::{self, indexing::TryIndexOp},
    transforms::{eval, value_and_grad_with_argnums},
    with_stream,
};
use ncm_core::{CodecConfig, LocalGeneratorConfig, LocalHeadKind, LocalNcmConfig};
use std::{collections::HashMap, path::Path, time::Instant};

fn m_err(e: Exception) -> TrainError {
    TrainError::Backend(e.to_string())
}

fn f32_scalar(v: f32) -> Array {
    Array::from_f32(v)
}

fn from_f32_slice(vals: &[f32], shape: &[usize]) -> Array {
    let dims: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
    Array::from_slice(vals, &dims)
}

fn from_u32_slice(vals: &[u32], shape: &[usize]) -> Array {
    let dims: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
    Array::from_slice(vals, &dims)
}

fn to_f32_vec(arr: &Array) -> Result<Vec<f32>> {
    arr.eval().map_err(m_err)?;
    let f = arr.as_dtype(Dtype::Float32).map_err(m_err)?;
    f.eval().map_err(m_err)?;
    f.to_vec_exact::<f32>()
        .map_err(|e| TrainError::Backend(e.to_string()))
}

fn to_u32_vec(arr: &Array) -> Result<Vec<u32>> {
    arr.eval().map_err(m_err)?;
    let u = arr.as_dtype(Dtype::Uint32).map_err(m_err)?;
    u.eval().map_err(m_err)?;
    u.to_vec_exact::<u32>()
        .map_err(|e| TrainError::Backend(e.to_string()))
}

// ---------------------------------------------------------------------------
// Differentiable MLX building blocks (pure functional over &[Array])
// ---------------------------------------------------------------------------

fn linear_3d_prec(
    x: &Array,
    w: &Array,
    b: Option<&Array>,
    bf16: bool,
) -> std::result::Result<Array, Exception> {
    let shape = x.shape();
    let (bsz, seq, d_in) = (shape[0], shape[1], shape[2]);
    let d_out = w.shape()[0];
    let x2 = x.reshape(&[bsz * seq, d_in])?;
    let wt = w.transpose_axes(&[1, 0])?;
    let y2 = if bf16 {
        let xb = if x2.dtype() == Dtype::Bfloat16 {
            x2
        } else {
            x2.as_dtype(Dtype::Bfloat16)?
        };
        let wb = if wt.dtype() == Dtype::Bfloat16 {
            wt
        } else {
            wt.as_dtype(Dtype::Bfloat16)?
        };
        ops::matmul(&xb, &wb)?.as_dtype(Dtype::Float32)?
    } else {
        ops::matmul(&x2, &wt)?
    };
    let y = y2.reshape(&[bsz, seq, d_out])?;
    match b {
        Some(bias) => y.add(bias),
        None => Ok(y),
    }
}

type ArrayLookup<'a> = &'a dyn Fn(&str) -> std::result::Result<Array, Exception>;

fn linear_3d_maybe_lora(
    x: &Array,
    weight_name: &str,
    w: &Array,
    b: Option<&Array>,
    lora_get: Option<ArrayLookup<'_>>,
    lora_scale: f32,
    bf16: bool,
) -> std::result::Result<Array, Exception> {
    let base = linear_3d_prec(x, w, b, bf16)?;
    if let Some(get_lora) = lora_get
        && lora::is_lora_target(weight_name)
    {
        let shape = x.shape();
        let (bsz, seq, d_in) = (shape[0], shape[1], shape[2]);
        let d_out = w.shape()[0];
        let a = get_lora(&lora::lora_a_name(weight_name))?;
        let b_mat = get_lora(&lora::lora_b_name(weight_name))?;
        let x2 = x.reshape(&[bsz * seq, d_in])?;
        let delta = if bf16 {
            let xb = x2.as_dtype(Dtype::Bfloat16)?;
            let ab = a.as_dtype(Dtype::Bfloat16)?;
            let bb = b_mat.as_dtype(Dtype::Bfloat16)?;
            let h = ops::matmul(&xb, &ab.transpose_axes(&[1, 0])?)?;
            ops::matmul(&h, &bb.transpose_axes(&[1, 0])?)?
                .as_dtype(Dtype::Float32)?
                .reshape(&[bsz, seq, d_out])?
                .multiply(f32_scalar(lora_scale))?
        } else {
            let h = ops::matmul(&x2, &a.transpose_axes(&[1, 0])?)?;
            ops::matmul(&h, &b_mat.transpose_axes(&[1, 0])?)?
                .reshape(&[bsz, seq, d_out])?
                .multiply(f32_scalar(lora_scale))?
        };
        return base.add(&delta);
    }
    Ok(base)
}

fn layer_norm(
    x: &Array,
    w: &Array,
    b: Option<&Array>,
    eps: f32,
) -> std::result::Result<Array, Exception> {
    let mean = ops::mean_axis(x, -1, true)?;
    let xc = x.subtract(&mean)?;
    let var = ops::mean_axis(&ops::square(&xc)?, -1, true)?;
    let inv = ops::sqrt(&var.add(f32_scalar(eps))?)?;
    let normed = xc.divide(&inv)?.multiply(w)?;
    match b {
        Some(bias) => normed.add(bias),
        None => Ok(normed),
    }
}

fn rms_norm(x: &Array, w: &Array, eps: f32) -> std::result::Result<Array, Exception> {
    let ms = ops::mean_axis(&ops::square(x)?, -1, true)?;
    let inv = ops::sqrt(&ms.add(f32_scalar(eps))?)?;
    x.divide(&inv)?.multiply(w)
}

fn group_norm_nlc(x: &Array, groups: usize, eps: f32) -> std::result::Result<Array, Exception> {
    let shape = x.shape();
    let (bsz, seq, c) = (shape[0], shape[1], shape[2]);
    let g = groups as i32;
    let cpg = c / g;
    let gx = x
        .reshape(&[bsz, seq, g, cpg])?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[bsz, g, seq * cpg])?;
    let mean = ops::mean_axis(&gx, -1, true)?;
    let gc = gx.subtract(&mean)?;
    let var = ops::mean_axis(&ops::square(&gc)?, -1, true)?;
    let normed = gc.divide(&ops::sqrt(&var.add(f32_scalar(eps))?)?)?;
    normed
        .reshape(&[bsz, g, seq, cpg])?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[bsz, seq, c])
}

fn resample_mat(x: &Array, mat: &Array) -> std::result::Result<Array, Exception> {
    let shape = x.shape();
    let (bsz, t_in, d) = (shape[0], shape[1], shape[2]);
    let t_out = mat.shape()[0];
    let xt = x.transpose_axes(&[0, 2, 1])?.reshape(&[bsz * d, t_in])?;
    let yt = ops::matmul(&xt, &mat.transpose_axes(&[1, 0])?)?;
    yt.reshape(&[bsz, d, t_out])?.transpose_axes(&[0, 2, 1])
}

fn conv1d_stride_eq_k(
    x: &Array,
    w: &Array,
    b: &Array,
    stride: usize,
) -> std::result::Result<Array, Exception> {
    let shape = x.shape();
    let (bsz, seq, c_in) = (shape[0], shape[1], shape[2]);
    let c_out = w.shape()[0];
    let s = stride as i32;
    let out_len = seq / s;
    let x_col = x
        .reshape(&[bsz, out_len, s, c_in])?
        .transpose_axes(&[0, 1, 3, 2])?
        .reshape(&[bsz * out_len, c_in * s])?;
    let w_col = w.reshape(&[c_out, c_in * s])?;
    ops::matmul(&x_col, &w_col.transpose_axes(&[1, 0])?)?
        .reshape(&[bsz, out_len, c_out])?
        .add(b)
}

fn conv_transpose1d_stride_eq_k(
    x: &Array,
    w: &Array,
    b: &Array,
    stride: usize,
) -> std::result::Result<Array, Exception> {
    let shape = x.shape();
    let (bsz, seq, c_in) = (shape[0], shape[1], shape[2]);
    let c_out = w.shape()[1];
    let s = stride as i32;
    let x2 = x.reshape(&[bsz * seq, c_in])?;
    let w2 = w.transpose_axes(&[0, 2, 1])?.reshape(&[c_in, s * c_out])?;
    ops::matmul(&x2, &w2)?
        .reshape(&[bsz, seq * s, c_out])?
        .add(b)
}

fn conv1d_k3_p1(x: &Array, w: &Array, b: &Array) -> std::result::Result<Array, Exception> {
    let shape = x.shape();
    let (bsz, seq, c_in) = (shape[0], shape[1], shape[2]);
    let c_out = w.shape()[0];
    let zeros = ops::zeros::<f32>(&[bsz, 1, c_in])?;
    let x_left = if seq > 1 {
        ops::concatenate(&[&zeros, &x.try_index((.., 0..seq - 1, ..))?], 1)?
    } else {
        zeros.clone()
    };
    let x_right = if seq > 1 {
        ops::concatenate(&[&x.try_index((.., 1..seq, ..))?, &zeros], 1)?
    } else {
        zeros
    };
    let x_cat = ops::concatenate(&[&x_left, x, &x_right], 2)?.reshape(&[bsz * seq, 3 * c_in])?;
    let w_col = w.transpose_axes(&[0, 2, 1])?.reshape(&[c_out, 3 * c_in])?;
    ops::matmul(&x_cat, &w_col.transpose_axes(&[1, 0])?)?
        .reshape(&[bsz, seq, c_out])?
        .add(b)
}

fn apply_rope(x: &Array, cos: &Array, sin: &Array) -> std::result::Result<Array, Exception> {
    let head_dim = x.shape()[3];
    let half = head_dim / 2;
    let x1 = x.try_index((.., .., .., 0..half))?;
    let x2 = x.try_index((.., .., .., half..head_dim))?;
    let seq = cos.shape()[0];
    let c = cos.reshape(&[1, 1, seq, half])?;
    let s = sin.reshape(&[1, 1, seq, half])?;
    let o1 = x1.multiply(&c)?.subtract(&x2.multiply(&s)?)?;
    let o2 = x2.multiply(&c)?.add(&x1.multiply(&s)?)?;
    ops::concatenate(&[&o1, &o2], 3)
}

fn attention(
    q: &Array,
    k: &Array,
    v: &Array,
    mask: Option<&Array>,
    bf16: bool,
) -> std::result::Result<Array, Exception> {
    let head_dim = q.shape()[3] as f32;
    let scale = f32_scalar(1.0 / head_dim.sqrt());
    let kt = k.transpose_axes(&[0, 1, 3, 2])?;
    let mut scores = if bf16 {
        ops::matmul(
            &q.as_dtype(Dtype::Bfloat16)?,
            &kt.as_dtype(Dtype::Bfloat16)?,
        )?
        .as_dtype(Dtype::Float32)?
        .multiply(&scale)?
    } else {
        ops::matmul(q, &kt)?.multiply(&scale)?
    };
    if let Some(m) = mask {
        scores = scores.add(m)?;
    }
    let probs = ops::softmax_axis(&scores, -1, true)?;
    if bf16 {
        ops::matmul(
            &probs.as_dtype(Dtype::Bfloat16)?,
            &v.as_dtype(Dtype::Bfloat16)?,
        )?
        .as_dtype(Dtype::Float32)
    } else {
        ops::matmul(&probs, v)
    }
}

fn embedding_gather(table: &Array, ids: &Array) -> std::result::Result<Array, Exception> {
    let ids_shape = ids.shape();
    let flat = ids.reshape(&[-1])?;
    let gathered = ops::indexing::take_axis(table, &flat, 0)?;
    let d = table.shape()[1];
    let mut out_shape = ids_shape.to_vec();
    out_shape.push(d);
    gathered.reshape(&out_shape)
}

fn squared_distances(x: &Array, book: &Array) -> std::result::Result<Array, Exception> {
    let xx = ops::sum_axis(&ops::square(x)?, 1, true)?;
    let ee = ops::sum_axis(&ops::square(book)?, 1, true)?.transpose_axes(&[1, 0])?;
    let dot = ops::matmul(x, &book.transpose_axes(&[1, 0])?)?.multiply(f32_scalar(2.0))?;
    xx.add(&ee)?.subtract(&dot)
}

fn cross_entropy_loss(
    logits_2d: &Array,
    targets_1d: &Array,
) -> std::result::Result<Array, Exception> {
    let log_probs = mlx_rs::nn::log_softmax(logits_2d, -1)?;
    let n = targets_1d.shape()[0];
    let idx = targets_1d.reshape(&[n, 1])?;
    let gathered = ops::indexing::take_along_axis(&log_probs, &idx, -1)?;
    ops::mean(&gathered.negative()?, false)
}

fn stop_grad(x: &Array) -> std::result::Result<Array, Exception> {
    mlx_rs::stop_gradient(x)
}

// ---------------------------------------------------------------------------
// Indexed parameter store + decoupled AdamW on MLX GPU
// ---------------------------------------------------------------------------

struct MlxParamStore {
    names: Vec<String>,
    index: HashMap<String, usize>,
    params: Vec<Array>,
    m: Vec<Array>,
    v: Vec<Array>,
}

impl MlxParamStore {
    fn from_host(
        weights: &TensorMap,
        optim_m: Option<&TensorMap>,
        optim_v: Option<&TensorMap>,
    ) -> Result<Self> {
        let mut names = Vec::with_capacity(weights.len());
        let mut index = HashMap::with_capacity(weights.len());
        let mut params = Vec::with_capacity(weights.len());
        let mut m = Vec::with_capacity(weights.len());
        let mut v = Vec::with_capacity(weights.len());

        for (i, (name, (shape, vals))) in weights.iter().enumerate() {
            names.push(name.clone());
            index.insert(name.clone(), i);
            params.push(from_f32_slice(vals, shape));
            let m_arr = if let Some(mm) = optim_m
                && let Some((ms, mv)) = mm.get(name)
            {
                from_f32_slice(mv, ms)
            } else {
                let dims: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
                ops::zeros::<f32>(&dims).map_err(m_err)?
            };
            let v_arr = if let Some(vv) = optim_v
                && let Some((vs, val)) = vv.get(name)
            {
                from_f32_slice(val, vs)
            } else {
                let dims: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
                ops::zeros::<f32>(&dims).map_err(m_err)?
            };
            m.push(m_arr);
            v.push(v_arr);
        }
        eval(params.iter().chain(m.iter()).chain(v.iter())).map_err(m_err)?;
        Ok(Self {
            names,
            index,
            params,
            m,
            v,
        })
    }

    fn export_weights(&self) -> Result<TensorMap> {
        let mut map = TensorMap::new();
        for (name, arr) in self.names.iter().zip(&self.params) {
            let shape: Vec<usize> = arr.shape().iter().map(|&d| d as usize).collect();
            let vals = to_f32_vec(arr)?;
            map.insert(name.clone(), (shape, vals));
        }
        Ok(map)
    }

    fn export_moments(&self) -> Result<(TensorMap, TensorMap)> {
        let mut mm = TensorMap::new();
        let mut vv = TensorMap::new();
        for (i, name) in self.names.iter().enumerate() {
            let shape: Vec<usize> = self.params[i].shape().iter().map(|&d| d as usize).collect();
            mm.insert(name.clone(), (shape.clone(), to_f32_vec(&self.m[i])?));
            vv.insert(name.clone(), (shape, to_f32_vec(&self.v[i])?));
        }
        Ok((mm, vv))
    }

    #[allow(clippy::too_many_arguments)]
    fn adamw_step(
        &mut self,
        grads: &[Array],
        skip_prefix: Option<&str>,
        step_one_based: usize,
        lr: f64,
        weight_decay: f64,
        grad_clip_norm: f64,
    ) -> Result<f64> {
        let beta1 = 0.9_f32;
        let beta2 = 0.999_f32;
        let eps = 1e-8_f32;

        let mut sq_sum_arr = f32_scalar(0.0);
        for (i, g) in grads.iter().enumerate() {
            if let Some(pfx) = skip_prefix
                && self.names[i].starts_with(pfx)
            {
                continue;
            }
            let s = ops::sum(&ops::square(g).map_err(m_err)?, false).map_err(m_err)?;
            sq_sum_arr = sq_sum_arr.add(&s).map_err(m_err)?;
        }
        sq_sum_arr.eval().map_err(m_err)?;
        let grad_norm = (sq_sum_arr.item_exact::<f32>() as f64).sqrt();
        let clip_scale = if grad_clip_norm > 0.0 && grad_norm > grad_clip_norm {
            (grad_clip_norm / (grad_norm + 1e-6)) as f32
        } else {
            1.0_f32
        };

        let bc1 = 1.0 - beta1.powi(step_one_based as i32);
        let bc2 = 1.0 - beta2.powi(step_one_based as i32);
        let lr_f = lr as f32;
        let wd_f = weight_decay as f32;

        for (i, g_ref) in grads.iter().enumerate().take(self.params.len()) {
            if let Some(pfx) = skip_prefix
                && self.names[i].starts_with(pfx)
            {
                continue;
            }
            let g = if clip_scale < 1.0 {
                g_ref.multiply(f32_scalar(clip_scale)).map_err(m_err)?
            } else {
                g_ref.clone()
            };
            let m_new = self.m[i]
                .multiply(f32_scalar(beta1))
                .map_err(m_err)?
                .add(&g.multiply(f32_scalar(1.0 - beta1)).map_err(m_err)?)
                .map_err(m_err)?;
            let v_new = self.v[i]
                .multiply(f32_scalar(beta2))
                .map_err(m_err)?
                .add(
                    &ops::square(&g)
                        .map_err(m_err)?
                        .multiply(f32_scalar(1.0 - beta2))
                        .map_err(m_err)?,
                )
                .map_err(m_err)?;
            let m_hat = m_new.divide(f32_scalar(bc1)).map_err(m_err)?;
            let v_hat = v_new.divide(f32_scalar(bc2)).map_err(m_err)?;
            let denom = ops::sqrt(&v_hat)
                .map_err(m_err)?
                .add(f32_scalar(eps))
                .map_err(m_err)?;
            let update = m_hat
                .divide(&denom)
                .map_err(m_err)?
                .multiply(f32_scalar(lr_f))
                .map_err(m_err)?;

            let name = &self.names[i];
            let apply_wd = wd_f > 0.0
                && self.params[i].ndim() >= 2
                && !name.contains("ln_")
                && !name.contains("q_norm")
                && !name.contains("k_norm");
            let base = if apply_wd {
                self.params[i]
                    .multiply(f32_scalar(1.0 - lr_f * wd_f))
                    .map_err(m_err)?
            } else {
                self.params[i].clone()
            };
            self.params[i] = base.subtract(&update).map_err(m_err)?;
            self.m[i] = m_new;
            self.v[i] = v_new;
        }
        eval(self.params.iter().chain(self.m.iter()).chain(self.v.iter())).map_err(m_err)?;
        Ok(grad_norm)
    }
}

// ---------------------------------------------------------------------------
// Codec functional forward
// ---------------------------------------------------------------------------

fn codec_cascade(
    codec_cfg: &CodecConfig,
    get: &impl Fn(&str) -> std::result::Result<Array, Exception>,
    x: &Array,
    up: bool,
) -> std::result::Result<Array, Exception> {
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
        let w = get(&format!("{name}.weight"))?;
        let b = get(&format!("{name}.bias"))?;
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
    get: &impl Fn(&str) -> std::result::Result<Array, Exception>,
    x: &Array,
    level: usize,
) -> std::result::Result<Array, Exception> {
    let ratio = codec_cfg.raw().vector_quantizer_config.quant_resi.ratio as f32;
    let name = format!("vq.quant_resi.{level}");
    let conv = conv1d_k3_p1(
        x,
        &get(&format!("{name}.weight"))?,
        &get(&format!("{name}.bias"))?,
    )?;
    x.multiply(f32_scalar(1.0 - ratio))?
        .add(&conv.multiply(f32_scalar(ratio))?)
}

fn codec_decoder_layer(
    codec_cfg: &CodecConfig,
    get: &impl Fn(&str) -> std::result::Result<Array, Exception>,
    x: &Array,
    name: &str,
    bf16: bool,
) -> std::result::Result<Array, Exception> {
    let shape = x.shape();
    let (bsz, seq, width) = (shape[0], shape[1], shape[2]);
    let heads = codec_cfg.raw().n_layers as i32;
    let head_dim = width / heads;

    let norm = layer_norm(
        x,
        &get(&format!("{name}.ln_1.weight"))?,
        Some(&get(&format!("{name}.ln_1.bias"))?),
        1e-6,
    )?;
    let qkv = linear_3d_prec(
        &norm,
        &get(&format!("{name}.attn.c_attn.weight"))?,
        Some(&get(&format!("{name}.attn.c_attn.bias"))?),
        bf16,
    )?;
    let q = qkv
        .try_index((.., .., 0..width))?
        .reshape(&[bsz, seq, heads, head_dim])?
        .transpose_axes(&[0, 2, 1, 3])?;
    let k = qkv
        .try_index((.., .., width..2 * width))?
        .reshape(&[bsz, seq, heads, head_dim])?
        .transpose_axes(&[0, 2, 1, 3])?;
    let v = qkv
        .try_index((.., .., 2 * width..3 * width))?
        .reshape(&[bsz, seq, heads, head_dim])?
        .transpose_axes(&[0, 2, 1, 3])?;

    let q = rms_norm(&q, &get(&format!("{name}.attn.q_norm.weight"))?, 1e-6)?;
    let k = rms_norm(&k, &get(&format!("{name}.attn.k_norm.weight"))?, 1e-6)?;
    let attn = attention(&q, &k, &v, None, bf16)?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[bsz, seq, width])?;
    let attn_out = linear_3d_prec(
        &attn,
        &get(&format!("{name}.attn.c_proj.weight"))?,
        Some(&get(&format!("{name}.attn.c_proj.bias"))?),
        bf16,
    )?;
    let res = x.add(&attn_out)?;

    let norm2 = layer_norm(
        &res,
        &get(&format!("{name}.ln_2.weight"))?,
        Some(&get(&format!("{name}.ln_2.bias"))?),
        1e-6,
    )?;
    let h = mlx_rs::nn::gelu(&linear_3d_prec(
        &norm2,
        &get(&format!("{name}.mlp.c_fc.weight"))?,
        Some(&get(&format!("{name}.mlp.c_fc.bias"))?),
        bf16,
    )?)?;
    let mlp = linear_3d_prec(
        &h,
        &get(&format!("{name}.mlp.c_proj.weight"))?,
        Some(&get(&format!("{name}.mlp.c_proj.bias"))?),
        bf16,
    )?;
    res.add(&mlp)
}

type CodecFuncOut = (Array, Array, Array, Vec<(Array, Array)>);

#[allow(clippy::too_many_arguments)]
fn codec_loss_functional(
    codec_cfg: &CodecConfig,
    param_index: &HashMap<String, usize>,
    n_params: usize,
    area_slots: &[Option<usize>],
    linear_slots: &[Option<usize>],
    ids_slot: usize,
    batch: usize,
    commitment_weight: f32,
    bf16: bool,
    args: &[Array],
) -> std::result::Result<CodecFuncOut, Exception> {
    let raw = codec_cfg.raw();
    let ctx = raw.context_length as i32;
    let width = raw.embed_dim as i32;
    let bsz = batch as i32;
    let get = |name: &str| -> std::result::Result<Array, Exception> {
        let idx = *param_index
            .get(name)
            .ok_or_else(|| Exception::custom(format!("missing weight `{name}`")))?;
        Ok(args[idx].clone())
    };
    let ids = &args[n_params + ids_slot];

    let mut enc = embedding_gather(&get("encoder.wte.weight")?, ids)?;
    if raw.enc_positional {
        enc = enc.add(&get("encoder.wpe.weight")?)?;
    }
    let mut residual = group_norm_nlc(&enc, raw.pre_quant_groupnorm, 1e-5)?;
    let levels = codec_cfg.levels();
    let mut cumulative: Option<Array> = None;
    let mut total_commit: Option<Array> = None;
    let mut ema_pairs = Vec::with_capacity(levels.len());

    for (level, &level_len) in levels.iter().enumerate() {
        let terminal = level + 1 == levels.len();
        let down = if terminal {
            residual.clone()
        } else if level == 0 {
            codec_cascade(codec_cfg, &get, &residual, false)?
        } else {
            let slot = area_slots[level].expect("area mat slot");
            resample_mat(&residual, &args[n_params + slot])?
        };
        let flat_ze = down.reshape(&[bsz * (level_len as i32), width])?;
        let book = stop_grad(&get(&format!("vq.embeddings.{level}.weight"))?)?;
        let dists = squared_distances(&stop_grad(&flat_ze)?, &book)?;
        let code_ids = stop_grad(&ops::indexing::argmin_axis(&dists, 1, false)?)?;
        let z_q = ops::indexing::take_axis(&book, &code_ids, 0)?.reshape(&[
            bsz,
            level_len as i32,
            width,
        ])?;

        ema_pairs.push((stop_grad(&flat_ze)?, code_ids));

        let commit = ops::mean(&ops::square(&down.subtract(&z_q)?)?, false)?;
        total_commit = Some(match total_commit {
            Some(prev) => prev.add(&commit)?,
            None => commit,
        });

        let z_st = down.add(&stop_grad(&z_q.subtract(&down)?)?)?;
        let up = if level == 0 {
            codec_cascade(codec_cfg, &get, &z_st, true)?
        } else if terminal {
            z_st
        } else {
            let slot = linear_slots[level].expect("linear mat slot");
            resample_mat(&z_st, &args[n_params + slot])?
        };
        let contrib = codec_refine(codec_cfg, &get, &up, level)?;
        let sum = match cumulative {
            Some(ref prev) => prev.add(&contrib)?,
            None => contrib.clone(),
        };
        residual = residual.subtract(&contrib)?;
        cumulative = Some(sum);
    }

    let mut hidden = cumulative.expect("nonempty codec levels");
    if raw.dec_positional {
        hidden = hidden.add(&get("decoder.wpe.weight")?)?;
    }
    hidden = codec_decoder_layer(codec_cfg, &get, &hidden, "decoder.middle_block", bf16)?;
    for idx in 0..raw.dec_layers - 1 {
        hidden = codec_decoder_layer(
            codec_cfg,
            &get,
            &hidden,
            &format!("decoder.extra_blocks.{idx}"),
            bf16,
        )?;
    }
    let ln_f = layer_norm(&hidden, &get("decoder.ln_f.weight")?, None, 1e-6)?;
    let logits = linear_3d_prec(&ln_f, &get("decoder.output_layer.weight")?, None, bf16)?;
    let logits_2d = logits.reshape(&[bsz * ctx, raw.in_vocab_size as i32])?;
    let targets_1d = ids.reshape(&[bsz * ctx])?;
    let recon = cross_entropy_loss(&logits_2d, &targets_1d)?;
    let commit = total_commit.expect("nonempty levels");
    let total = recon.add(&commit.multiply(f32_scalar(commitment_weight))?)?;
    Ok((total, recon, commit, ema_pairs))
}

/// Train a `local-ncm-v1` codec (RVQ + STE + EMA) on Apple Silicon Metal via MLX `value_and_grad`.
#[allow(clippy::too_many_arguments)]
pub fn train_codec(
    config: &LocalNcmConfig,
    dataset: &TokenDataset,
    val_dataset: Option<&TokenDataset>,
    init_weights: &TensorMap,
    resume: Option<&LoadedCheckpoint>,
    train_cfg: &TrainConfig,
    checkpoint_dir: Option<&Path>,
) -> Result<(TensorMap, Vec<StepMetrics>)> {
    with_stream(&Stream::gpu(), || {
        let codec_cfg = config.codec();
        let ctx = codec_cfg.raw().context_length;
        let levels = codec_cfg.levels();
        let bf16 = train_cfg.train_precision == TrainPrecision::Bf16;
        let accum_steps = train_cfg.grad_accum.max(1);
        let inv_accum_f32 = 1.0f32 / (accum_steps as f32);
        let inv_accum = 1.0 / (accum_steps as f64);

        // Build static extra constant arrays (resampling matrices) passed as non-diff args.
        let mut const_arrays: Vec<Array> = Vec::new();
        let mut area_slots = vec![None; levels.len()];
        let mut linear_slots = vec![None; levels.len()];
        for (idx, &len) in levels.iter().enumerate() {
            let terminal = idx + 1 == levels.len();
            if idx > 0 && !terminal {
                area_slots[idx] = Some(const_arrays.len());
                const_arrays.push(from_f32_slice(&area1d_matrix(ctx, len), &[len, ctx]));
                linear_slots[idx] = Some(const_arrays.len());
                const_arrays.push(from_f32_slice(&linear1d_matrix(len, ctx), &[ctx, len]));
            }
        }
        let ids_slot = const_arrays.len();

        let (start_step, mut rng_state, mut store, mut ema) = if let Some(ckpt) = resume {
            let store =
                MlxParamStore::from_host(&ckpt.weights, Some(&ckpt.optim_m), Some(&ckpt.optim_v))?;
            let ema = ckpt
                .ema
                .clone()
                .unwrap_or_else(|| EmaCodebookState::from_codebooks(codec_cfg, &ckpt.weights));
            (ckpt.meta.step, ckpt.meta.rng_state, store, ema)
        } else {
            let store = MlxParamStore::from_host(init_weights, None, None)?;
            let ema = EmaCodebookState::from_codebooks(codec_cfg, init_weights);
            (0usize, splitmix64(train_cfg.seed ^ 0xE4A0_0001), store, ema)
        };

        let n_params = store.params.len();
        let argnums: Vec<i32> = (0..n_params as i32).collect();
        let param_index = store.index.clone();
        let batch = train_cfg.batch_size;
        let commit_w = train_cfg.commitment_weight as f32;

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
            let eff_batch = batch * accum_steps;
            let (_all_prefixes, all_targets) =
                dataset.sample_batch(eff_batch, step, train_cfg.seed);

            let mut accum_grads: Option<Vec<Array>> = None;
            let mut loss_val = 0.0_f64;
            let mut recon_val = 0.0_f64;
            let mut commit_val = 0.0_f64;
            let mut merged_ema: Vec<(Vec<f32>, Vec<u32>)> = (0..levels.len())
                .map(|_| (Vec::new(), Vec::new()))
                .collect();

            let codec_cfg_ref = codec_cfg;
            let index_ref = &param_index;
            let area_ref = &area_slots;
            let lin_ref = &linear_slots;

            for mb in 0..accum_steps {
                let t_slice = &all_targets[mb * batch * ctx..(mb + 1) * batch * ctx];
                let ids_arr = from_u32_slice(t_slice, &[batch, ctx]);
                let mut all_args = Vec::with_capacity(n_params + const_arrays.len() + 1);
                all_args.extend(store.params.iter().cloned());
                all_args.extend(const_arrays.iter().cloned());
                all_args.push(ids_arr);

                let (vals, grads_mb) = value_and_grad_with_argnums(
                    |a: &[Array]| -> std::result::Result<Vec<Array>, Exception> {
                        let (loss, _, _, _) = codec_loss_functional(
                            codec_cfg_ref,
                            index_ref,
                            n_params,
                            area_ref,
                            lin_ref,
                            ids_slot,
                            batch,
                            commit_w,
                            bf16,
                            a,
                        )?;
                        Ok(vec![loss.multiply(f32_scalar(inv_accum_f32))?])
                    },
                    &argnums[..],
                )(&all_args)
                .map_err(m_err)?;

                let (_tot, recon_arr, commit_arr, ema_pairs) = codec_loss_functional(
                    codec_cfg_ref,
                    index_ref,
                    n_params,
                    area_ref,
                    lin_ref,
                    ids_slot,
                    batch,
                    commit_w,
                    bf16,
                    &all_args,
                )
                .map_err(m_err)?;

                eval(
                    std::iter::once(&vals[0])
                        .chain(std::iter::once(&recon_arr))
                        .chain(std::iter::once(&commit_arr))
                        .chain(grads_mb.iter()),
                )
                .map_err(m_err)?;

                let lv = vals[0].item_exact::<f32>() as f64;
                if !lv.is_finite() {
                    return Err(TrainError::Backend(format!(
                        "non-finite MLX codec loss at step {step}"
                    )));
                }
                loss_val += lv;
                recon_val += (recon_arr.item_exact::<f32>() as f64) * inv_accum;
                commit_val += (commit_arr.item_exact::<f32>() as f64) * inv_accum;

                for (level, (ze_arr, id_arr)) in ema_pairs.iter().enumerate() {
                    merged_ema[level].0.extend(to_f32_vec(ze_arr)?);
                    merged_ema[level].1.extend(to_u32_vec(id_arr)?);
                }

                if let Some(ref mut acc) = accum_grads {
                    for (dst, g) in acc.iter_mut().zip(grads_mb) {
                        *dst = dst.add(&g).map_err(m_err)?;
                    }
                    eval(acc.iter()).map_err(m_err)?;
                } else {
                    accum_grads = Some(grads_mb);
                }
            }

            let grads = accum_grads.expect("nonempty grad_accum");
            let grad_norm = store.adamw_step(
                &grads,
                Some("vq.embeddings."),
                step + 1,
                lr,
                train_cfg.weight_decay,
                train_cfg.grad_clip_norm,
            )?;

            let mut total_restarted = 0usize;
            for (level, (ze_host, ids_host)) in merged_ema.iter().enumerate() {
                let name = format!("vq.embeddings.{level}.weight");
                let idx = store.index[&name];
                let shape: Vec<usize> = store.params[idx]
                    .shape()
                    .iter()
                    .map(|&d| d as usize)
                    .collect();
                let mut book = to_f32_vec(&store.params[idx])?;
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
                store.params[idx] = from_f32_slice(&book, &shape);
                store.params[idx].eval().map_err(m_err)?;
            }

            let val_loss = if should_eval_val(
                step + 1,
                train_cfg.steps,
                train_cfg.eval_every,
                val_dataset.is_some(),
            ) {
                let vds = val_dataset.unwrap();
                let mut v_sum = 0.0_f64;
                let mut v_count = 0usize;
                for (_vp, vt, vb) in vds.eval_batches(batch, 16) {
                    let ids_arr = from_u32_slice(&vt, &[vb, ctx]);
                    let mut v_args = Vec::with_capacity(n_params + const_arrays.len() + 1);
                    v_args.extend(store.params.iter().cloned());
                    v_args.extend(const_arrays.iter().cloned());
                    v_args.push(ids_arr);
                    let (v_tot, _, _, _) = codec_loss_functional(
                        codec_cfg_ref,
                        index_ref,
                        n_params,
                        area_ref,
                        lin_ref,
                        ids_slot,
                        vb,
                        commit_w,
                        bf16,
                        &v_args,
                    )
                    .map_err(m_err)?;
                    v_tot.eval().map_err(m_err)?;
                    v_sum += (v_tot.item_exact::<f32>() as f64) * (vb as f64);
                    v_count += vb;
                }
                Some(v_sum / (v_count.max(1) as f64))
            } else {
                None
            };

            let step_ms = t0.elapsed().as_secs_f64() * 1000.0;
            history.push(StepMetrics {
                stage: TrainStage::Codec,
                step: step + 1,
                loss: loss_val,
                recon_loss: Some(recon_val),
                commit_loss: Some(commit_val),
                val_loss,
                grad_norm,
                learning_rate: lr,
                restarted_codes: total_restarted,
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
    })
}

// ---------------------------------------------------------------------------
// Frozen-codec encode + Generator functional forward
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn generator_block(
    gen_cfg: &LocalGeneratorConfig,
    get: &impl Fn(&str) -> std::result::Result<Array, Exception>,
    lora_get: Option<ArrayLookup<'_>>,
    lora_scale: f32,
    x: &Array,
    cos: &Array,
    sin: &Array,
    mask: &Array,
    layer: usize,
    bf16: bool,
) -> std::result::Result<Array, Exception> {
    let shape = x.shape();
    let (bsz, seq, width) = (shape[0], shape[1], shape[2]);
    let heads = gen_cfg.n_head as i32;
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
        lora_get,
        lora_scale,
        bf16,
    )?;
    let q = qkv
        .try_index((.., .., 0..width))?
        .reshape(&[bsz, seq, heads, head_dim])?
        .transpose_axes(&[0, 2, 1, 3])?;
    let k = qkv
        .try_index((.., .., width..2 * width))?
        .reshape(&[bsz, seq, heads, head_dim])?
        .transpose_axes(&[0, 2, 1, 3])?;
    let v = qkv
        .try_index((.., .., 2 * width..3 * width))?
        .reshape(&[bsz, seq, heads, head_dim])?
        .transpose_axes(&[0, 2, 1, 3])?;

    let q = apply_rope(
        &rms_norm(&q, &get(&format!("{p}.attn.q_norm.weight"))?, 1e-6)?,
        cos,
        sin,
    )?;
    let k = apply_rope(
        &rms_norm(&k, &get(&format!("{p}.attn.k_norm.weight"))?, 1e-6)?,
        cos,
        sin,
    )?;
    let attn = attention(&q, &k, &v, Some(mask), bf16)?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[bsz, seq, width])?;
    let c_proj_w_name = format!("{p}.attn.c_proj.weight");
    let attn_out = linear_3d_maybe_lora(
        &attn,
        &c_proj_w_name,
        &get(&c_proj_w_name)?,
        Some(&get(&format!("{p}.attn.c_proj.bias"))?),
        lora_get,
        lora_scale,
        bf16,
    )?;
    let res = x.add(&attn_out)?;

    let norm2 = layer_norm(
        &res,
        &get(&format!("{p}.ln_2.weight"))?,
        Some(&get(&format!("{p}.ln_2.bias"))?),
        1e-6,
    )?;
    let c_fc_w_name = format!("{p}.mlp.c_fc.weight");
    let h = mlx_rs::nn::gelu(&linear_3d_maybe_lora(
        &norm2,
        &c_fc_w_name,
        &get(&c_fc_w_name)?,
        Some(&get(&format!("{p}.mlp.c_fc.bias"))?),
        lora_get,
        lora_scale,
        bf16,
    )?)?;
    let mlp_proj_w_name = format!("{p}.mlp.c_proj.weight");
    let mlp = linear_3d_maybe_lora(
        &h,
        &mlp_proj_w_name,
        &get(&mlp_proj_w_name)?,
        Some(&get(&format!("{p}.mlp.c_proj.bias"))?),
        lora_get,
        lora_scale,
        bf16,
    )?;
    res.add(&mlp)
}

#[allow(clippy::too_many_arguments)]
fn mlx_generator_level_loss(
    gen_cfg: &LocalGeneratorConfig,
    get_base: &impl Fn(&str) -> std::result::Result<Array, Exception>,
    lora_fn: Option<ArrayLookup<'_>>,
    lora_scale_f32: f32,
    p_raw: &Array,
    gt: &Array,
    cond: &Array,
    cos: &Array,
    sin: &Array,
    mask: &Array,
    level: usize,
    level_len: usize,
    bsz: i32,
    bf16: bool,
) -> std::result::Result<Array, Exception> {
    let wte_w = get_base("transformer.wte.weight")?;
    let wte_b = get_base("transformer.wte.bias")?;
    let prefix_h = linear_3d_prec(p_raw, &wte_w, Some(&wte_b), bf16)?;
    let wle = get_base("transformer.wle.weight")?;
    let ln_f_w = get_base("transformer.ln_f.weight")?;
    let ln_f_b = get_base("transformer.ln_f.bias")?;

    let latent = if level == 0 {
        let l0_q = get_base("l0_queries")?;
        l0_q.add(cond)?
    } else {
        cond.clone()
    };
    let lvl_emb = wle.try_index((level as i32..level as i32 + 1, ..))?;
    let target_h = linear_3d_prec(&latent, &wte_w, Some(&wte_b), bf16)?.add(&lvl_emb)?;
    let mut x = ops::concatenate(&[&prefix_h, &target_h], 1)?;
    for layer in 0..gen_cfg.n_layer {
        x = generator_block(
            gen_cfg,
            get_base,
            lora_fn,
            lora_scale_f32,
            &x,
            cos,
            sin,
            mask,
            layer,
            bf16,
        )?;
    }
    let p_len = gen_cfg.prefix_len as i32;
    let target_out = x.try_index((.., p_len..p_len + (level_len as i32), ..))?;
    let normed = layer_norm(&target_out, &ln_f_w, Some(&ln_f_b), 1e-6)?;
    let logits = match gen_cfg.head_kind {
        LocalHeadKind::UntiedAffine => linear_3d_prec(
            &normed,
            &get_base(&format!("lm_heads.{level}.weight"))?,
            Some(&get_base(&format!("lm_heads.{level}.bias"))?),
            bf16,
        )?,
        LocalHeadKind::UntiedLinear => linear_3d_prec(
            &normed,
            &get_base(&format!("lm_heads.{level}.weight"))?,
            None,
            bf16,
        )?,
        LocalHeadKind::SharedHeadBiases => linear_3d_prec(
            &normed,
            &get_base("lm_head.weight")?,
            Some(&get_base(&format!("head_biases.{level}"))?),
            bf16,
        )?,
    };
    let logits_2d =
        logits.reshape(&[bsz * (level_len as i32), gen_cfg.vocab_size[level] as i32])?;
    cross_entropy_loss(&logits_2d, gt)
}

#[allow(clippy::too_many_arguments)]
fn mlx_frozen_codec_encode(
    codec_cfg: &CodecConfig,
    frozen_codec: &HashMap<String, Array>,
    frozen_wte: &Array,
    area_mats: &[Option<Array>],
    linear_mats: &[Option<Array>],
    gen_area_mats: &[Option<Array>],
    targets: &[u32],
    batch: usize,
) -> Result<(Vec<Array>, Vec<Array>)> {
    let ctx = codec_cfg.raw().context_length;
    let width_c = codec_cfg.raw().embed_dim;
    let levels = codec_cfg.levels();
    let n_levels = levels.len();
    let bsz = batch as i32;
    let get_frozen = |name: &str| -> std::result::Result<Array, Exception> {
        frozen_codec
            .get(name)
            .cloned()
            .ok_or_else(|| Exception::custom(format!("missing frozen codec `{name}`")))
    };

    let target_ids = from_u32_slice(targets, &[batch, ctx]);
    let mut enc = embedding_gather(frozen_wte, &target_ids).map_err(m_err)?;
    if codec_cfg.raw().enc_positional {
        enc = enc
            .add(&get_frozen("encoder.wpe.weight").map_err(m_err)?)
            .map_err(m_err)?;
    }
    let mut residual =
        group_norm_nlc(&enc, codec_cfg.raw().pre_quant_groupnorm, 1e-5).map_err(m_err)?;
    let mut cumulative: Option<Array> = None;
    let mut gt_codes = Vec::with_capacity(n_levels);
    let mut cond_latents = Vec::with_capacity(n_levels);

    for (level, &len) in levels.iter().enumerate() {
        let terminal = level + 1 == n_levels;
        if level == 0 {
            cond_latents
                .push(ops::zeros::<f32>(&[bsz, len as i32, width_c as i32]).map_err(m_err)?);
        } else {
            let cum = cumulative.as_ref().expect("cumulative latent");
            let cond = if len == ctx {
                cum.clone()
            } else {
                resample_mat(cum, gen_area_mats[level].as_ref().unwrap()).map_err(m_err)?
            };
            cond_latents.push(cond);
        }
        let down = if terminal {
            residual.clone()
        } else if level == 0 {
            codec_cascade(codec_cfg, &get_frozen, &residual, false).map_err(m_err)?
        } else {
            resample_mat(&residual, area_mats[level].as_ref().unwrap()).map_err(m_err)?
        };
        let flat = down
            .reshape(&[bsz * (len as i32), width_c as i32])
            .map_err(m_err)?;
        let book = get_frozen(&format!("vq.embeddings.{level}.weight")).map_err(m_err)?;
        let code_ids =
            ops::indexing::argmin_axis(&squared_distances(&flat, &book).map_err(m_err)?, 1, false)
                .map_err(m_err)?;
        let z_q = ops::indexing::take_axis(&book, &code_ids, 0)
            .map_err(m_err)?
            .reshape(&[bsz, len as i32, width_c as i32])
            .map_err(m_err)?;
        let up = if level == 0 {
            codec_cascade(codec_cfg, &get_frozen, &z_q, true).map_err(m_err)?
        } else if terminal {
            z_q
        } else {
            resample_mat(&z_q, linear_mats[level].as_ref().unwrap()).map_err(m_err)?
        };
        let contrib = codec_refine(codec_cfg, &get_frozen, &up, level).map_err(m_err)?;
        let sum = match cumulative {
            Some(ref prev) => prev.add(&contrib).map_err(m_err)?,
            None => contrib.clone(),
        };
        residual = residual.subtract(&contrib).map_err(m_err)?;
        cumulative = Some(sum);
        gt_codes.push(code_ids);
    }
    Ok((gt_codes, cond_latents))
}

fn build_combined_mask_arr(
    base_mask: &Array,
    prefixes: &[u32],
    batch: usize,
    prefix_len: usize,
    level_len: usize,
    pad_id: Option<u32>,
) -> Array {
    let Some(pad) = pad_id else {
        return base_mask.clone();
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
    from_f32_slice(&mask, &[batch, 1, seq, seq])
}

/// Train or fine-tune a `local-ncm-v1` generator against a frozen codec on Apple Silicon Metal.
#[allow(clippy::too_many_arguments)]
pub fn train_generator(
    config: &LocalNcmConfig,
    dataset: &TokenDataset,
    val_dataset: Option<&TokenDataset>,
    frozen_codec_weights: &TensorMap,
    init_gen_weights: &TensorMap,
    resume: Option<&LoadedCheckpoint>,
    train_cfg: &TrainConfig,
    checkpoint_dir: Option<&Path>,
) -> Result<(TensorMap, Vec<StepMetrics>)> {
    let (weights, _adapters, history) = train_generator_with_adapters(
        config,
        dataset,
        val_dataset,
        frozen_codec_weights,
        init_gen_weights,
        resume,
        train_cfg,
        checkpoint_dir,
        true,
    )?;
    Ok((weights, history))
}

/// Train or fine-tune a generator on Metal and optionally return unmerged LoRA adapters
/// alongside (or instead of) merged weights.
#[allow(clippy::too_many_arguments)]
pub fn train_generator_with_adapters(
    config: &LocalNcmConfig,
    dataset: &TokenDataset,
    val_dataset: Option<&TokenDataset>,
    frozen_codec_weights: &TensorMap,
    init_gen_weights: &TensorMap,
    resume: Option<&LoadedCheckpoint>,
    train_cfg: &TrainConfig,
    checkpoint_dir: Option<&Path>,
    merge_lora: bool,
) -> Result<(TensorMap, Option<TensorMap>, Vec<StepMetrics>)> {
    with_stream(&Stream::gpu(), || {
        let codec_cfg = config.codec();
        let gen_cfg = config.model();
        let ctx = codec_cfg.raw().context_length;
        let levels = codec_cfg.levels();
        let n_levels = levels.len();
        let head_dim = gen_cfg.n_embd / gen_cfg.n_head;
        let bf16 = train_cfg.train_precision == TrainPrecision::Bf16;
        let accum_steps = train_cfg.grad_accum.max(1);
        let prefix_len = gen_cfg.prefix_len;

        let mut frozen_codec: HashMap<String, Array> =
            HashMap::with_capacity(frozen_codec_weights.len());
        for (name, (shape, vals)) in frozen_codec_weights {
            if name.starts_with("decoder.") {
                continue;
            }
            frozen_codec.insert(name.clone(), from_f32_slice(vals, shape));
        }
        eval(frozen_codec.values()).map_err(m_err)?;
        let frozen_wte = frozen_codec
            .get("encoder.wte.weight")
            .ok_or_else(|| TrainError::Artifact("missing frozen encoder.wte.weight".into()))?
            .clone();

        let mut area_mats = Vec::with_capacity(n_levels);
        let mut linear_mats = Vec::with_capacity(n_levels);
        let mut gen_area_mats = Vec::with_capacity(n_levels);
        let mut cos_tables = Vec::with_capacity(n_levels);
        let mut sin_tables = Vec::with_capacity(n_levels);
        let mut base_masks = Vec::with_capacity(n_levels);

        for (idx, &len) in levels.iter().enumerate() {
            let terminal = idx + 1 == n_levels;
            if idx > 0 && !terminal {
                area_mats.push(Some(from_f32_slice(&area1d_matrix(ctx, len), &[len, ctx])));
                linear_mats.push(Some(from_f32_slice(
                    &linear1d_matrix(len, ctx),
                    &[ctx, len],
                )));
            } else {
                area_mats.push(None);
                linear_mats.push(None);
            }
            if idx > 0 {
                gen_area_mats.push(Some(from_f32_slice(&area1d_matrix(ctx, len), &[len, ctx])));
            } else {
                gen_area_mats.push(None);
            }
            let seq = gen_cfg.prefix_len + len;
            let (c, s) = rope_tables(gen_cfg.prefix_len, len, head_dim, gen_cfg.rope_base);
            let m = level_attention_mask(gen_cfg.prefix_len, len);
            cos_tables.push(from_f32_slice(&c, &[seq, head_dim / 2]));
            sin_tables.push(from_f32_slice(&s, &[seq, head_dim / 2]));
            base_masks.push(from_f32_slice(&m, &[1, 1, seq, seq]));
        }

        let lora_rank = train_cfg.lora_rank;
        let lora_alpha = train_cfg.effective_lora_alpha();
        let lora_scale_f32 = if lora_rank > 0 {
            lora::lora_scale(lora_rank, lora_alpha) as f32
        } else {
            0.0f32
        };

        let (start_step, rng_state, frozen_gen, mut store) = if lora_rank > 0 {
            let mut frozen = HashMap::with_capacity(init_gen_weights.len());
            for (name, (shape, vals)) in init_gen_weights {
                let arr = from_f32_slice(vals, shape);
                let arr = if bf16 && shape.len() == 2 && name != "transformer.wle.weight" {
                    arr.as_dtype(Dtype::Bfloat16).map_err(m_err)?
                } else {
                    arr
                };
                frozen.insert(name.clone(), arr);
            }
            eval(frozen.values()).map_err(m_err)?;
            if let Some(ckpt) = resume {
                if ckpt.meta.lora_rank != lora_rank
                    || (ckpt.meta.lora_alpha - lora_alpha).abs() > 1e-9
                {
                    return Err(TrainError::Artifact(format!(
                        "checkpoint LoRA config (rank={}, alpha={}) does not match current (rank={}, alpha={})",
                        ckpt.meta.lora_rank, ckpt.meta.lora_alpha, lora_rank, lora_alpha
                    )));
                }
                lora::validate_lora_adapters(init_gen_weights, &ckpt.weights, lora_rank)?;
                let store = MlxParamStore::from_host(
                    &ckpt.weights,
                    Some(&ckpt.optim_m),
                    Some(&ckpt.optim_v),
                )?;
                (ckpt.meta.step, ckpt.meta.rng_state, Some(frozen), store)
            } else {
                let adapters =
                    lora::init_lora_adapters(init_gen_weights, lora_rank, train_cfg.seed)?;
                let store = MlxParamStore::from_host(&adapters, None, None)?;
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
            let store =
                MlxParamStore::from_host(&ckpt.weights, Some(&ckpt.optim_m), Some(&ckpt.optim_v))?;
            (ckpt.meta.step, ckpt.meta.rng_state, None, store)
        } else {
            let store = MlxParamStore::from_host(init_gen_weights, None, None)?;
            (
                0usize,
                splitmix64(train_cfg.seed ^ 0x6E4E_0001),
                None,
                store,
            )
        };

        let n_params = store.params.len();
        let argnums: Vec<i32> = (0..n_params as i32).collect();
        let param_index = store.index.clone();
        let batch = train_cfg.batch_size;
        let bsz = batch as i32;
        let inv_scale_f32 = 1.0f32 / ((n_levels * accum_steps) as f32);

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
            let eff_batch = batch * accum_steps;
            let (all_prefixes, all_targets) = dataset.sample_batch(eff_batch, step, train_cfg.seed);

            // Accumulate gradients across `grad_accum` micro-batches and one level at a time.
            let mut accum_grads: Option<Vec<Array>> = None;
            let mut loss_val = 0.0_f64;
            let index_ref = &param_index;
            let frozen_gen_ref = frozen_gen.as_ref();

            for mb in 0..accum_steps {
                let p_slice = &all_prefixes[mb * batch * prefix_len..(mb + 1) * batch * prefix_len];
                let t_slice = &all_targets[mb * batch * ctx..(mb + 1) * batch * ctx];
                let (gt_codes, cond_latents) = mlx_frozen_codec_encode(
                    codec_cfg,
                    &frozen_codec,
                    &frozen_wte,
                    &area_mats,
                    &linear_mats,
                    &gen_area_mats,
                    t_slice,
                    batch,
                )?;
                let prefix_ids = from_u32_slice(p_slice, &[batch, prefix_len]);
                let prefix_raw = embedding_gather(&frozen_wte, &prefix_ids).map_err(m_err)?;
                eval(
                    std::iter::once(&prefix_raw)
                        .chain(gt_codes.iter())
                        .chain(cond_latents.iter()),
                )
                .map_err(m_err)?;

                for (l, &len) in levels.iter().enumerate() {
                    let mask_l = build_combined_mask_arr(
                        &base_masks[l],
                        p_slice,
                        batch,
                        prefix_len,
                        len,
                        gen_cfg.pad_token_id,
                    );
                    let mut level_args = Vec::with_capacity(n_params + 6);
                    level_args.extend(store.params.iter().cloned());
                    level_args.push(prefix_raw.clone());
                    level_args.push(gt_codes[l].clone());
                    level_args.push(cond_latents[l].clone());
                    level_args.push(cos_tables[l].clone());
                    level_args.push(sin_tables[l].clone());
                    level_args.push(mask_l);

                    let (vals, grads_l) = value_and_grad_with_argnums(
                        |a: &[Array]| -> std::result::Result<Vec<Array>, Exception> {
                            let get_base = |name: &str| -> std::result::Result<Array, Exception> {
                                if let Some(fg) = frozen_gen_ref {
                                    fg.get(name).cloned().ok_or_else(|| {
                                        Exception::custom(format!(
                                            "missing frozen gen weight `{name}`"
                                        ))
                                    })
                                } else {
                                    let idx = *index_ref.get(name).ok_or_else(|| {
                                        Exception::custom(format!("missing gen weight `{name}`"))
                                    })?;
                                    Ok(a[idx].clone())
                                }
                            };
                            let get_lora = |name: &str| -> std::result::Result<Array, Exception> {
                                let idx = *index_ref.get(name).ok_or_else(|| {
                                    Exception::custom(format!("missing LoRA weight `{name}`"))
                                })?;
                                Ok(a[idx].clone())
                            };
                            let lora_fn: Option<ArrayLookup<'_>> =
                                if lora_rank > 0 { Some(&get_lora) } else { None };
                            let loss_l = mlx_generator_level_loss(
                                gen_cfg,
                                &get_base,
                                lora_fn,
                                lora_scale_f32,
                                &a[n_params],
                                &a[n_params + 1],
                                &a[n_params + 2],
                                &a[n_params + 3],
                                &a[n_params + 4],
                                &a[n_params + 5],
                                l,
                                len,
                                bsz,
                                bf16,
                            )?
                            .multiply(f32_scalar(inv_scale_f32))?;
                            Ok(vec![loss_l])
                        },
                        &argnums[..],
                    )(&level_args)
                    .map_err(m_err)?;

                    eval(std::iter::once(&vals[0]).chain(grads_l.iter())).map_err(m_err)?;
                    let lv = vals[0].item_exact::<f32>() as f64;
                    if !lv.is_finite() {
                        return Err(TrainError::Backend(format!(
                            "non-finite MLX generator loss at step {step} mb {mb} level {l}"
                        )));
                    }
                    loss_val += lv;

                    if let Some(ref mut acc) = accum_grads {
                        for (dst, g) in acc.iter_mut().zip(grads_l) {
                            *dst = dst.add(&g).map_err(m_err)?;
                        }
                        eval(acc.iter()).map_err(m_err)?;
                    } else {
                        accum_grads = Some(grads_l);
                    }
                }
            }

            let grads = accum_grads.expect("nonempty levels");
            let grad_norm = store.adamw_step(
                &grads,
                None,
                step + 1,
                lr,
                train_cfg.weight_decay,
                train_cfg.grad_clip_norm,
            )?;

            let val_loss = if should_eval_val(
                step + 1,
                train_cfg.steps,
                train_cfg.eval_every,
                val_dataset.is_some(),
            ) {
                let vds = val_dataset.unwrap();
                let mut v_sum = 0.0_f64;
                let mut v_count = 0usize;
                for (vp, vt, vb) in vds.eval_batches(batch, 16) {
                    let (vgt, vcond) = mlx_frozen_codec_encode(
                        codec_cfg,
                        &frozen_codec,
                        &frozen_wte,
                        &area_mats,
                        &linear_mats,
                        &gen_area_mats,
                        &vt,
                        vb,
                    )?;
                    let v_pids = from_u32_slice(&vp, &[vb, prefix_len]);
                    let v_praw = embedding_gather(&frozen_wte, &v_pids).map_err(m_err)?;
                    let get_base = |name: &str| -> std::result::Result<Array, Exception> {
                        if let Some(fg) = frozen_gen_ref {
                            fg.get(name).cloned().ok_or_else(|| {
                                Exception::custom(format!("missing frozen gen weight `{name}`"))
                            })
                        } else {
                            let idx = *index_ref.get(name).ok_or_else(|| {
                                Exception::custom(format!("missing gen weight `{name}`"))
                            })?;
                            Ok(store.params[idx].clone())
                        }
                    };
                    let get_lora = |name: &str| -> std::result::Result<Array, Exception> {
                        let idx = *index_ref.get(name).ok_or_else(|| {
                            Exception::custom(format!("missing LoRA weight `{name}`"))
                        })?;
                        Ok(store.params[idx].clone())
                    };
                    let lora_fn: Option<ArrayLookup<'_>> =
                        if lora_rank > 0 { Some(&get_lora) } else { None };
                    let mut batch_loss = 0.0_f64;
                    for (l, &len) in levels.iter().enumerate() {
                        let vmask = build_combined_mask_arr(
                            &base_masks[l],
                            &vp,
                            vb,
                            prefix_len,
                            len,
                            gen_cfg.pad_token_id,
                        );
                        let loss_l = mlx_generator_level_loss(
                            gen_cfg,
                            &get_base,
                            lora_fn,
                            lora_scale_f32,
                            &v_praw,
                            &vgt[l],
                            &vcond[l],
                            &cos_tables[l],
                            &sin_tables[l],
                            &vmask,
                            l,
                            len,
                            vb as i32,
                            bf16,
                        )
                        .map_err(m_err)?;
                        loss_l.eval().map_err(m_err)?;
                        batch_loss += loss_l.item_exact::<f32>() as f64;
                    }
                    v_sum += (batch_loss / (n_levels as f64)) * (vb as f64);
                    v_count += vb;
                }
                Some(v_sum / (v_count.max(1) as f64))
            } else {
                None
            };

            let step_ms = t0.elapsed().as_secs_f64() * 1000.0;
            history.push(StepMetrics {
                stage: TrainStage::Generator,
                step: step + 1,
                loss: loss_val,
                recon_loss: None,
                commit_loss: None,
                val_loss,
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

        let (final_weights, raw_adapters) = if lora_rank > 0 {
            let trained_adapters = store.export_weights()?;
            let merged = if merge_lora {
                lora::merge_lora_weights(
                    init_gen_weights,
                    &trained_adapters,
                    lora_rank,
                    lora_alpha,
                )?
            } else {
                TensorMap::new()
            };
            (merged, Some(trained_adapters))
        } else {
            (store.export_weights()?, None)
        };
        Ok((final_weights, raw_adapters, history))
    })
}
