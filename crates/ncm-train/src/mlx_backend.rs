//! Native Apple Silicon Metal (MLX) training backend (`value_and_grad` + AdamW + RVQ/EMA).
#![cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]

use crate::{
    Result, StepMetrics, TrainConfig, TrainError, TrainStage,
    artifact::{CheckpointMeta, LoadedCheckpoint, save_checkpoint},
    data::TokenDataset,
    ema::EmaCodebookState,
    schedule::{
        TensorMap, area1d_matrix, level_attention_mask, linear1d_matrix, lr_at_step, rope_tables,
        splitmix64,
    },
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

fn linear_3d(x: &Array, w: &Array, b: Option<&Array>) -> std::result::Result<Array, Exception> {
    let shape = x.shape();
    let (bsz, seq, d_in) = (shape[0], shape[1], shape[2]);
    let d_out = w.shape()[0];
    let x2 = x.reshape(&[bsz * seq, d_in])?;
    let wt = w.transpose_axes(&[1, 0])?;
    let y = ops::matmul(&x2, &wt)?.reshape(&[bsz, seq, d_out])?;
    match b {
        Some(bias) => y.add(bias),
        None => Ok(y),
    }
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
    let inv = ops::sqrt(&var.add(&f32_scalar(eps))?)?;
    let normed = xc.divide(&inv)?.multiply(w)?;
    match b {
        Some(bias) => normed.add(bias),
        None => Ok(normed),
    }
}

fn rms_norm(x: &Array, w: &Array, eps: f32) -> std::result::Result<Array, Exception> {
    let ms = ops::mean_axis(&ops::square(x)?, -1, true)?;
    let inv = ops::sqrt(&ms.add(&f32_scalar(eps))?)?;
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
    let normed = gc.divide(&ops::sqrt(&var.add(&f32_scalar(eps))?)?)?;
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
        ops::concatenate_axis(&[&zeros, &x.try_index((.., 0..seq - 1, ..))?], 1)?
    } else {
        zeros.clone()
    };
    let x_right = if seq > 1 {
        ops::concatenate_axis(&[&x.try_index((.., 1..seq, ..))?, &zeros], 1)?
    } else {
        zeros
    };
    let x_cat =
        ops::concatenate_axis(&[&x_left, x, &x_right], 2)?.reshape(&[bsz * seq, 3 * c_in])?;
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
    ops::concatenate_axis(&[&o1, &o2], 3)
}

fn attention(
    q: &Array,
    k: &Array,
    v: &Array,
    mask: Option<&Array>,
) -> std::result::Result<Array, Exception> {
    let head_dim = q.shape()[3] as f32;
    let scale = f32_scalar(1.0 / head_dim.sqrt());
    let mut scores = ops::matmul(q, &k.transpose_axes(&[0, 1, 3, 2])?)?.multiply(&scale)?;
    if let Some(m) = mask {
        scores = scores.add(m)?;
    }
    let probs = ops::softmax_axis(&scores, -1, true)?;
    ops::matmul(&probs, v)
}

fn embedding_gather(table: &Array, ids: &Array) -> std::result::Result<Array, Exception> {
    let ids_shape = ids.shape();
    let flat = ids.reshape(&[-1])?;
    let gathered = ops::take_axis(table, &flat, 0)?;
    let d = table.shape()[1];
    let mut out_shape = ids_shape.to_vec();
    out_shape.push(d);
    gathered.reshape(&out_shape)
}

fn squared_distances(x: &Array, book: &Array) -> std::result::Result<Array, Exception> {
    let xx = ops::sum_axis(&ops::square(x)?, 1, true)?;
    let ee = ops::sum_axis(&ops::square(book)?, 1, true)?.transpose_axes(&[1, 0])?;
    let dot = ops::matmul(x, &book.transpose_axes(&[1, 0])?)?.multiply(&f32_scalar(2.0))?;
    xx.add(&ee)?.subtract(&dot)
}

fn cross_entropy_loss(
    logits_2d: &Array,
    targets_1d: &Array,
) -> std::result::Result<Array, Exception> {
    let log_probs = mlx_rs::nn::log_softmax(logits_2d, -1)?;
    let n = targets_1d.shape()[0];
    let idx = targets_1d.reshape(&[n, 1])?;
    let gathered = ops::take_along_axis(&log_probs, &idx, -1)?;
    ops::mean_all(&gathered.negative()?)
}

fn stop_grad(x: &Array) -> std::result::Result<Array, Exception> {
    x.stop_gradient_device(&Stream::gpu())
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
            let s = ops::sum_all(&ops::square(g).map_err(m_err)?).map_err(m_err)?;
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

        for i in 0..self.params.len() {
            if let Some(pfx) = skip_prefix
                && self.names[i].starts_with(pfx)
            {
                continue;
            }
            let g = if clip_scale < 1.0 {
                grads[i].multiply(&f32_scalar(clip_scale)).map_err(m_err)?
            } else {
                grads[i].clone()
            };
            let m_new = self.m[i]
                .multiply(&f32_scalar(beta1))
                .map_err(m_err)?
                .add(&g.multiply(&f32_scalar(1.0 - beta1)).map_err(m_err)?)
                .map_err(m_err)?;
            let v_new = self.v[i]
                .multiply(&f32_scalar(beta2))
                .map_err(m_err)?
                .add(
                    &ops::square(&g)
                        .map_err(m_err)?
                        .multiply(&f32_scalar(1.0 - beta2))
                        .map_err(m_err)?,
                )
                .map_err(m_err)?;
            let m_hat = m_new.divide(&f32_scalar(bc1)).map_err(m_err)?;
            let v_hat = v_new.divide(&f32_scalar(bc2)).map_err(m_err)?;
            let denom = ops::sqrt(&v_hat)
                .map_err(m_err)?
                .add(&f32_scalar(eps))
                .map_err(m_err)?;
            let update = m_hat
                .divide(&denom)
                .map_err(m_err)?
                .multiply(&f32_scalar(lr_f))
                .map_err(m_err)?;

            let name = &self.names[i];
            let apply_wd = wd_f > 0.0
                && self.params[i].ndim() >= 2
                && !name.contains("ln_")
                && !name.contains("q_norm")
                && !name.contains("k_norm");
            let base = if apply_wd {
                self.params[i]
                    .multiply(&f32_scalar(1.0 - lr_f * wd_f))
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
    x.multiply(&f32_scalar(1.0 - ratio))?
        .add(&conv.multiply(&f32_scalar(ratio))?)
}

fn codec_decoder_layer(
    codec_cfg: &CodecConfig,
    get: &impl Fn(&str) -> std::result::Result<Array, Exception>,
    x: &Array,
    name: &str,
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
    let qkv = linear_3d(
        &norm,
        &get(&format!("{name}.attn.c_attn.weight"))?,
        Some(&get(&format!("{name}.attn.c_attn.bias"))?),
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
    let attn = attention(&q, &k, &v, None)?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[bsz, seq, width])?;
    let attn_out = linear_3d(
        &attn,
        &get(&format!("{name}.attn.c_proj.weight"))?,
        Some(&get(&format!("{name}.attn.c_proj.bias"))?),
    )?;
    let res = x.add(&attn_out)?;

    let norm2 = layer_norm(
        &res,
        &get(&format!("{name}.ln_2.weight"))?,
        Some(&get(&format!("{name}.ln_2.bias"))?),
        1e-6,
    )?;
    let h = mlx_rs::nn::gelu(&linear_3d(
        &norm2,
        &get(&format!("{name}.mlp.c_fc.weight"))?,
        Some(&get(&format!("{name}.mlp.c_fc.bias"))?),
    )?)?;
    let mlp = linear_3d(
        &h,
        &get(&format!("{name}.mlp.c_proj.weight"))?,
        Some(&get(&format!("{name}.mlp.c_proj.bias"))?),
    )?;
    res.add(&mlp)
}

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
    args: &[Array],
) -> std::result::Result<(Array, Array, Array, Vec<(Array, Array)>), Exception> {
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
        let code_ids = stop_grad(&ops::argmin_axis(&dists, 1, false)?)?;
        let z_q = ops::take_axis(&book, &code_ids, 0)?.reshape(&[bsz, level_len as i32, width])?;

        ema_pairs.push((stop_grad(&flat_ze)?, code_ids));

        let commit = ops::mean_all(&ops::square(&down.subtract(&z_q)?)?)?;
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
    hidden = codec_decoder_layer(codec_cfg, &get, &hidden, "decoder.middle_block")?;
    for idx in 0..raw.dec_layers - 1 {
        hidden = codec_decoder_layer(
            codec_cfg,
            &get,
            &hidden,
            &format!("decoder.extra_blocks.{idx}"),
        )?;
    }
    let ln_f = layer_norm(&hidden, &get("decoder.ln_f.weight")?, None, 1e-6)?;
    let logits = linear_3d(&ln_f, &get("decoder.output_layer.weight")?, None)?;
    let logits_2d = logits.reshape(&[bsz * ctx, raw.in_vocab_size as i32])?;
    let targets_1d = ids.reshape(&[bsz * ctx])?;
    let recon = cross_entropy_loss(&logits_2d, &targets_1d)?;
    let commit = total_commit.expect("nonempty levels");
    let total = recon.add(&commit.multiply(&f32_scalar(commitment_weight))?)?;
    Ok((total, recon, commit, ema_pairs))
}

/// Train a `local-ncm-v1` codec (RVQ + STE + EMA) on Apple Silicon Metal via MLX `value_and_grad`.
#[allow(clippy::too_many_arguments)]
pub fn train_codec(
    config: &LocalNcmConfig,
    dataset: &TokenDataset,
    init_weights: &TensorMap,
    resume: Option<&LoadedCheckpoint>,
    train_cfg: &TrainConfig,
    checkpoint_dir: Option<&Path>,
) -> Result<(TensorMap, Vec<StepMetrics>)> {
    with_stream(&Stream::gpu(), || {
        let codec_cfg = config.codec();
        let ctx = codec_cfg.raw().context_length;
        let levels = codec_cfg.levels();

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
            let (_prefixes, targets) = dataset.sample_batch(batch, step, train_cfg.seed);
            let ids_arr = from_u32_slice(&targets, &[batch, ctx]);

            let mut all_args = Vec::with_capacity(n_params + const_arrays.len() + 1);
            all_args.extend(store.params.iter().cloned());
            all_args.extend(const_arrays.iter().cloned());
            all_args.push(ids_arr);

            let codec_cfg_ref = codec_cfg;
            let index_ref = &param_index;
            let area_ref = &area_slots;
            let lin_ref = &linear_slots;

            let (vals, grads) = value_and_grad_with_argnums(
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
                        a,
                    )?;
                    Ok(vec![loss])
                },
                &argnums[..],
            )(&all_args)
            .map_err(m_err)?;

            // Also capture recon/commit and EMA stats from the same forward inputs.
            let (_tot, recon_arr, commit_arr, ema_pairs) = codec_loss_functional(
                codec_cfg_ref,
                index_ref,
                n_params,
                area_ref,
                lin_ref,
                ids_slot,
                batch,
                commit_w,
                &all_args,
            )
            .map_err(m_err)?;

            eval(
                std::iter::once(&vals[0])
                    .chain(std::iter::once(&recon_arr))
                    .chain(std::iter::once(&commit_arr))
                    .chain(grads.iter()),
            )
            .map_err(m_err)?;

            let loss_val = vals[0].item_exact::<f32>() as f64;
            let recon_val = recon_arr.item_exact::<f32>() as f64;
            let commit_val = commit_arr.item_exact::<f32>() as f64;
            if !loss_val.is_finite() {
                return Err(TrainError::Backend(format!(
                    "non-finite MLX codec loss at step {step}"
                )));
            }

            let grad_norm = store.adamw_step(
                &grads,
                Some("vq.embeddings."),
                step + 1,
                lr,
                train_cfg.weight_decay,
                train_cfg.grad_clip_norm,
            )?;

            let mut total_restarted = 0usize;
            for (level, (ze_arr, ids_arr)) in ema_pairs.iter().enumerate() {
                let name = format!("vq.embeddings.{level}.weight");
                let idx = store.index[&name];
                let shape: Vec<usize> = store.params[idx]
                    .shape()
                    .iter()
                    .map(|&d| d as usize)
                    .collect();
                let mut book = to_f32_vec(&store.params[idx])?;
                let ze_host = to_f32_vec(ze_arr)?;
                let ids_host = to_u32_vec(ids_arr)?;
                let stats = ema.update_level(
                    level,
                    &ze_host,
                    &ids_host,
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

            let step_ms = t0.elapsed().as_secs_f64() * 1000.0;
            history.push(StepMetrics {
                stage: TrainStage::Codec,
                step: step + 1,
                loss: loss_val,
                recon_loss: Some(recon_val),
                commit_loss: Some(commit_val),
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

fn generator_block(
    gen_cfg: &LocalGeneratorConfig,
    get: &impl Fn(&str) -> std::result::Result<Array, Exception>,
    x: &Array,
    cos: &Array,
    sin: &Array,
    mask: &Array,
    layer: usize,
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
    let qkv = linear_3d(
        &norm1,
        &get(&format!("{p}.attn.c_attn.weight"))?,
        Some(&get(&format!("{p}.attn.c_attn.bias"))?),
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
    let attn = attention(&q, &k, &v, Some(mask))?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[bsz, seq, width])?;
    let attn_out = linear_3d(
        &attn,
        &get(&format!("{p}.attn.c_proj.weight"))?,
        Some(&get(&format!("{p}.attn.c_proj.bias"))?),
    )?;
    let res = x.add(&attn_out)?;

    let norm2 = layer_norm(
        &res,
        &get(&format!("{p}.ln_2.weight"))?,
        Some(&get(&format!("{p}.ln_2.bias"))?),
        1e-6,
    )?;
    let h = mlx_rs::nn::gelu(&linear_3d(
        &norm2,
        &get(&format!("{p}.mlp.c_fc.weight"))?,
        Some(&get(&format!("{p}.mlp.c_fc.bias"))?),
    )?)?;
    let mlp = linear_3d(
        &h,
        &get(&format!("{p}.mlp.c_proj.weight"))?,
        Some(&get(&format!("{p}.mlp.c_proj.bias"))?),
    )?;
    res.add(&mlp)
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
    frozen_codec_weights: &TensorMap,
    init_gen_weights: &TensorMap,
    resume: Option<&LoadedCheckpoint>,
    train_cfg: &TrainConfig,
    checkpoint_dir: Option<&Path>,
) -> Result<(TensorMap, Vec<StepMetrics>)> {
    with_stream(&Stream::gpu(), || {
        let codec_cfg = config.codec();
        let gen_cfg = config.model();
        let ctx = codec_cfg.raw().context_length;
        let width_c = codec_cfg.raw().embed_dim;
        let levels = codec_cfg.levels();
        let n_levels = levels.len();
        let head_dim = gen_cfg.n_embd / gen_cfg.n_head;

        let mut frozen_codec: HashMap<String, Array> =
            HashMap::with_capacity(frozen_codec_weights.len());
        for (name, (shape, vals)) in frozen_codec_weights {
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

        let (start_step, rng_state, mut store) = if let Some(ckpt) = resume {
            let store =
                MlxParamStore::from_host(&ckpt.weights, Some(&ckpt.optim_m), Some(&ckpt.optim_v))?;
            (ckpt.meta.step, ckpt.meta.rng_state, store)
        } else {
            let store = MlxParamStore::from_host(init_gen_weights, None, None)?;
            (0usize, splitmix64(train_cfg.seed ^ 0x6E4E_0001), store)
        };

        let n_params = store.params.len();
        let argnums: Vec<i32> = (0..n_params as i32).collect();
        let param_index = store.index.clone();
        let batch = train_cfg.batch_size;
        let bsz = batch as i32;

        let get_frozen = |name: &str| -> std::result::Result<Array, Exception> {
            frozen_codec
                .get(name)
                .cloned()
                .ok_or_else(|| Exception::custom(format!("missing frozen codec `{name}`")))
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
            let (prefixes, targets) = dataset.sample_batch(batch, step, train_cfg.seed);

            // Frozen codec encode for teacher-forced targets and conditioning latents.
            let target_ids = from_u32_slice(&targets, &[batch, ctx]);
            let mut enc = embedding_gather(&frozen_wte, &target_ids).map_err(m_err)?;
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
                    cond_latents.push(
                        ops::zeros::<f32>(&[bsz, len as i32, width_c as i32]).map_err(m_err)?,
                    );
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
                    ops::argmin_axis(&squared_distances(&flat, &book).map_err(m_err)?, 1, false)
                        .map_err(m_err)?;
                let z_q = ops::take_axis(&book, &code_ids, 0)
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

            let prefix_ids = from_u32_slice(&prefixes, &[batch, gen_cfg.prefix_len]);
            let prefix_raw = embedding_gather(&frozen_wte, &prefix_ids).map_err(m_err)?;
            let step_masks: Vec<Array> = levels
                .iter()
                .enumerate()
                .map(|(l, &len)| {
                    build_combined_mask_arr(
                        &base_masks[l],
                        &prefixes,
                        batch,
                        gen_cfg.prefix_len,
                        len,
                        gen_cfg.pad_token_id,
                    )
                })
                .collect();

            // Pack all non-parameter arrays into args after params:
            // [prefix_raw, gt_codes[0..L], cond_latents[0..L], cos[0..L], sin[0..L], masks[0..L]]
            let mut all_args = Vec::with_capacity(n_params + 1 + 5 * n_levels);
            all_args.extend(store.params.iter().cloned());
            all_args.push(prefix_raw);
            all_args.extend(gt_codes);
            all_args.extend(cond_latents);
            all_args.extend(cos_tables.iter().cloned());
            all_args.extend(sin_tables.iter().cloned());
            all_args.extend(step_masks);

            let index_ref = &param_index;
            let (vals, grads) = value_and_grad_with_argnums(
                |a: &[Array]| -> std::result::Result<Vec<Array>, Exception> {
                    let get = |name: &str| -> std::result::Result<Array, Exception> {
                        let idx = *index_ref.get(name).ok_or_else(|| {
                            Exception::custom(format!("missing gen weight `{name}`"))
                        })?;
                        Ok(a[idx].clone())
                    };
                    let p_raw = &a[n_params];
                    let wte_w = get("transformer.wte.weight")?;
                    let wte_b = get("transformer.wte.bias")?;
                    let prefix_h = linear_3d(p_raw, &wte_w, Some(&wte_b))?;
                    let l0_q = get("l0_queries")?;
                    let wle = get("transformer.wle.weight")?;
                    let ln_f_w = get("transformer.ln_f.weight")?;
                    let ln_f_b = get("transformer.ln_f.bias")?;

                    let mut total: Option<Array> = None;
                    for (l, &len) in levels.iter().enumerate() {
                        let gt = &a[n_params + 1 + l];
                        let cond = &a[n_params + 1 + n_levels + l];
                        let cos = &a[n_params + 1 + 2 * n_levels + l];
                        let sin = &a[n_params + 1 + 3 * n_levels + l];
                        let mask = &a[n_params + 1 + 4 * n_levels + l];

                        let latent = if l == 0 {
                            l0_q.add(cond)?
                        } else {
                            cond.clone()
                        };
                        let lvl_emb = wle.try_index((l as i32..l as i32 + 1, ..))?;
                        let target_h = linear_3d(&latent, &wte_w, Some(&wte_b))?.add(&lvl_emb)?;
                        let mut x = ops::concatenate_axis(&[&prefix_h, &target_h], 1)?;
                        for layer in 0..gen_cfg.n_layer {
                            x = generator_block(gen_cfg, &get, &x, cos, sin, mask, layer)?;
                        }
                        let p_len = gen_cfg.prefix_len as i32;
                        let target_out = x.try_index((.., p_len..p_len + (len as i32), ..))?;
                        let normed = layer_norm(&target_out, &ln_f_w, Some(&ln_f_b), 1e-6)?;
                        let logits = match gen_cfg.head_kind {
                            LocalHeadKind::UntiedAffine => linear_3d(
                                &normed,
                                &get(&format!("lm_heads.{l}.weight"))?,
                                Some(&get(&format!("lm_heads.{l}.bias"))?),
                            )?,
                            LocalHeadKind::UntiedLinear => {
                                linear_3d(&normed, &get(&format!("lm_heads.{l}.weight"))?, None)?
                            }
                            LocalHeadKind::SharedHeadBiases => linear_3d(
                                &normed,
                                &get("lm_head.weight")?,
                                Some(&get(&format!("head_biases.{l}"))?),
                            )?,
                        };
                        let logits_2d =
                            logits.reshape(&[bsz * (len as i32), gen_cfg.vocab_size[l] as i32])?;
                        let loss_l = cross_entropy_loss(&logits_2d, gt)?;
                        total = Some(match total {
                            Some(prev) => prev.add(&loss_l)?,
                            None => loss_l,
                        });
                    }
                    let mean_loss = total
                        .expect("nonempty levels")
                        .divide(&f32_scalar(n_levels as f32))?;
                    Ok(vec![mean_loss])
                },
                &argnums[..],
            )(&all_args)
            .map_err(m_err)?;

            eval(std::iter::once(&vals[0]).chain(grads.iter())).map_err(m_err)?;
            let loss_val = vals[0].item_exact::<f32>() as f64;
            if !loss_val.is_finite() {
                return Err(TrainError::Backend(format!(
                    "non-finite MLX generator loss at step {step}"
                )));
            }
            let grad_norm = store.adamw_step(
                &grads,
                None,
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
                };
                save_checkpoint(dir, &meta, &weights, &m, &v, None)?;
            }
        }

        let final_weights = store.export_weights()?;
        Ok((final_weights, history))
    })
}
