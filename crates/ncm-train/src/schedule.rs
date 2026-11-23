//! Learning-rate scheduling, deterministic initialization, and fixed resampling/RoPE tables.
use ncm_core::{CodecConfig, LocalGeneratorConfig};
use std::collections::BTreeMap;

/// Named host FP32 tensor map (`name -> (shape, row-major values)`).
pub type TensorMap = BTreeMap<String, (Vec<usize>, Vec<f32>)>;

/// SplitMix64 deterministic step.
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Deterministic uniform sample in `(0, 1]` from mutable SplitMix64 state.
pub fn next_uniform_f32(state: &mut u64) -> f32 {
    *state = splitmix64(*state);
    let bits = (*state >> 40) as u32;
    ((bits as f32) + 1.0) / ((1u32 << 24) as f32 + 1.0)
}

/// Deterministic Box-Muller standard normal sample from mutable SplitMix64 state.
pub fn next_normal_f32(state: &mut u64) -> f32 {
    let u1 = next_uniform_f32(state).max(1e-7);
    let u2 = next_uniform_f32(state);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// Linear warmup followed by cosine annealing down to `min_lr`.
pub fn lr_at_step(
    step: usize,
    total_steps: usize,
    base_lr: f64,
    min_lr: f64,
    warmup_steps: usize,
) -> f64 {
    if total_steps == 0 {
        return base_lr;
    }
    if warmup_steps > 0 && step < warmup_steps {
        return base_lr * ((step + 1) as f64 / warmup_steps as f64);
    }
    let decay_steps = total_steps.saturating_sub(warmup_steps).max(1);
    let progress =
        ((step.saturating_sub(warmup_steps)) as f64 / decay_steps as f64).clamp(0.0, 1.0);
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
    min_lr + (base_lr - min_lr) * cosine
}

fn init_tensor(name: &str, shape: &[usize], rng: &mut u64) -> Vec<f32> {
    let count: usize = shape.iter().product();
    let is_norm_weight =
        (name.contains("ln_") || name.contains("q_norm") || name.contains("k_norm"))
            && name.ends_with(".weight");
    let is_bias = name.ends_with(".bias") || name.starts_with("head_biases.");
    let is_codebook = name.starts_with("vq.embeddings.");
    let mut vals = Vec::with_capacity(count);
    for _ in 0..count {
        let v = if is_norm_weight {
            1.0
        } else if is_bias {
            0.0
        } else if is_codebook {
            0.25 * next_normal_f32(rng)
        } else {
            0.02 * next_normal_f32(rng)
        };
        vals.push(v);
    }
    vals
}

/// Deterministically initialize all `local-ncm-v1` codec weights.
pub fn init_codec_weights(codec_cfg: &CodecConfig, seed: u64) -> TensorMap {
    let mut rng = splitmix64(seed ^ 0xC0DE_C001_u64);
    let mut map = TensorMap::new();
    for (name, shape) in codec_cfg.weight_shapes() {
        let vals = init_tensor(&name, &shape, &mut rng);
        map.insert(name, (shape, vals));
    }
    map
}

/// Deterministically initialize all `local-ncm-v1` generator weights.
pub fn init_generator_weights(
    codec_cfg: &CodecConfig,
    gen_cfg: &LocalGeneratorConfig,
    seed: u64,
) -> TensorMap {
    let mut rng = splitmix64(seed ^ 0x6E4E_A701_u64);
    let mut map = TensorMap::new();
    for (name, shape) in gen_cfg.weight_shapes(codec_cfg) {
        let vals = init_tensor(&name, &shape, &mut rng);
        map.insert(name, (shape, vals));
    }
    map
}

/// Exact FP32 matrix `[output_length, input_length]` matching `ncm_backend_cuda::ops::area1d`.
pub fn area1d_matrix(input_length: usize, output_length: usize) -> Vec<f32> {
    let mut mat = vec![0.0f32; output_length * input_length];
    for i in 0..output_length {
        let start = i * input_length / output_length;
        let end = ((i + 1) * input_length).div_ceil(output_length);
        let scale = 1.0 / (end - start) as f32;
        for j in start..end {
            mat[i * input_length + j] = scale;
        }
    }
    mat
}

/// Exact FP32 matrix `[output_length, input_length]` matching `ncm_backend_cuda::ops::linear1d`.
pub fn linear1d_matrix(input_length: usize, output_length: usize) -> Vec<f32> {
    let mut mat = vec![0.0f32; output_length * input_length];
    for i in 0..output_length {
        let source = ((i as f64 + 0.5) * input_length as f64 / output_length as f64 - 0.5).max(0.0);
        let left = (source.floor() as usize).min(input_length - 1);
        let right = (left + 1).min(input_length - 1);
        let weight = (source - left as f64) as f32;
        mat[i * input_length + left] += 1.0 - weight;
        mat[i * input_length + right] += weight;
    }
    mat
}

/// Exact split-half RoPE `(cos, sin)` tables `[seq, head_dim / 2]` matching `ncm-model`.
pub fn rope_tables(
    prefix_len: usize,
    level_len: usize,
    head_dim: usize,
    rope_base: usize,
) -> (Vec<f32>, Vec<f32>) {
    let seq = prefix_len + level_len;
    let half = head_dim / 2;
    let mut cosine = Vec::with_capacity(seq * half);
    let mut sine = Vec::with_capacity(seq * half);
    for pos in (0..prefix_len).chain(0..level_len) {
        for i in 0..half {
            let freq = (pos as f32) / (rope_base as f32).powf(i as f32 / half as f32);
            cosine.push(freq.cos());
            sine.push(freq.sin());
        }
    }
    (cosine, sine)
}

/// Block attention mask `[1, 1, seq, seq]` where prefix tokens cannot attend to target tokens.
pub fn level_attention_mask(prefix_len: usize, level_len: usize) -> Vec<f32> {
    let seq = prefix_len + level_len;
    let mut mask = vec![0.0f32; seq * seq];
    for row in 0..prefix_len {
        for col in prefix_len..seq {
            mask[row * seq + col] = f32::NEG_INFINITY;
        }
    }
    mask
}
