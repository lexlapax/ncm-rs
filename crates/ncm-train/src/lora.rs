//! Low-Rank Adaptation (LoRA) initialization, target selection, and exact export merging.
//!
//! For each targeted generator linear projection `W: [out_dim, in_dim]`, LoRA freezes `W`
//! and trains two low-rank matrices `A: [rank, in_dim]` (Kaiming normal init) and
//! `B: [out_dim, rank]` (zero init) with effective weight `W + (alpha / rank) * (B @ A)`.
//! At export time, adapters are merged directly into `generator.safetensors` so inference
//! runs with zero runtime overhead across `ncm sample`, `ncm run`, and `ncm serve`.

use crate::{
    Result, TrainError,
    schedule::{TensorMap, next_normal_f32, splitmix64},
};

/// Whether a generator tensor name is a LoRA-adapted attention or MLP projection weight.
pub fn is_lora_target(name: &str) -> bool {
    name.starts_with("transformer.blocks.")
        && (name.ends_with(".attn.c_attn.weight")
            || name.ends_with(".attn.c_proj.weight")
            || name.ends_with(".mlp.c_fc.weight")
            || name.ends_with(".mlp.c_proj.weight"))
}

/// Name of the LoRA `A` (`[rank, in_dim]`) adapter tensor for base weight `base`.
pub fn lora_a_name(base: &str) -> String {
    format!("{base}.lora_a")
}

/// Name of the LoRA `B` (`[out_dim, rank]`) adapter tensor for base weight `base`.
pub fn lora_b_name(base: &str) -> String {
    format!("{base}.lora_b")
}

/// Effective LoRA scaling factor `alpha / rank`.
pub fn lora_scale(rank: usize, alpha: f64) -> f64 {
    debug_assert!(rank > 0);
    alpha / (rank as f64)
}

/// Total number of scalar elements in a `TensorMap`.
pub fn count_params(weights: &TensorMap) -> usize {
    weights.values().map(|(_, vals)| vals.len()).sum()
}

/// Deterministically initialize LoRA `A` (Kaiming normal `std = 1 / sqrt(in_dim)`)
/// and `B` (exact zeros) adapters for every targeted projection in `base_weights`.
pub fn init_lora_adapters(base_weights: &TensorMap, rank: usize, seed: u64) -> Result<TensorMap> {
    if rank == 0 {
        return Err(TrainError::Config(
            "LoRA initialization requires rank >= 1".into(),
        ));
    }
    let mut rng = splitmix64(seed ^ 0x10CA_A001_u64);
    let mut adapters = TensorMap::new();
    for (name, (shape, _)) in base_weights {
        if !is_lora_target(name) {
            continue;
        }
        if shape.len() != 2 {
            return Err(TrainError::Config(format!(
                "LoRA target `{name}` must be 2D, got {shape:?}"
            )));
        }
        let (out_dim, in_dim) = (shape[0], shape[1]);
        let std = (1.0 / (in_dim as f64).sqrt()) as f32;
        let mut a_vals = Vec::with_capacity(rank * in_dim);
        for _ in 0..(rank * in_dim) {
            a_vals.push(std * next_normal_f32(&mut rng));
        }
        let b_vals = vec![0.0f32; out_dim * rank];
        adapters.insert(lora_a_name(name), (vec![rank, in_dim], a_vals));
        adapters.insert(lora_b_name(name), (vec![out_dim, rank], b_vals));
    }
    if adapters.is_empty() {
        return Err(TrainError::Config(
            "no LoRA target projections found in generator weights".into(),
        ));
    }
    Ok(adapters)
}

/// Validate that `adapters` contains the exact expected `lora_a` and `lora_b` tensors
/// and shapes for `base_weights` at `rank`.
pub fn validate_lora_adapters(
    base_weights: &TensorMap,
    adapters: &TensorMap,
    rank: usize,
) -> Result<()> {
    let mut expected_count = 0usize;
    for (name, (shape, _)) in base_weights {
        if !is_lora_target(name) {
            continue;
        }
        let (out_dim, in_dim) = (shape[0], shape[1]);
        let a_key = lora_a_name(name);
        let b_key = lora_b_name(name);
        let Some((a_shape, a_vals)) = adapters.get(&a_key) else {
            return Err(TrainError::Artifact(format!(
                "missing LoRA adapter `{a_key}`"
            )));
        };
        let Some((b_shape, b_vals)) = adapters.get(&b_key) else {
            return Err(TrainError::Artifact(format!(
                "missing LoRA adapter `{b_key}`"
            )));
        };
        if a_shape.as_slice() != [rank, in_dim] || a_vals.len() != rank * in_dim {
            return Err(TrainError::Artifact(format!(
                "LoRA adapter `{a_key}` shape {a_shape:?} != expected [{rank}, {in_dim}]"
            )));
        }
        if b_shape.as_slice() != [out_dim, rank] || b_vals.len() != out_dim * rank {
            return Err(TrainError::Artifact(format!(
                "LoRA adapter `{b_key}` shape {b_shape:?} != expected [{out_dim}, {rank}]"
            )));
        }
        expected_count += 2;
    }
    if adapters.len() != expected_count {
        return Err(TrainError::Artifact(format!(
            "LoRA adapter count {} != expected {expected_count}",
            adapters.len()
        )));
    }
    Ok(())
}

/// Merge trained LoRA adapters into `base_weights`:
/// `W_merged = W_base + (alpha / rank) * (B @ A)` (FP64-accumulated, FP32 output).
pub fn merge_lora_weights(
    base_weights: &TensorMap,
    adapters: &TensorMap,
    rank: usize,
    alpha: f64,
) -> Result<TensorMap> {
    if rank == 0 || !alpha.is_finite() || alpha <= 0.0 {
        return Err(TrainError::Config(format!(
            "invalid LoRA merge parameters: rank={rank}, alpha={alpha}"
        )));
    }
    validate_lora_adapters(base_weights, adapters, rank)?;
    let scale = lora_scale(rank, alpha);
    let mut merged = TensorMap::new();
    for (name, (shape, base_vals)) in base_weights {
        if !is_lora_target(name) {
            merged.insert(name.clone(), (shape.clone(), base_vals.clone()));
            continue;
        }
        let (out_dim, in_dim) = (shape[0], shape[1]);
        let (_, a_vals) = &adapters[&lora_a_name(name)];
        let (_, b_vals) = &adapters[&lora_b_name(name)];
        let mut out = base_vals.clone();
        for o in 0..out_dim {
            let b_row = &b_vals[o * rank..(o + 1) * rank];
            let out_row = &mut out[o * in_dim..(o + 1) * in_dim];
            for (i, slot) in out_row.iter_mut().enumerate() {
                let mut acc = 0.0f64;
                for (k, &b_ok) in b_row.iter().enumerate() {
                    acc += f64::from(b_ok) * f64::from(a_vals[k * in_dim + i]);
                }
                let val = (f64::from(*slot) + scale * acc) as f32;
                if !val.is_finite() {
                    return Err(TrainError::Artifact(format!(
                        "non-finite merged LoRA weight in `{name}`"
                    )));
                }
                *slot = val;
            }
        }
        merged.insert(name.clone(), (shape.clone(), out));
    }
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lora_target_filter_selects_only_block_attention_and_mlp_weights() {
        assert!(is_lora_target("transformer.blocks.0.attn.c_attn.weight"));
        assert!(is_lora_target("transformer.blocks.11.attn.c_proj.weight"));
        assert!(is_lora_target("transformer.blocks.3.mlp.c_fc.weight"));
        assert!(is_lora_target("transformer.blocks.7.mlp.c_proj.weight"));

        assert!(!is_lora_target("transformer.blocks.0.attn.c_attn.bias"));
        assert!(!is_lora_target("transformer.blocks.0.ln_1.weight"));
        assert!(!is_lora_target("transformer.blocks.0.attn.q_norm.weight"));
        assert!(!is_lora_target("transformer.wte.weight"));
        assert!(!is_lora_target("lm_heads.0.weight"));
    }

    #[test]
    fn lora_zero_init_merges_identically_and_nonzero_matches_formula() {
        let mut base = TensorMap::new();
        base.insert(
            "transformer.blocks.0.attn.c_proj.weight".into(),
            (vec![4, 3], vec![1.0; 12]),
        );
        base.insert(
            "transformer.blocks.0.attn.c_proj.bias".into(),
            (vec![4], vec![0.5; 4]),
        );

        let adapters = init_lora_adapters(&base, 2, 42).unwrap();
        let merged_zero = merge_lora_weights(&base, &adapters, 2, 4.0).unwrap();
        assert_eq!(
            merged_zero["transformer.blocks.0.attn.c_proj.weight"].1,
            vec![1.0; 12]
        );

        let mut custom = adapters;
        // A: [2, 3] = [[1, 2, 3], [4, 5, 6]], B: [4, 2] = [[1, 0], [0, 1], [1, 1], [2, -1]]
        custom.insert(
            lora_a_name("transformer.blocks.0.attn.c_proj.weight"),
            (vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        );
        custom.insert(
            lora_b_name("transformer.blocks.0.attn.c_proj.weight"),
            (vec![4, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0]),
        );
        // scale = alpha / rank = 4.0 / 2 = 2.0
        let merged = merge_lora_weights(&base, &custom, 2, 4.0).unwrap();
        let w = &merged["transformer.blocks.0.attn.c_proj.weight"].1;
        // Row 0: 1 + 2*[1, 2, 3] = [3, 5, 7]
        assert_eq!(&w[0..3], &[3.0, 5.0, 7.0]);
        // Row 1: 1 + 2*[4, 5, 6] = [9, 11, 13]
        assert_eq!(&w[3..6], &[9.0, 11.0, 13.0]);
        // Row 2: 1 + 2*[5, 7, 9] = [11, 15, 19]
        assert_eq!(&w[6..9], &[11.0, 15.0, 19.0]);
        // Row 3: 1 + 2*[-2, -1, 0] = [-3, -1, 1]
        assert_eq!(&w[9..12], &[-3.0, -1.0, 1.0]);
    }
}
