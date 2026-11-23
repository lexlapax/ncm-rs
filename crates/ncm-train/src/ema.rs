//! Exponential moving average (EMA) codebook updates and dead-code restarts for RVQ training.
use crate::schedule::{TensorMap, next_normal_f32, splitmix64};
use ncm_core::CodecConfig;

/// Per-level EMA statistics for multiscale residual vector quantization codebooks.
#[derive(Debug, Clone)]
pub struct EmaCodebookState {
    pub embed_dim: usize,
    pub vocab_sizes: Vec<usize>,
    pub cluster_size: Vec<Vec<f32>>,
    pub embed_sum: Vec<Vec<f32>>,
}

/// Summary statistics from one EMA codebook update across a level.
#[derive(Debug, Clone, Copy, Default)]
pub struct EmaLevelStats {
    pub active_codes: usize,
    pub restarted_codes: usize,
    pub total_codes: usize,
}

impl EmaCodebookState {
    /// Initialize EMA buffers from the current codebook weights (`cluster_size = 1.0`).
    pub fn from_codebooks(codec_cfg: &CodecConfig, weights: &TensorMap) -> Self {
        let embed_dim = codec_cfg.raw().embed_dim;
        let vocab_sizes = codec_cfg.raw().vqvae_vocab_size.clone();
        let mut cluster_size = Vec::with_capacity(vocab_sizes.len());
        let mut embed_sum = Vec::with_capacity(vocab_sizes.len());
        for (level, &v) in vocab_sizes.iter().enumerate() {
            let name = format!("vq.embeddings.{level}.weight");
            let book = weights
                .get(&name)
                .map(|(_, vals)| vals.clone())
                .unwrap_or_else(|| vec![0.0; v * embed_dim]);
            cluster_size.push(vec![1.0f32; v]);
            embed_sum.push(book);
        }
        Self {
            embed_dim,
            vocab_sizes,
            cluster_size,
            embed_sum,
        }
    }

    /// Serialize EMA buffers into a `TensorMap` for checkpointing.
    pub fn to_tensor_map(&self) -> TensorMap {
        let mut map = TensorMap::new();
        for (level, &v) in self.vocab_sizes.iter().enumerate() {
            map.insert(
                format!("vq.ema_cluster_size_{level}"),
                (vec![v], self.cluster_size[level].clone()),
            );
            map.insert(
                format!("vq.ema_w_{level}"),
                (vec![v, self.embed_dim], self.embed_sum[level].clone()),
            );
        }
        map
    }

    /// Restore EMA buffers from a checkpoint `TensorMap`.
    pub fn from_tensor_map(codec_cfg: &CodecConfig, map: &TensorMap) -> Option<Self> {
        let embed_dim = codec_cfg.raw().embed_dim;
        let vocab_sizes = codec_cfg.raw().vqvae_vocab_size.clone();
        let mut cluster_size = Vec::with_capacity(vocab_sizes.len());
        let mut embed_sum = Vec::with_capacity(vocab_sizes.len());
        for (level, &v) in vocab_sizes.iter().enumerate() {
            let (_, cs) = map.get(&format!("vq.ema_cluster_size_{level}"))?;
            let (_, ew) = map.get(&format!("vq.ema_w_{level}"))?;
            if cs.len() != v || ew.len() != v * embed_dim {
                return None;
            }
            cluster_size.push(cs.clone());
            embed_sum.push(ew.clone());
        }
        Some(Self {
            embed_dim,
            vocab_sizes,
            cluster_size,
            embed_sum,
        })
    }

    /// Apply one EMA update with Laplace smoothing and deterministic dead-code restarts.
    ///
    /// - `flat_ze`: encoder pre-quantization vectors `[N, D]`
    /// - `assigned`: nearest codebook index per row `[N]`
    /// - `codebook`: mutable codebook slice `[V, D]` updated in-place
    #[allow(clippy::too_many_arguments)]
    pub fn update_level(
        &mut self,
        level: usize,
        flat_ze: &[f32],
        assigned: &[u32],
        codebook: &mut [f32],
        decay: f32,
        eps: f32,
        dead_threshold: f32,
        rng_state: &mut u64,
    ) -> EmaLevelStats {
        let v = self.vocab_sizes[level];
        let d = self.embed_dim;
        let n_vecs = assigned.len();
        debug_assert_eq!(flat_ze.len(), n_vecs * d);
        debug_assert_eq!(codebook.len(), v * d);

        let mut counts = vec![0.0f32; v];
        let mut sums = vec![0.0f32; v * d];
        for (row, &code_u32) in assigned.iter().enumerate() {
            let code = code_u32 as usize;
            if code >= v {
                continue;
            }
            counts[code] += 1.0;
            let src = &flat_ze[row * d..(row + 1) * d];
            let dst = &mut sums[code * d..(code + 1) * d];
            for j in 0..d {
                dst[j] += src[j];
            }
        }

        let active_codes = counts.iter().filter(|&&c| c > 0.0).count();
        let cs = &mut self.cluster_size[level];
        let es = &mut self.embed_sum[level];

        let one_minus_decay = 1.0 - decay;
        for k in 0..v {
            cs[k] = decay * cs[k] + one_minus_decay * counts[k];
            let off = k * d;
            for j in 0..d {
                es[off + j] = decay * es[off + j] + one_minus_decay * sums[off + j];
            }
        }

        // Laplace smoothing to prevent division by zero.
        let total_n: f32 = cs.iter().sum();
        let denom_scale = total_n + (v as f32) * eps;
        for (k, &c_val) in cs.iter().enumerate().take(v) {
            let smoothed = if denom_scale > 0.0 {
                ((c_val + eps) / denom_scale) * total_n
            } else {
                1.0
            }
            .max(eps);
            let off = k * d;
            for j in 0..d {
                codebook[off + j] = es[off + j] / smoothed;
            }
        }

        // Dead-code restart from randomly selected encoder vectors in this batch.
        let mut restarted_codes = 0usize;
        if dead_threshold > 0.0 && n_vecs > 0 {
            for (k, c_val) in cs.iter_mut().enumerate() {
                if *c_val < dead_threshold {
                    *rng_state = splitmix64(*rng_state);
                    let pick = (*rng_state as usize) % n_vecs;
                    let src = &flat_ze[pick * d..(pick + 1) * d];
                    let off = k * d;
                    for j in 0..d {
                        let jitter = 1e-4 * next_normal_f32(rng_state);
                        let v_new = src[j] + jitter;
                        codebook[off + j] = v_new;
                        es[off + j] = v_new;
                    }
                    *c_val = 1.0;
                    restarted_codes += 1;
                }
            }
        }

        EmaLevelStats {
            active_codes,
            restarted_codes,
            total_codes: v,
        }
    }
}
