//! Runtime LoRA adapter overlay uploaded to any [`GeneratorBackend`] (`fp32`, `bf16`, `fp16`, `q8`, `q4`).
//!
//! Pre-scales `B` by `alpha / rank` once on host FP32 before upload so runtime inference
//! adds only two low-rank dense projections (`x @ A^T @ B_scaled^T`) onto each targeted
//! attention/MLP linear layer without duplicating or modifying base weights in GPU memory.

use ncm_core::{Error, GeneratorBackend, Result};
use ncm_io::LoraAdapterArtifact;

/// One uploaded LoRA projection pair `(A, B_scaled)` with `scale = alpha / rank` folded into `B_scaled`.
#[derive(Clone)]
pub struct LoraPair<T> {
    pub a: T,
    pub b_scaled: T,
}

/// Per-transformer-block LoRA deltas matching `GeneratorBlock`.
#[derive(Clone)]
pub struct LoraBlock<T> {
    pub qkv: LoraPair<T>,
    pub attention_projection: LoraPair<T>,
    pub mlp_in: LoraPair<T>,
    pub mlp_out: LoraPair<T>,
}

/// Uploaded runtime LoRA adapter ready to attach to any compatible NCM generator model.
#[derive(Clone)]
pub struct RuntimeLoraAdapter<T> {
    n_layer: usize,
    n_embd: usize,
    rank: usize,
    alpha: f64,
    base_manifest_sha256: String,
    blocks: std::sync::Arc<[LoraBlock<T>]>,
}

impl<T> RuntimeLoraAdapter<T> {
    pub fn n_layer(&self) -> usize {
        self.n_layer
    }

    pub fn n_embd(&self) -> usize {
        self.n_embd
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn base_manifest_sha256(&self) -> &str {
        &self.base_manifest_sha256
    }

    pub(crate) fn blocks(&self) -> &[LoraBlock<T>] {
        &self.blocks
    }
}

impl<T> RuntimeLoraAdapter<T> {
    /// Upload a verified [`LoraAdapterArtifact`] onto `backend` (dense tensors in the backend's
    /// active compute precision: `BF16` for `Q8`/`Q4`/`BF16`, `FP16` for `FP16`, `FP32` for `FP32`).
    pub fn from_artifact<B>(
        backend: &B,
        artifact: &LoraAdapterArtifact,
        expected_n_layer: usize,
        expected_n_embd: usize,
    ) -> Result<Self>
    where
        B: GeneratorBackend<Tensor = T>,
    {
        let m = &artifact.manifest;
        if m.n_layer != expected_n_layer || m.n_embd != expected_n_embd {
            return Err(Error::Config(format!(
                "LoRA adapter dimensions (n_layer={}, n_embd={}) do not match base model ({expected_n_layer}, {expected_n_embd})",
                m.n_layer, m.n_embd
            )));
        }
        let scale = artifact.scale();
        let upload_pair = |base_key: &str| -> Result<LoraPair<T>> {
            let a_key = format!("{base_key}.lora_a");
            let b_key = format!("{base_key}.lora_b");
            let (a_shape, a_vals) = artifact
                .tensors
                .get(&a_key)
                .ok_or_else(|| Error::Weights(format!("missing LoRA tensor `{a_key}`")))?;
            let (b_shape, b_vals) = artifact
                .tensors
                .get(&b_key)
                .ok_or_else(|| Error::Weights(format!("missing LoRA tensor `{b_key}`")))?;
            let mut b_scaled = Vec::with_capacity(b_vals.len());
            for &v in b_vals {
                let s = (f64::from(v) * scale) as f32;
                if !s.is_finite() {
                    return Err(Error::Weights(format!(
                        "non-finite scaled LoRA weight in `{b_key}`"
                    )));
                }
                b_scaled.push(s);
            }
            let a = backend.upload_f32(a_vals, a_shape)?;
            let b_scaled = backend.upload_f32(&b_scaled, b_shape)?;
            Ok(LoraPair { a, b_scaled })
        };

        let mut blocks = Vec::with_capacity(m.n_layer);
        for i in 0..m.n_layer {
            let p = format!("transformer.blocks.{i}");
            blocks.push(LoraBlock {
                qkv: upload_pair(&format!("{p}.attn.c_attn.weight"))?,
                attention_projection: upload_pair(&format!("{p}.attn.c_proj.weight"))?,
                mlp_in: upload_pair(&format!("{p}.mlp.c_fc.weight"))?,
                mlp_out: upload_pair(&format!("{p}.mlp.c_proj.weight"))?,
            });
        }

        Ok(Self {
            n_layer: m.n_layer,
            n_embd: m.n_embd,
            rank: m.rank,
            alpha: m.alpha,
            base_manifest_sha256: m.base_manifest_sha256.clone(),
            blocks: blocks.into(),
        })
    }
}
