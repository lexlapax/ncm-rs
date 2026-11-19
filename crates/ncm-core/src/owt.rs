//! Config-only description of the pinned upstream OpenWebText (OWT) profile.
//!
//! Upstream pins `config/eval/owt.yaml`, but no OWT checkpoint has been released. This module
//! validates that configuration and records the semantics an implementation must preserve. It
//! never enables execution: [`OwtConfigOnly::require_released_checkpoint`] always fails explicitly.
use crate::{Error, Result};
use serde_json::Value;

/// Validated, non-executable facts from the pinned OWT configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwtConfigOnly {
    /// Code positions per level: the squares 1, 4, ..., 1024.
    pub levels: Vec<usize>,
    pub codebook_size: usize,
    pub generator_layers: usize,
    pub generator_heads: usize,
    pub generator_width: usize,
    pub rope_base: usize,
    pub codec_width: usize,
    pub codec_heads: usize,
    pub codec_vocab: usize,
    /// Upstream generation keeps tokens after the first EOS (`cut_at_eos: false`).
    pub cut_at_eos: bool,
}

impl OwtConfigOnly {
    /// Validate the pinned upstream OWT configuration, converted to JSON.
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let doc: Value = serde_json::from_slice(bytes)?;
        let get = |path: &str| {
            doc.pointer(path)
                .ok_or_else(|| Error::Config(format!("OWT config lacks {path}")))
        };
        let number = |path: &str| -> Result<usize> {
            get(path)?
                .as_u64()
                .and_then(|x| usize::try_from(x).ok())
                .ok_or_else(|| Error::Config(format!("OWT {path} is not a count")))
        };
        let flag = |path: &str| -> Result<bool> {
            get(path)?
                .as_bool()
                .ok_or_else(|| Error::Config(format!("OWT {path} is not a flag")))
        };
        let float = |path: &str| -> Result<f64> {
            get(path)?
                .as_f64()
                .ok_or_else(|| Error::Config(format!("OWT {path} is not a number")))
        };
        let counts = |path: &str| -> Result<Vec<usize>> {
            get(path)?
                .as_array()
                .ok_or_else(|| Error::Config(format!("OWT {path} is not a list")))?
                .iter()
                .map(|x| {
                    x.as_u64()
                        .and_then(|x| usize::try_from(x).ok())
                        .ok_or_else(|| Error::Config(format!("OWT {path} entry")))
                })
                .collect()
        };
        let levels = counts("/tokenizer/vqvae/vector_quantizer_config/levels/manual_levels")?;
        let squares: Vec<usize> = (1..=32).map(|n| n * n).collect();
        let checks = [
            (get("/task")? == "owt", "task"),
            (levels == squares, "level ladder"),
            (
                counts("/tokenizer/vqvae/vqvae_vocab_size")? == [8192; 32],
                "codec codebooks",
            ),
            (
                counts("/model/vocab_size")? == [8192; 31],
                "generator heads",
            ),
            (number("/model/n_layer")? == 16, "generator depth"),
            (number("/model/n_head")? == 16, "generator heads"),
            (number("/model/n_embd")? == 1024, "generator width"),
            (number("/model/context_length")? == 1025, "context length"),
            (number("/model/prefix_len")? == 1, "prefix length"),
            (number("/model/rope_base")? == 100_000, "rotary base"),
            (get("/model/attn_pattern")? == "block_diagonal", "attention"),
            (
                flag("/model/use_rope")?
                    && flag("/model/use_qk_norm")?
                    && flag("/model/use_level_encoding")?
                    && flag("/model/use_positional_encoding")?,
                "generator encodings",
            ),
            (
                !flag("/model/predict_level0")?
                    && !flag("/model/query_level0")?
                    && !flag("/model/ctx_encoder/enabled")?,
                "level-zero source",
            ),
            (
                flag("/model/post_upsample_conv/enabled")?
                    && number("/model/post_upsample_conv/kernel_size")? == 3,
                "post-upsample convolution",
            ),
            (
                flag("/model/tie_heads_to_codebook")?
                    && flag("/model/tie_heads_learn_scale")?
                    && flag("/model/tie_heads_bias")?,
                "codebook-tied heads with scale and bias",
            ),
            (
                get("/tokenizer/pre_tokenizer/model_id")? == "gpt2"
                    && get("/tokenizer/pre_tokenizer/special_tokens/pad_token")? == "<|pad|>",
                "GPT-2 tokenizer with pad",
            ),
            (
                number("/tokenizer/vqvae/in_vocab_size")? == 50304,
                "codec vocabulary",
            ),
            (
                number("/tokenizer/vqvae/context_length")? == 1024,
                "codec length",
            ),
            (number("/tokenizer/vqvae/embed_dim")? == 384, "codec width"),
            (number("/tokenizer/vqvae/n_layers")? == 6, "codec heads"),
            (
                number("/tokenizer/vqvae/pre_quant_groupnorm")? == 8,
                "codec group norm",
            ),
            (
                flag("/tokenizer/vqvae/vector_quantizer_config/learned_l1_cascade")?
                    && number("/tokenizer/vqvae/vector_quantizer_config/learned_l1_cascade_base")?
                        == 4
                    && !flag("/tokenizer/vqvae/vector_quantizer_config/learned_l1_cascade_norm")?,
                "learned level-one cascade",
            ),
            (
                float("/generation/temperature")? == 1.0
                    && number("/generation/top_k")? == 0
                    && float("/generation/top_p")? == 1.0
                    && float("/generation/cfg_scale")? == 0.0
                    && get("/generation/level0_unigram")?
                        == "generator.pt:inference.level0_unigram",
                "sampling settings",
            ),
            (!flag("/generation/cut_at_eos")?, "no EOS truncation"),
        ];
        if let Some((_, name)) = checks.iter().find(|(ok, _)| !ok) {
            return Err(Error::Config(format!(
                "unexpected pinned OWT config: {name}"
            )));
        }
        Ok(Self {
            levels,
            codebook_size: 8192,
            generator_layers: 16,
            generator_heads: 16,
            generator_width: 1024,
            rope_base: 100_000,
            codec_width: 384,
            codec_heads: 6,
            codec_vocab: 50304,
            cut_at_eos: false,
        })
    }

    /// OWT inference is unavailable until upstream releases and this project validates a checkpoint.
    pub fn require_released_checkpoint(&self) -> Result<()> {
        Err(Error::Config(
            "OWT is config-only: no released checkpoint has been validated".into(),
        ))
    }
}
