use crate::{Error, Result};
use serde::Deserialize;
use std::collections::BTreeMap;

/// Serialized upstream codec kwargs. Unknown options are errors, not ignored kwargs.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawCodecConfig {
    pub checkpoint_path: String,
    pub in_vocab_size: usize,
    pub context_length: usize,
    pub embed_dim: usize,
    pub vqvae_vocab_size: Vec<usize>,
    /// Historical upstream name: this is attention head count, NOT depth.
    pub n_layers: usize,
    pub compression_factor: usize,
    pub dropout: f64,
    pub pre_quant_dropout: f64,
    pub pre_quant_groupnorm: usize,
    pub pre_quant_norm_type: String,
    pub use_qk_norm: bool,
    pub dec_layers: usize,
    #[serde(default)]
    pub enc_layers: usize,
    #[serde(default)]
    pub enc_attn_window: usize,
    #[serde(default)]
    pub enc_conv_layers: usize,
    #[serde(default = "three")]
    pub enc_conv_kernel: usize,
    #[serde(default = "yes")]
    pub enc_positional: bool,
    #[serde(default)]
    pub dec_positional: bool,
    pub vector_quantizer_config: QuantizerConfig,
}
fn three() -> usize {
    3
}
fn yes() -> bool {
    true
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizerConfig {
    pub clss: String,
    pub commitment_cost: f64,
    pub decay: f64,
    pub epsilon: f64,
    pub learned_all_sampling: bool,
    pub learned_l1_sampling: bool,
    pub learned_l1_cascade: bool,
    pub learned_l1_cascade_base: usize,
    pub learned_l1_cascade_norm: bool,
    pub learned_l1_cascade_norm_final: bool,
    pub levels: Levels,
    pub quant_resi: Residual,
    pub aux: Auxiliary,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Levels {
    pub use_manual_levels: bool,
    pub manual_levels: Vec<usize>,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Residual {
    pub enabled: bool,
    pub learnable_ratio: bool,
    pub ratio: f64,
    pub share_mode: usize,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Auxiliary {
    pub fine_drop: FineDrop,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FineDrop {
    pub min_keep: usize,
    pub prob: f64,
}

/// Validated, immutable configuration for the initial uncompressed codec family.
/// Bounds are deliberate implementation limits, not claims of text-model support.
#[derive(Clone, Debug)]
pub struct CodecConfig {
    raw: RawCodecConfig,
    strides: Vec<usize>,
    code_length: usize,
}
impl CodecConfig {
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > 1024 * 1024 {
            return Err(Error::Config("config exceeds 1 MiB".into()));
        }
        Self::validate(serde_json::from_slice(bytes)?)
    }
    pub fn validate(raw: RawCodecConfig) -> Result<Self> {
        let q = &raw.vector_quantizer_config;
        let levels = &q.levels.manual_levels;
        let require = |valid, message: &str| {
            if valid {
                Ok(())
            } else {
                Err(Error::Config(message.into()))
            }
        };
        require(
            (2..=65536).contains(&raw.in_vocab_size),
            "vocabulary must be in 2..=65536",
        )?;
        require(
            (2..=256).contains(&raw.context_length) && (8..=256).contains(&raw.embed_dim),
            "length/width outside initial codec bounds",
        )?;
        require(
            raw.n_layers > 0 && raw.embed_dim.is_multiple_of(raw.n_layers),
            "head count must divide width",
        )?;
        require(
            raw.pre_quant_groupnorm > 0 && raw.embed_dim.is_multiple_of(raw.pre_quant_groupnorm),
            "group count must divide width",
        )?;
        require(
            raw.compression_factor == 1
                && raw.enc_layers == 0
                && raw.enc_conv_layers == 0
                && raw.enc_attn_window == 0
                && raw.enc_conv_kernel == 3
                && raw.dec_layers == 1,
            "unsupported encoder/decoder architecture",
        )?;
        require(
            raw.enc_positional
                && !raw.dec_positional
                && raw.use_qk_norm
                && raw.pre_quant_norm_type == "group",
            "unsupported positioning or normalization",
        )?;
        require(
            raw.checkpoint_path == "tokenizer.pt",
            "checkpoint path is metadata only and must be tokenizer.pt",
        )?;
        require(
            [
                raw.dropout,
                raw.pre_quant_dropout,
                q.decay,
                q.aux.fine_drop.prob,
            ]
            .into_iter()
            .all(|x| x.is_finite() && (0.0..1.0).contains(&x)),
            "invalid dropout/decay probability",
        )?;
        require(
            q.commitment_cost.is_finite()
                && q.commitment_cost >= 0.0
                && q.epsilon.is_finite()
                && q.epsilon > 0.0,
            "invalid training metadata",
        )?;
        require(
            q.clss == "multiscale_residual_vector_quantizer"
                && !q.learned_all_sampling
                && q.learned_l1_sampling
                && q.learned_l1_cascade,
            "unsupported quantizer or sampling mode",
        )?;
        require(
            (2..=256).contains(&q.learned_l1_cascade_base),
            "invalid cascade base",
        )?;
        require(
            !(q.learned_l1_cascade_norm || q.learned_l1_cascade_norm_final)
                || raw.embed_dim.is_multiple_of(8),
            "cascade GroupNorm requires eight groups",
        )?;
        require(
            q.levels.use_manual_levels
                && (2..=32).contains(&levels.len())
                && levels.first() == Some(&1)
                && levels.last() == Some(&raw.context_length)
                && levels.windows(2).all(|w| w[0] < w[1]),
            "levels must increase from one to context length",
        )?;
        require(
            raw.vqvae_vocab_size.len() == levels.len()
                && raw.vqvae_vocab_size.iter().all(|n| (2..=65536).contains(n)),
            "invalid per-level codebook sizes",
        )?;
        require(
            q.aux.fine_drop.min_keep > 0 && q.aux.fine_drop.min_keep <= levels.len(),
            "invalid fine-drop minimum",
        )?;
        require(
            q.quant_resi.enabled
                && !q.quant_resi.learnable_ratio
                && q.quant_resi.share_mode == 0
                && q.quant_resi.ratio.is_finite()
                && (0.0..=1.0).contains(&q.quant_resi.ratio)
                && q.quant_resi.ratio > 1e-6,
            "unsupported residual refinement",
        )?;
        let mut remaining = raw.context_length;
        let mut strides = Vec::new();
        while remaining > 1 {
            let mut factor = q.learned_l1_cascade_base.min(remaining);
            while factor > 1 && !remaining.is_multiple_of(factor) {
                factor -= 1;
            }
            if factor <= 1 {
                factor = remaining;
            }
            strides.push(factor);
            remaining /= factor;
        }
        let code_length = levels.iter().sum(); // bounded by 32 * 256 above
        let config = Self {
            raw,
            strides,
            code_length,
        };
        let bytes: usize = config
            .weight_shapes()
            .values()
            .map(|s| s.iter().product::<usize>() * 4)
            .sum();
        require(bytes <= 256 * 1024 * 1024, "codec weights exceed 256 MiB")?;
        Ok(config)
    }
    pub fn raw(&self) -> &RawCodecConfig {
        &self.raw
    }
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    pub fn levels(&self) -> &[usize] {
        &self.raw.vector_quantizer_config.levels.manual_levels
    }
    pub fn code_length(&self) -> usize {
        self.code_length
    }

    /// Exact runtime inventory; training buffers are deliberately absent.
    pub fn weight_shapes(&self) -> BTreeMap<String, Vec<usize>> {
        let r = &self.raw;
        let d = r.embed_dim;
        let mut shapes = BTreeMap::new();
        let mut put = |name: String, shape: Vec<usize>| {
            shapes.insert(name, shape);
        };
        put("encoder.wte.weight".into(), vec![r.in_vocab_size, d]);
        put("encoder.wpe.weight".into(), vec![r.context_length, d]);
        put("decoder.ln_f.weight".into(), vec![d]);
        put(
            "decoder.output_layer.weight".into(),
            vec![r.in_vocab_size, d],
        );
        for layer in ["ln_1", "ln_2"] {
            for suffix in ["weight", "bias"] {
                put(format!("decoder.middle_block.{layer}.{suffix}"), vec![d]);
            }
        }
        for layer in ["q_norm", "k_norm"] {
            put(
                format!("decoder.middle_block.attn.{layer}.weight"),
                vec![d / r.n_layers],
            );
        }
        for (layer, output, input) in [
            ("attn.c_attn", 3 * d, d),
            ("attn.c_proj", d, d),
            ("mlp.c_fc", 4 * d, d),
            ("mlp.c_proj", d, 4 * d),
        ] {
            put(
                format!("decoder.middle_block.{layer}.weight"),
                vec![output, input],
            );
            put(format!("decoder.middle_block.{layer}.bias"), vec![output]);
        }
        for (level, &size) in r.vqvae_vocab_size.iter().enumerate() {
            put(format!("vq.embeddings.{level}.weight"), vec![size, d]);
            put(format!("vq.quant_resi.{level}.weight"), vec![d, d, 3]);
            put(format!("vq.quant_resi.{level}.bias"), vec![d]);
        }
        for (name, strides) in [
            ("down", self.strides.clone()),
            ("up", self.strides.iter().rev().copied().collect()),
        ] {
            for (i, stride) in strides.into_iter().enumerate() {
                let index = if r.vector_quantizer_config.learned_l1_cascade_norm {
                    2 * i
                } else {
                    i
                };
                put(
                    format!("vq.learned_{name}_l1.{index}.weight"),
                    vec![d, d, stride],
                );
                put(format!("vq.learned_{name}_l1.{index}.bias"), vec![d]);
            }
        }
        shapes
    }
    pub fn check_tokens(&self, tokens: &[u32], batch: usize) -> Result<()> {
        self.check_ids(tokens, batch, self.raw.context_length)?;
        if tokens.iter().any(|&x| x as usize >= self.raw.in_vocab_size) {
            return Err(Error::Input("token ID outside vocabulary".into()));
        }
        Ok(())
    }
    pub fn check_codes(&self, codes: &[u32], batch: usize) -> Result<()> {
        self.check_ids(codes, batch, self.code_length)?;
        for row in codes.chunks_exact(self.code_length) {
            let mut offset = 0;
            for (&length, &size) in self.levels().iter().zip(&self.raw.vqvae_vocab_size) {
                if row[offset..offset + length]
                    .iter()
                    .any(|&x| x as usize >= size)
                {
                    return Err(Error::Input("code ID outside level codebook".into()));
                }
                offset += length;
            }
        }
        Ok(())
    }
    fn check_ids(&self, ids: &[u32], batch: usize, length: usize) -> Result<()> {
        // Explicit small-batch bound for this initial implementation, not OOM fallback.
        if !(1..=128).contains(&batch) || batch.checked_mul(length) != Some(ids.len()) {
            return Err(Error::Input(
                "batch/sequence shape is invalid or unsupported".into(),
            ));
        }
        Ok(())
    }
}
