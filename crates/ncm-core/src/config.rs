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
    #[serde(default)]
    pub dropout: f64,
    #[serde(default)]
    pub pre_quant_dropout: f64,
    pub pre_quant_groupnorm: usize,
    #[serde(default = "group")]
    pub pre_quant_norm_type: String,
    pub use_qk_norm: bool,
    #[serde(default = "one")]
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
fn one() -> usize {
    1
}
fn group() -> String {
    "group".into()
}
fn yes() -> bool {
    true
}
fn default_epsilon() -> f64 {
    1e-5
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizerConfig {
    pub clss: String,
    #[serde(default)]
    pub commitment_cost: f64,
    #[serde(default)]
    pub decay: f64,
    #[serde(default = "default_epsilon")]
    pub epsilon: f64,
    pub learned_all_sampling: bool,
    pub learned_l1_sampling: bool,
    pub learned_l1_cascade: bool,
    pub learned_l1_cascade_base: usize,
    pub learned_l1_cascade_norm: bool,
    pub learned_l1_cascade_norm_final: bool,
    /// Training-only upstream flag; the supported inference profile requires false.
    #[serde(default)]
    pub ema_scale_by_world_size: bool,
    pub levels: Levels,
    pub quant_resi: Residual,
    #[serde(default)]
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
#[derive(Clone, Debug, Default, Deserialize)]
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
impl Default for FineDrop {
    fn default() -> Self {
        Self {
            min_keep: 1,
            prob: 0.0,
        }
    }
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
        // The historical Sudoku JSON schema requires these fields and does not
        // admit TinyStories' training-only EMA flag. Keep that boundary even
        // though the shared wire struct also represents TinyStories defaults.
        let value: serde_json::Value = serde_json::from_slice(bytes)?;
        let q = value.get("vector_quantizer_config");
        if value.get("pre_quant_norm_type").is_none()
            || value.get("dec_layers").is_none()
            || value.get("pre_quant_dropout").is_none()
            || q.and_then(|q| q.get("commitment_cost")).is_none()
            || q.and_then(|q| q.get("decay")).is_none()
            || q.and_then(|q| q.get("epsilon")).is_none()
            || q.and_then(|q| q.get("aux")).is_none()
            || q.and_then(|q| q.get("ema_scale_by_world_size")).is_some()
        {
            return Err(Error::Config("unsupported Sudoku codec JSON schema".into()));
        }
        Self::validate(serde_json::from_slice(bytes)?)
    }
    pub fn validate(raw: RawCodecConfig) -> Result<Self> {
        Self::validate_with_width(raw, 256, 1, 65536, 256 * 1024 * 1024, false)
    }
    /// Explicit pinned TinyStories codec profile; does not widen Sudoku bounds.
    pub fn validate_tinystories(raw: RawCodecConfig) -> Result<Self> {
        let expected_levels = [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128];
        if raw.in_vocab_size != 8192
            || raw.context_length != 128
            || raw.embed_dim != 384
            || raw.n_layers != 6
            || raw.vqvae_vocab_size != [4096; 12]
            || raw.vector_quantizer_config.levels.manual_levels != expected_levels
        {
            return Err(Error::Config(
                "unsupported TinyStories codec profile".into(),
            ));
        }
        Self::validate_with_width(raw, 384, 1, 65536, 256 * 1024 * 1024, false)
    }
    /// Explicit pinned XSum codec profile with a four-position initial level.
    pub fn validate_xsum(raw: RawCodecConfig) -> Result<Self> {
        let expected_levels = [4, 6, 10, 14, 20, 26, 32, 40, 48];
        if raw.in_vocab_size != 50304
            || raw.context_length != 48
            || raw.embed_dim != 384
            || raw.n_layers != 6
            || raw.vqvae_vocab_size != [4096; 9]
            || raw.vector_quantizer_config.levels.manual_levels != expected_levels
            || !raw.vector_quantizer_config.learned_l1_cascade_norm
            || !raw.vector_quantizer_config.learned_l1_cascade_norm_final
        {
            return Err(Error::Config("unsupported XSum codec profile".into()));
        }
        Self::validate_with_width(raw, 384, 4, 65536, 256 * 1024 * 1024, false)
    }
    /// Host-only semantic configuration for the pinned WMT14 codec. This does
    /// not enable codec execution or widen the existing profiles' bounds.
    pub fn validate_wmt14(raw: RawCodecConfig) -> Result<Self> {
        let expected_levels = [1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64];
        if raw.in_vocab_size != 151680
            || raw.context_length != 64
            || raw.embed_dim != 384
            || raw.n_layers != 6
            || raw.pre_quant_groupnorm != 8
            || raw.dropout != 0.1
            || raw.pre_quant_dropout != 0.25
            || raw.vqvae_vocab_size != [4096; 11]
            || raw.vector_quantizer_config.levels.manual_levels != expected_levels
            || raw.vector_quantizer_config.learned_l1_cascade_base != 4
            || !raw.vector_quantizer_config.learned_l1_cascade_norm
            || !raw.vector_quantizer_config.learned_l1_cascade_norm_final
            || raw.vector_quantizer_config.quant_resi.ratio != 0.5
            || raw.vector_quantizer_config.decay != 0.99
            || raw.vector_quantizer_config.epsilon != 1e-5
            || raw.vector_quantizer_config.commitment_cost != 0.25
            || raw.vector_quantizer_config.aux.fine_drop.prob != 0.5
            || raw.vector_quantizer_config.aux.fine_drop.min_keep != 1
        {
            return Err(Error::Config("unsupported WMT14 codec profile".into()));
        }
        Self::validate_with_width(raw, 384, 1, 151680, 640 * 1024 * 1024, false)
    }
    /// Experimental semantic profile for the pinned LM1B-small codec.
    pub fn validate_lm1b_small(raw: RawCodecConfig) -> Result<Self> {
        let levels = [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128];
        if raw.in_vocab_size != 30528
            || raw.context_length != 128
            || raw.embed_dim != 384
            || raw.n_layers != 6
            || raw.pre_quant_groupnorm != 4
            || raw.dec_layers != 4
            || raw.enc_positional
            || !raw.dec_positional
            || raw.vqvae_vocab_size != [8192; 12]
            || raw.vector_quantizer_config.levels.manual_levels != levels
            || raw.vector_quantizer_config.learned_l1_cascade
            || raw.vector_quantizer_config.learned_l1_cascade_norm
            || raw.vector_quantizer_config.learned_l1_cascade_norm_final
            || raw.vector_quantizer_config.quant_resi.ratio != 0.5
        {
            return Err(Error::Config("unsupported LM1B-small codec profile".into()));
        }
        Self::validate_with_width(raw, 384, 1, 65536, 448 * 1024 * 1024, true)
    }
    /// Generic bounded codec validation for `local-ncm-v1` user checkpoints.
    /// Does not relax or alter any pinned upstream profile bounds.
    pub fn validate_local(raw: RawCodecConfig) -> Result<Self> {
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
            (2..=262_144).contains(&raw.in_vocab_size),
            "local codec vocabulary outside bounds (2..=262144)",
        )?;
        require(
            (2..=1024).contains(&raw.context_length) && (8..=2048).contains(&raw.embed_dim),
            "local codec length/width outside bounds",
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
                && (1..=8).contains(&raw.dec_layers),
            "unsupported local codec encoder/decoder architecture",
        )?;
        require(
            raw.use_qk_norm && raw.pre_quant_norm_type == "group",
            "local codec requires QK norm and group pre-quant norm",
        )?;
        require(
            !raw.checkpoint_path.is_empty() && raw.checkpoint_path.len() <= 256,
            "local codec checkpoint_path must be a bounded string",
        )?;
        require(
            [raw.dropout, raw.pre_quant_dropout, q.decay]
                .into_iter()
                .all(|x| x.is_finite() && (0.0..1.0).contains(&x))
                && q.aux.fine_drop.prob.is_finite()
                && (0.0..=1.0).contains(&q.aux.fine_drop.prob),
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
                && !q.ema_scale_by_world_size,
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
        let first_level = *levels
            .first()
            .ok_or_else(|| Error::Config("local codec levels must be non-empty".into()))?;
        require(
            q.levels.use_manual_levels
                && (2..=32).contains(&levels.len())
                && first_level > 0
                && raw.context_length.is_multiple_of(first_level)
                && levels.last() == Some(&raw.context_length)
                && levels.windows(2).all(|w| w[0] < w[1]),
            "levels must increase from a divisor of context_length to context_length",
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
        let mut remaining = raw.context_length / first_level;
        let mut strides = Vec::new();
        if q.learned_l1_cascade {
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
        } else {
            strides.push(remaining);
        }
        let code_length: usize = levels.iter().sum();
        let config = Self {
            raw,
            strides,
            code_length,
        };
        let bytes: usize = config
            .weight_shapes()
            .values()
            .try_fold(0usize, |acc, s| {
                let elems = s.iter().try_fold(1usize, |p, &d| p.checked_mul(d))?;
                acc.checked_add(elems.checked_mul(4)?)
            })
            .ok_or_else(|| Error::Config("local codec weight size overflow".into()))?;
        require(
            bytes <= 2 * 1024 * 1024 * 1024,
            "local codec weights exceed 2 GiB limit",
        )?;
        Ok(config)
    }
    fn validate_with_width(
        raw: RawCodecConfig,
        max_width: usize,
        first_level: usize,
        max_vocab: usize,
        max_weight_bytes: usize,
        lm1b: bool,
    ) -> Result<Self> {
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
            (2..=max_vocab).contains(&raw.in_vocab_size),
            "vocabulary outside profile bounds",
        )?;
        require(
            (2..=256).contains(&raw.context_length) && (8..=max_width).contains(&raw.embed_dim),
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
                && raw.dec_layers == if lm1b { 4 } else { 1 },
            "unsupported encoder/decoder architecture",
        )?;
        require(
            raw.enc_positional != lm1b
                && raw.dec_positional == lm1b
                && raw.use_qk_norm
                && raw.pre_quant_norm_type == "group",
            "unsupported positioning or normalization",
        )?;
        require(
            raw.checkpoint_path == "tokenizer.pt",
            "checkpoint path is metadata only and must be tokenizer.pt",
        )?;
        require(
            [raw.dropout, raw.pre_quant_dropout, q.decay]
                .into_iter()
                .all(|x| x.is_finite() && (0.0..1.0).contains(&x))
                && q.aux.fine_drop.prob.is_finite()
                && (0.0..=1.0).contains(&q.aux.fine_drop.prob)
                && (lm1b || q.aux.fine_drop.prob < 1.0),
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
                && q.learned_l1_cascade != lm1b
                && !q.ema_scale_by_world_size,
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
                && levels.first() == Some(&first_level)
                && raw.context_length.is_multiple_of(first_level)
                && levels.last() == Some(&raw.context_length)
                && levels.windows(2).all(|w| w[0] < w[1]),
            "levels must increase from the profile first level to context length",
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
        let mut remaining = raw.context_length / first_level;
        let mut strides = Vec::new();
        while remaining > 1 && !lm1b {
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
        if lm1b {
            strides.push(remaining);
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
        require(
            bytes <= max_weight_bytes,
            "codec weights exceed profile limit",
        )?;
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
        if r.enc_positional {
            put("encoder.wpe.weight".into(), vec![r.context_length, d]);
        }
        if r.dec_positional {
            put("decoder.wpe.weight".into(), vec![r.context_length, d]);
        }
        put("decoder.ln_f.weight".into(), vec![d]);
        put(
            "decoder.output_layer.weight".into(),
            vec![r.in_vocab_size, d],
        );
        for block in std::iter::once("decoder.middle_block".to_owned())
            .chain((0..r.dec_layers - 1).map(|i| format!("decoder.extra_blocks.{i}")))
        {
            for layer in ["ln_1", "ln_2"] {
                for suffix in ["weight", "bias"] {
                    put(format!("{block}.{layer}.{suffix}"), vec![d]);
                }
            }
            for layer in ["q_norm", "k_norm"] {
                put(format!("{block}.attn.{layer}.weight"), vec![d / r.n_layers]);
            }
            for (layer, output, input) in [
                ("attn.c_attn", 3 * d, d),
                ("attn.c_proj", d, d),
                ("mlp.c_fc", 4 * d, d),
                ("mlp.c_proj", d, 4 * d),
            ] {
                put(format!("{block}.{layer}.weight"), vec![output, input]);
                put(format!("{block}.{layer}.bias"), vec![output]);
            }
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
                if !r.vector_quantizer_config.learned_l1_cascade {
                    put(format!("vq.learned_{name}_l1.weight"), vec![d, d, stride]);
                    put(format!("vq.learned_{name}_l1.bias"), vec![d]);
                    continue;
                }
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

/// Output projection style for a `local-ncm-v1` generator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalHeadKind {
    /// Per-level `lm_heads.{i}.weight` and `lm_heads.{i}.bias` (Sudoku / XSum / WMT14 style).
    #[default]
    UntiedAffine,
    /// Per-level `lm_heads.{i}.weight` without bias (TinyStories style).
    UntiedLinear,
    /// Shared `lm_head.weight` plus per-level `head_biases.{i}` (LM1B style; requires uniform codebook sizes).
    SharedHeadBiases,
}

fn default_temperature() -> f64 {
    1.0
}

/// Generic generator configuration for `local-ncm-v1` checkpoints.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalGeneratorConfig {
    pub attn_pattern: String,
    pub bias: bool,
    pub context_length: usize,
    pub dropout: f64,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub prefix_len: usize,
    pub query_level0: bool,
    pub raw_token_prefix: bool,
    pub rope_base: usize,
    pub tie_heads_to_codebook: bool,
    pub use_level_encoding: bool,
    pub use_positional_encoding: bool,
    pub use_qk_norm: bool,
    pub use_rope: bool,
    pub vocab_size: Vec<usize>,
    #[serde(default)]
    pub head_kind: LocalHeadKind,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
}

impl LocalGeneratorConfig {
    pub fn validate(&self, codec: &CodecConfig) -> Result<()> {
        let width = self.n_embd;
        let require = |valid, message: &str| {
            if valid {
                Ok(())
            } else {
                Err(Error::Config(message.into()))
            }
        };
        require(
            self.attn_pattern == "block_diagonal",
            "local generator requires block_diagonal attention pattern",
        )?;
        require(
            self.bias
                && self.query_level0
                && self.raw_token_prefix
                && !self.tie_heads_to_codebook
                && self.use_level_encoding
                && self.use_positional_encoding
                && self.use_qk_norm
                && self.use_rope,
            "unsupported local generator architecture flags",
        )?;
        require(
            (8..=4096).contains(&width)
                && self.n_head > 0
                && width.is_multiple_of(self.n_head)
                && (width / self.n_head).is_multiple_of(2)
                && (1..=48).contains(&self.n_layer)
                && (1..=1024).contains(&self.prefix_len)
                && (1..=2048).contains(&self.context_length)
                && self.rope_base >= 2
                && self.dropout.is_finite()
                && (0.0..=1.0).contains(&self.dropout),
            "local generator dimensions or hyperparameters outside bounds",
        )?;
        require(
            self.vocab_size == codec.raw().vqvae_vocab_size,
            "local generator vocab_size must match codec vqvae_vocab_size",
        )?;
        if self.head_kind == LocalHeadKind::SharedHeadBiases {
            let first = self.vocab_size[0];
            require(
                self.vocab_size.iter().all(|&v| v == first),
                "shared_head_biases requires uniform codebook sizes across all levels",
            )?;
        }
        if let Some(pad) = self.pad_token_id {
            require(
                (pad as usize) < codec.raw().in_vocab_size,
                "pad_token_id must be within codec in_vocab_size",
            )?;
        }
        if let Some(eos) = self.eos_token_id {
            require(
                (eos as usize) < codec.raw().in_vocab_size,
                "eos_token_id must be within codec in_vocab_size",
            )?;
        }
        require(
            self.temperature.is_finite() && self.temperature > 0.0 && self.temperature <= 10.0,
            "temperature must be finite and in (0.0, 10.0]",
        )?;
        let bytes: usize = self
            .weight_shapes(codec)
            .values()
            .try_fold(0usize, |acc, s| {
                let elems = s.iter().try_fold(1usize, |p, &d| p.checked_mul(d))?;
                acc.checked_add(elems.checked_mul(4)?)
            })
            .ok_or_else(|| Error::Config("local generator weight size overflow".into()))?;
        require(
            bytes <= 4 * 1024 * 1024 * 1024,
            "local generator weights exceed 4 GiB limit",
        )?;
        Ok(())
    }

    pub fn weight_shapes(&self, codec: &CodecConfig) -> BTreeMap<String, Vec<usize>> {
        let mut shapes = BTreeMap::new();
        let d = self.n_embd;
        let insert_linear =
            |shapes: &mut BTreeMap<String, Vec<usize>>, name: &str, input, output| {
                shapes.insert(format!("{name}.weight"), vec![output, input]);
                shapes.insert(format!("{name}.bias"), vec![output]);
            };
        insert_linear(&mut shapes, "transformer.wte", codec.raw().embed_dim, d);
        match self.head_kind {
            LocalHeadKind::UntiedAffine => {
                for (i, &vocab) in self.vocab_size.iter().enumerate() {
                    insert_linear(&mut shapes, &format!("lm_heads.{i}"), d, vocab);
                }
            }
            LocalHeadKind::UntiedLinear => {
                for (i, &vocab) in self.vocab_size.iter().enumerate() {
                    shapes.insert(format!("lm_heads.{i}.weight"), vec![vocab, d]);
                }
            }
            LocalHeadKind::SharedHeadBiases => {
                shapes.insert("lm_head.weight".into(), vec![self.vocab_size[0], d]);
                for (i, &vocab) in self.vocab_size.iter().enumerate() {
                    shapes.insert(format!("head_biases.{i}"), vec![vocab]);
                }
            }
        }
        for i in 0..self.n_layer {
            let p = format!("transformer.blocks.{i}");
            insert_linear(&mut shapes, &format!("{p}.attn.c_attn"), d, 3 * d);
            insert_linear(&mut shapes, &format!("{p}.attn.c_proj"), d, d);
            insert_linear(&mut shapes, &format!("{p}.mlp.c_fc"), d, 4 * d);
            insert_linear(&mut shapes, &format!("{p}.mlp.c_proj"), 4 * d, d);
        }
        shapes.insert(
            "l0_queries".into(),
            vec![1, codec.levels()[0], codec.raw().embed_dim],
        );
        shapes.insert(
            "transformer.wle.weight".into(),
            vec![codec.levels().len(), d],
        );
        for suffix in ["weight", "bias"] {
            shapes.insert(format!("transformer.ln_f.{suffix}"), vec![d]);
            for i in 0..self.n_layer {
                for layer in ["ln_1", "ln_2"] {
                    shapes.insert(format!("transformer.blocks.{i}.{layer}.{suffix}"), vec![d]);
                }
            }
        }
        for i in 0..self.n_layer {
            for name in ["q_norm", "k_norm"] {
                shapes.insert(
                    format!("transformer.blocks.{i}.attn.{name}.weight"),
                    vec![d / self.n_head],
                );
            }
        }
        shapes
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLocalNcmConfig {
    tokenizer: RawCodecConfig,
    model: LocalGeneratorConfig,
}

/// Validated configuration for a `local-ncm-v1` artifact.
#[derive(Clone, Debug)]
pub struct LocalNcmConfig {
    codec: CodecConfig,
    model: LocalGeneratorConfig,
}

impl LocalNcmConfig {
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > 1024 * 1024 {
            return Err(Error::Config("local config exceeds 1 MiB".into()));
        }
        let raw: RawLocalNcmConfig = serde_json::from_slice(bytes)?;
        let codec = CodecConfig::validate_local(raw.tokenizer)?;
        raw.model.validate(&codec)?;
        Ok(Self {
            codec,
            model: raw.model,
        })
    }
    pub fn codec(&self) -> &CodecConfig {
        &self.codec
    }
    pub fn model(&self) -> &LocalGeneratorConfig {
        &self.model
    }
}

#[cfg(test)]
mod local_config_tests {
    use super::{LocalHeadKind, LocalNcmConfig};

    fn sample_json(head_kind: &str, pad: Option<u32>, vocab_sizes: &str) -> String {
        let pad_line = match pad {
            Some(id) => format!(r#""pad_token_id": {id},"#),
            None => String::new(),
        };
        format!(
            r#"{{
              "tokenizer": {{
                "checkpoint_path": "local-codec.pt",
                "in_vocab_size": 64,
                "context_length": 8,
                "embed_dim": 16,
                "vqvae_vocab_size": {vocab_sizes},
                "n_layers": 2,
                "compression_factor": 1,
                "dropout": 0.0,
                "pre_quant_dropout": 0.0,
                "pre_quant_groupnorm": 4,
                "pre_quant_norm_type": "group",
                "use_qk_norm": true,
                "dec_layers": 2,
                "enc_positional": true,
                "dec_positional": false,
                "vector_quantizer_config": {{
                  "clss": "multiscale_residual_vector_quantizer",
                  "commitment_cost": 0.25,
                  "decay": 0.99,
                  "epsilon": 1e-5,
                  "learned_all_sampling": false,
                  "learned_l1_sampling": true,
                  "learned_l1_cascade": true,
                  "learned_l1_cascade_base": 2,
                  "learned_l1_cascade_norm": true,
                  "learned_l1_cascade_norm_final": false,
                  "levels": {{
                    "use_manual_levels": true,
                    "manual_levels": [2, 4, 8]
                  }},
                  "quant_resi": {{
                    "enabled": true,
                    "learnable_ratio": false,
                    "ratio": 0.5,
                    "share_mode": 0
                  }},
                  "aux": {{
                    "fine_drop": {{
                      "min_keep": 1,
                      "prob": 0.0
                    }}
                  }}
                }}
              }},
              "model": {{
                "attn_pattern": "block_diagonal",
                "bias": true,
                "context_length": 24,
                "dropout": 0.0,
                "n_embd": 32,
                "n_head": 4,
                "n_layer": 2,
                "prefix_len": 6,
                "query_level0": true,
                "raw_token_prefix": true,
                "rope_base": 10000,
                "tie_heads_to_codebook": false,
                "use_level_encoding": true,
                "use_positional_encoding": true,
                "use_qk_norm": true,
                "use_rope": true,
                "vocab_size": {vocab_sizes},
                "head_kind": "{head_kind}",
                {pad_line}
                "eos_token_id": 0
              }}
            }}"#
        )
    }

    #[test]
    fn local_config_validates_all_head_kinds_and_rejects_mismatches() {
        let cfg = LocalNcmConfig::from_json(
            sample_json("untied_affine", Some(1), "[16, 24, 32]").as_bytes(),
        )
        .unwrap();
        assert_eq!(cfg.model().head_kind, LocalHeadKind::UntiedAffine);
        assert_eq!(cfg.codec().strides(), &[2, 2]);
        assert!(
            cfg.model()
                .weight_shapes(cfg.codec())
                .contains_key("lm_heads.2.bias")
        );

        let cfg_lin = LocalNcmConfig::from_json(
            sample_json("untied_linear", None, "[16, 24, 32]").as_bytes(),
        )
        .unwrap();
        assert_eq!(cfg_lin.model().head_kind, LocalHeadKind::UntiedLinear);
        assert!(
            !cfg_lin
                .model()
                .weight_shapes(cfg_lin.codec())
                .contains_key("lm_heads.0.bias")
        );

        let cfg_shared = LocalNcmConfig::from_json(
            sample_json("shared_head_biases", Some(1), "[32, 32, 32]").as_bytes(),
        )
        .unwrap();
        assert_eq!(
            cfg_shared.model().head_kind,
            LocalHeadKind::SharedHeadBiases
        );
        assert!(
            cfg_shared
                .model()
                .weight_shapes(cfg_shared.codec())
                .contains_key("head_biases.2")
        );

        // Non-uniform codebook sizes must be rejected for shared_head_biases.
        assert!(
            LocalNcmConfig::from_json(
                sample_json("shared_head_biases", Some(1), "[16, 24, 32]").as_bytes()
            )
            .is_err()
        );
        // Out-of-range pad_token_id must be rejected.
        assert!(
            LocalNcmConfig::from_json(
                sample_json("untied_affine", Some(64), "[16, 24, 32]").as_bytes()
            )
            .is_err()
        );
    }
}
