//! Checked weights and eager greedy generation for the Sudoku profile.
use crate::{artifact_config, has_nonfinite_f32_le, upload_values};
use ncm_core::{CodecConfig, Error, Result, WeightBackend};
use ncm_io::Artifact;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratorConfig {
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
}

impl GeneratorConfig {
    fn validate(&self, codec: &CodecConfig) -> Result<()> {
        let width = self.n_embd;
        if self.attn_pattern != "block_diagonal"
            || !self.bias
            || !self.query_level0
            || !self.raw_token_prefix
            || self.tie_heads_to_codebook
            || !self.use_level_encoding
            || !self.use_positional_encoding
            || !self.use_qk_norm
            || !self.use_rope
            || !(8..=2048).contains(&width)
            || self.n_head == 0
            || !width.is_multiple_of(self.n_head)
            || !(width / self.n_head).is_multiple_of(2)
            || !(1..=32).contains(&self.n_layer)
            || !(1..=512).contains(&self.prefix_len)
            || !(1..=1024).contains(&self.context_length)
            || self.rope_base < 2
            || !(0.0..=1.0).contains(&self.dropout)
            || self.vocab_size != codec.raw().vqvae_vocab_size
        {
            return Err(Error::Config(
                "unsupported Sudoku generator configuration".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn shapes(&self, codec: &CodecConfig) -> BTreeMap<String, Vec<usize>> {
        let mut shapes = BTreeMap::new();
        let d = self.n_embd;
        let mut linear = |name: &str, input, output| {
            shapes.insert(format!("{name}.weight"), vec![output, input]);
            shapes.insert(format!("{name}.bias"), vec![output]);
        };
        linear("transformer.wte", codec.raw().embed_dim, d);
        for (i, &vocab) in self.vocab_size.iter().enumerate() {
            linear(&format!("lm_heads.{i}"), d, vocab);
        }
        for i in 0..self.n_layer {
            let p = format!("transformer.blocks.{i}");
            linear(&format!("{p}.attn.c_attn"), d, 3 * d);
            linear(&format!("{p}.attn.c_proj"), d, d);
            linear(&format!("{p}.mlp.c_fc"), d, 4 * d);
            linear(&format!("{p}.mlp.c_proj"), 4 * d, d);
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
struct Envelope {
    model: GeneratorConfig,
}

/// Immutable, fully preflight-checked generator parameters. Use `Generator`
/// to run inference with these parameters and the matching codec.
pub struct GeneratorWeights<B: WeightBackend> {
    backend: B,
    config: GeneratorConfig,
    parameters: GeneratorParameters<B::Tensor>,
    tensor_count: usize,
}
struct Affine<T> {
    weight: T,
    bias: T,
}
struct GeneratorBlock<T> {
    ln_1: Affine<T>,
    qkv: Affine<T>,
    q_norm: T,
    k_norm: T,
    attention_projection: Affine<T>,
    ln_2: Affine<T>,
    mlp_in: Affine<T>,
    mlp_out: Affine<T>,
}
struct GeneratorParameters<T> {
    input: Affine<T>,
    query: T,
    level_embedding: T,
    final_norm: Affine<T>,
    blocks: Box<[GeneratorBlock<T>]>,
    heads: Box<[Affine<T>]>,
}
fn take<T>(weights: &mut BTreeMap<String, T>, name: &str) -> Result<T> {
    weights
        .remove(name)
        .ok_or_else(|| Error::Weights(format!("missing generator tensor {name}")))
}
fn take_affine<T>(weights: &mut BTreeMap<String, T>, name: &str) -> Result<Affine<T>> {
    Ok(Affine {
        weight: take(weights, &format!("{name}.weight"))?,
        bias: take(weights, &format!("{name}.bias"))?,
    })
}
impl<B: WeightBackend> GeneratorWeights<B> {
    pub fn from_artifact(backend: B, artifact: &Artifact) -> Result<Self> {
        let config_bytes = artifact
            .file("config.json")
            .map_err(|e| Error::Weights(e.to_string()))?;
        crate::validate_artifact_profile(artifact, config_bytes)?;
        let codec = artifact_config(config_bytes)?;
        let envelope: Envelope = serde_json::from_slice(config_bytes)?;
        let bytes = artifact
            .file("generator.safetensors")
            .map_err(|e| Error::Weights(e.to_string()))?;
        Self::from_safetensors(backend, envelope.model, &codec, bytes)
    }
    pub fn from_safetensors(
        backend: B,
        config: GeneratorConfig,
        codec: &CodecConfig,
        bytes: &[u8],
    ) -> Result<Self> {
        config.validate(codec)?;
        if bytes.len() > 256 * 1024 * 1024 + 1024 * 1024 {
            return Err(Error::Weights("generator exceeds size limit".into()));
        }
        let tensors = SafeTensors::deserialize(bytes).map_err(|e| Error::Weights(e.to_string()))?;
        let shapes = config.shapes(codec);
        if tensors.len() != shapes.len() + 2 {
            return Err(Error::Weights("generator tensor inventory differs".into()));
        }
        for (name, shape) in &shapes {
            let t = tensors
                .tensor(name)
                .map_err(|e| Error::Weights(e.to_string()))?;
            if t.dtype() != Dtype::F32 || t.shape() != shape || has_nonfinite_f32_le(t.data()) {
                return Err(Error::Weights(format!("invalid generator tensor {name}")));
            }
        }
        // These upstream persistent buffers are validated, not uploaded: eager
        // inference constructs per-level masks and positions independently.
        let total: usize = codec.levels().iter().sum();
        let level = tensors
            .tensor("lvl_1L")
            .map_err(|e| Error::Weights(e.to_string()))?;
        if level.dtype() != Dtype::I64 || level.shape() != [1, total] {
            return Err(Error::Weights("invalid level buffer".into()));
        }
        let expected = codec
            .levels()
            .iter()
            .enumerate()
            .flat_map(|(i, &len)| std::iter::repeat_n(i as i64, len));
        if !level
            .data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|x| i64::from_le_bytes(*x))
            .eq(expected)
        {
            return Err(Error::Weights("incorrect level buffer".into()));
        }
        let seq = config.prefix_len + total;
        let mask = tensors
            .tensor("attn_mask_for_masking")
            .map_err(|e| Error::Weights(e.to_string()))?;
        if mask.dtype() != Dtype::F32 || mask.shape() != [1, 1, seq, seq] {
            return Err(Error::Weights("invalid attention mask".into()));
        }
        let mut levels = vec![usize::MAX; config.prefix_len];
        for (i, &len) in codec.levels().iter().enumerate() {
            levels.extend(std::iter::repeat_n(i, len));
        }
        for (i, raw) in mask.data().as_chunks::<4>().0.iter().enumerate() {
            let (row, col) = (i / seq, i % seq);
            let allowed =
                col < config.prefix_len || (row >= config.prefix_len && levels[row] == levels[col]);
            let expected = if allowed {
                0f32.to_bits()
            } else {
                f32::NEG_INFINITY.to_bits()
            };
            if u32::from_le_bytes(*raw) != expected {
                return Err(Error::Weights("incorrect attention mask".into()));
            }
        }
        let tensor_count = shapes.len();
        let mut weights = BTreeMap::new();
        for (name, shape) in shapes {
            let tensor = tensors
                .tensor(&name)
                .map_err(|e| Error::Weights(e.to_string()))?;
            let values: Vec<f32> = tensor
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|x| f32::from_le_bytes(*x))
                .collect();
            weights.insert(name, backend.upload_f32(&values, &shape)?);
        }
        let input = take_affine(&mut weights, "transformer.wte")?;
        let query = take(&mut weights, "l0_queries")?;
        let level_embedding = take(&mut weights, "transformer.wle.weight")?;
        let final_norm = take_affine(&mut weights, "transformer.ln_f")?;
        let mut blocks = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            let p = format!("transformer.blocks.{i}");
            blocks.push(GeneratorBlock {
                ln_1: take_affine(&mut weights, &format!("{p}.ln_1"))?,
                qkv: take_affine(&mut weights, &format!("{p}.attn.c_attn"))?,
                q_norm: take(&mut weights, &format!("{p}.attn.q_norm.weight"))?,
                k_norm: take(&mut weights, &format!("{p}.attn.k_norm.weight"))?,
                attention_projection: take_affine(&mut weights, &format!("{p}.attn.c_proj"))?,
                ln_2: take_affine(&mut weights, &format!("{p}.ln_2"))?,
                mlp_in: take_affine(&mut weights, &format!("{p}.mlp.c_fc"))?,
                mlp_out: take_affine(&mut weights, &format!("{p}.mlp.c_proj"))?,
            });
        }
        let mut heads = Vec::with_capacity(config.vocab_size.len());
        for i in 0..config.vocab_size.len() {
            heads.push(take_affine(&mut weights, &format!("lm_heads.{i}"))?);
        }
        if !weights.is_empty() {
            return Err(Error::Weights("unbound generator tensor".into()));
        }
        let parameters = GeneratorParameters {
            input,
            query,
            level_embedding,
            final_norm,
            blocks: blocks.into_boxed_slice(),
            heads: heads.into_boxed_slice(),
        };
        Ok(Self {
            backend,
            config,
            parameters,
            tensor_count,
        })
    }
    pub fn config(&self) -> &GeneratorConfig {
        &self.config
    }
    pub fn backend(&self) -> &B {
        &self.backend
    }
    pub fn tensor_count(&self) -> usize {
        self.tensor_count
    }
}

/// Eager, greedy Sudoku generator. Weights and codec share an explicitly
/// selected backend; immutable geometry is model-owned, and activations are request-local.
pub struct Generator<B: ncm_core::GeneratorBackend> {
    codec: crate::Codec<B>,
    model: GeneratorWeights<B>,
    geometry: Vec<LevelGeometry<B::Tensor>>,
    request_lock: std::sync::Mutex<()>,
}
struct LevelGeometry<T> {
    mask: T,
    cosine: T,
    sine: T,
}
/// Device-resident generated codes and final codec output.
pub struct Generated<B: ncm_core::GeneratorBackend> {
    pub codes: B::Indices,
    pub logits: B::Tensor,
    pub ids: B::Indices,
}
impl<B: ncm_core::GeneratorBackend + Clone> Generator<B> {
    pub fn from_artifact(backend: B, artifact: &Artifact) -> Result<Self> {
        let codec = crate::Codec::from_artifact(backend.clone(), artifact)?;
        let model = GeneratorWeights::from_artifact(backend, artifact)?;
        let geometry = Self::prepare_geometry(&model, codec.config().levels())?;
        Ok(Self {
            codec,
            model,
            geometry,
            request_lock: std::sync::Mutex::new(()),
        })
    }
}
impl<B: ncm_core::GeneratorBackend> Generator<B> {
    fn prepare_geometry(
        model: &GeneratorWeights<B>,
        levels: &[usize],
    ) -> Result<Vec<LevelGeometry<B::Tensor>>> {
        prepare_geometry(
            &model.backend,
            levels,
            model.config.prefix_len,
            model.config.n_embd,
            model.config.n_head,
            model.config.rope_base,
        )
    }

    fn linear(&self, affine: &Affine<B::Tensor>, x: &B::Tensor) -> Result<B::Tensor> {
        self.model
            .backend
            .project_bias(x, &affine.weight, &affine.bias)
    }
    fn decode_level(
        &self,
        prefix: &B::Tensor,
        target: &B::Tensor,
        length: usize,
        level: usize,
    ) -> Result<B::Tensor> {
        let geometry = self
            .geometry
            .get(level)
            .ok_or_else(|| Error::Input("invalid generator level".into()))?;
        decode_transformer(
            &self.model.backend,
            &self.model.parameters.blocks,
            &self.model.parameters.final_norm,
            geometry,
            prefix,
            target,
            self.model.config.prefix_len,
            self.model.config.n_head,
            self.model.config.n_embd,
            length,
        )
    }

    /// Greedy generation (temperature zero), without caching or CPU fallback.
    pub fn generate(&self, prefix_tokens: &[u32], batch: usize) -> Result<Generated<B>> {
        self.generate_traced(prefix_tokens, batch, &mut crate::NoTrace)
    }
    /// Trace callbacks must not recursively call this generator: requests on a
    /// shared generator are serialized, including the callback lifetime.
    pub fn generate_traced(
        &self,
        prefix_tokens: &[u32],
        batch: usize,
        trace: &mut impl crate::TraceSink<B>,
    ) -> Result<Generated<B>> {
        // Candle's shared CUDA stream may enter a driver deadlock if requests
        // concurrently look up kernels and launch work. Serialize requests on
        // this model; their activations and outputs remain request-local.
        let _request = self
            .request_lock
            .lock()
            .map_err(|_| Error::Backend("generator request lock poisoned".into()))?;
        let cfg = &self.model.config;
        let codec_cfg = self.codec.config();
        if !(1..=128).contains(&batch)
            || batch.checked_mul(cfg.prefix_len) != Some(prefix_tokens.len())
            || prefix_tokens
                .iter()
                .any(|&t| t as usize >= codec_cfg.raw().in_vocab_size)
        {
            return Err(Error::Input("invalid generator prefix or batch".into()));
        }
        let b = &self.model.backend;
        let ids = b.upload_indices(prefix_tokens, &[batch, cfg.prefix_len])?;
        let raw = b.embedding(self.codec.weight("encoder.wte.weight")?, &ids)?;
        let prefix = self.linear(&self.model.parameters.input, &raw)?;
        let mut cumulative: Option<B::Tensor> = None;
        let mut codes = Vec::with_capacity(codec_cfg.levels().len());
        let mut previous: Option<B::Indices> = None;
        for (level, &length) in codec_cfg.levels().iter().enumerate() {
            let latent = if level == 0 {
                let zeros = b.upload_f32(
                    &vec![0.0; batch * length * codec_cfg.raw().embed_dim],
                    &[batch, length, codec_cfg.raw().embed_dim],
                )?;
                b.add(&self.model.parameters.query, &zeros)?
            } else {
                let book = self
                    .codec
                    .weight(&format!("vq.embeddings.{}.weight", level - 1))?;
                let embedding =
                    b.embedding(book, previous.as_ref().expect("preceding generated level"))?;
                let up = self.codec.upsample(&embedding, level - 1)?;
                let contribution = self.codec.refine(&up, level - 1)?;
                let sum = match cumulative {
                    Some(ref old) => b.add(old, &contribution)?,
                    None => contribution,
                };
                let down = b.area1d(&sum, length)?;
                cumulative = Some(sum);
                down
            };
            let target = self.linear(&self.model.parameters.input, &latent)?;
            let embedding = b.narrow(&self.model.parameters.level_embedding, 0, level, 1)?;
            let target = b.add(&target, &embedding)?;
            let hidden = self.decode_level(&prefix, &target, length, level)?;
            let logits = self.linear(&self.model.parameters.heads[level], &hidden)?;
            trace.tensor(&format!("generator.level_{level:02}.logits"), &logits)?;
            let next = b.select_last(&logits, true)?;
            trace.indices(&format!("generator.level_{level:02}.codes"), &next)?;
            codes.push(next.clone());
            previous = Some(next);
        }
        // The preceding levels were already accumulated to predict the next
        // level. Add only the final generated codebook contribution rather than
        // reconstructing every level and retaining a full snapshot per level.
        let last = codec_cfg.levels().len() - 1;
        let book = self.codec.weight(&format!("vq.embeddings.{last}.weight"))?;
        let embedding = b.embedding(book, previous.as_ref().expect("final generated level"))?;
        let up = self.codec.upsample(&embedding, last)?;
        let contribution = self.codec.refine(&up, last)?;
        let latent = b.add(
            cumulative.as_ref().expect("preceding generated levels"),
            &contribution,
        )?;
        let codes = b.concat_indices(&codes, 1)?;
        let (logits, ids) = self
            .codec
            .decode_latent_outputs(&latent, &mut crate::NoTrace)?;
        Ok(Generated { codes, logits, ids })
    }
}

fn prepare_geometry<B: ncm_core::GeneratorBackend>(
    b: &B,
    levels: &[usize],
    prefix_len: usize,
    width: usize,
    heads: usize,
    rope_base: usize,
) -> Result<Vec<LevelGeometry<B::Tensor>>> {
    prepare_geometry_with_rope(b, levels, prefix_len, width, heads, rope_base, None)
}

fn validate_reference_rope(
    levels: &[usize],
    prefix_len: usize,
    head: usize,
    cosine: &[f32],
    sine: &[f32],
) -> Result<()> {
    let max_position = levels.iter().copied().max().unwrap_or(0).max(prefix_len);
    let required = max_position
        .checked_mul(head / 2)
        .ok_or_else(|| Error::Input("rotary reference size overflow".into()))?;
    if cosine.len() != required
        || sine.len() != required
        || cosine.iter().chain(sine).any(|x| !x.is_finite())
    {
        return Err(Error::Weights("invalid rotary reference tables".into()));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn prepare_geometry_with_rope<B: ncm_core::GeneratorBackend>(
    b: &B,
    levels: &[usize],
    prefix_len: usize,
    width: usize,
    heads: usize,
    rope_base: usize,
    reference_rope: Option<(&[f32], &[f32])>,
) -> Result<Vec<LevelGeometry<B::Tensor>>> {
    let head = width / heads;
    if let Some((cosine, sine)) = reference_rope {
        validate_reference_rope(levels, prefix_len, head, cosine, sine)?;
    }
    let mut result = Vec::with_capacity(levels.len());
    for &length in levels {
        let seq = prefix_len + length;
        let mut mask = vec![0.0f32; seq * seq];
        for row in 0..prefix_len {
            for col in prefix_len..seq {
                mask[row * seq + col] = f32::NEG_INFINITY;
            }
        }
        let mask = b.upload_f32(&mask, &[1, 1, seq, seq])?;
        let mut cosine = Vec::with_capacity(seq * head / 2);
        let mut sine = Vec::with_capacity(seq * head / 2);
        if let Some((reference_cosine, reference_sine)) = reference_rope {
            for pos in (0..prefix_len).chain(0..length) {
                let start = pos * (head / 2);
                cosine.extend_from_slice(&reference_cosine[start..start + head / 2]);
                sine.extend_from_slice(&reference_sine[start..start + head / 2]);
            }
        } else {
            for pos in (0..prefix_len).chain(0..length) {
                for i in 0..head / 2 {
                    let freq = (pos as f32) / (rope_base as f32).powf(i as f32 / (head / 2) as f32);
                    cosine.push(freq.cos());
                    sine.push(freq.sin());
                }
            }
        }
        result.push(LevelGeometry {
            mask,
            cosine: b.upload_f32(&cosine, &[seq, head / 2])?,
            sine: b.upload_f32(&sine, &[seq, head / 2])?,
        });
    }
    Ok(result)
}

// Shared block execution for profiles with the same upstream transformer.
// Profile-specific embeddings, attention masks and output heads stay outside.
#[allow(clippy::too_many_arguments)]
fn decode_transformer<B: ncm_core::GeneratorBackend>(
    b: &B,
    blocks: &[GeneratorBlock<B::Tensor>],
    final_norm: &Affine<B::Tensor>,
    geometry: &LevelGeometry<B::Tensor>,
    prefix: &B::Tensor,
    target: &B::Tensor,
    prefix_len: usize,
    heads: usize,
    width: usize,
    length: usize,
) -> Result<B::Tensor> {
    decode_transformer_observed(
        b, blocks, final_norm, geometry, prefix, target, prefix_len, heads, width, length, false,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn decode_transformer_observed<B: ncm_core::GeneratorBackend>(
    b: &B,
    blocks: &[GeneratorBlock<B::Tensor>],
    final_norm: &Affine<B::Tensor>,
    geometry: &LevelGeometry<B::Tensor>,
    prefix: &B::Tensor,
    target: &B::Tensor,
    prefix_len: usize,
    heads: usize,
    width: usize,
    length: usize,
    capture_qkv: bool,
    mut trace: Option<&mut dyn crate::TraceSink<B>>,
) -> Result<B::Tensor> {
    let batch = b.shape(prefix)[0];
    let seq = prefix_len + length;
    let head = width / heads;
    let linear =
        |affine: &Affine<B::Tensor>, x: &B::Tensor| b.project_bias(x, &affine.weight, &affine.bias);
    let norm = |affine: &Affine<B::Tensor>, x: &B::Tensor| {
        if batch < 16 {
            b.layer_norm(x, &affine.weight, Some(&affine.bias), 1e-6)
        } else {
            b.layer_norm_accelerated(x, &affine.weight, Some(&affine.bias), 1e-6)
        }
    };
    let mut x = b.concat_f32(&[prefix.clone(), target.clone()], 1)?;
    if let Some(trace) = trace.as_deref_mut() {
        trace.tensor("diagnostic.level_01.input", &x)?;
    }
    for (index, block) in blocks.iter().enumerate() {
        let normalized = norm(&block.ln_1, &x)?;
        if index == 0
            && let Some(trace) = trace.as_deref_mut()
        {
            trace.tensor("diagnostic.level_01.first_norm", &normalized)?;
        }
        let qkv = linear(&block.qkv, &normalized)?;
        if index == 0
            && let Some(trace) = trace.as_deref_mut()
        {
            trace.tensor("diagnostic.level_01.first_qkv", &qkv)?;
        }
        let [q, k, v] = b.split_qkv(&qkv, batch, seq, heads, head)?;
        if index == 0
            && capture_qkv
            && let Some(trace) = trace.as_deref_mut()
        {
            trace.tensor(
                "diagnostic.level_01.normalized_q",
                &b.rms_norm(&q, &block.q_norm, 1e-6)?,
            )?;
            trace.tensor(
                "diagnostic.level_01.normalized_k",
                &b.rms_norm(&k, &block.k_norm, 1e-6)?,
            )?;
            trace.tensor("diagnostic.level_01.cosine", &geometry.cosine)?;
            trace.tensor("diagnostic.level_01.sine", &geometry.sine)?;
        }
        let q = b.rms_rope(&q, &block.q_norm, &geometry.cosine, &geometry.sine, 1e-6)?;
        let k = b.rms_rope(&k, &block.k_norm, &geometry.cosine, &geometry.sine, 1e-6)?;
        if index == 0
            && capture_qkv
            && let Some(trace) = trace.as_deref_mut()
        {
            trace.tensor("diagnostic.level_01.rotated_q", &q)?;
            trace.tensor("diagnostic.level_01.rotated_k", &k)?;
            trace.tensor("diagnostic.level_01.value", &v)?;
        }
        let attn = b.attention(&q, &k, &v, Some(&geometry.mask))?;
        let attn = b.merge_heads(&attn, batch, seq, heads, head)?;
        if index == 0
            && let Some(trace) = trace.as_deref_mut()
        {
            trace.tensor("diagnostic.level_01.first_attention", &attn)?;
        }
        x = b.add(&x, &linear(&block.attention_projection, &attn)?)?;
        if index == 0
            && let Some(trace) = trace.as_deref_mut()
        {
            trace.tensor("diagnostic.level_01.first_residual", &x)?;
        }
        let normalized = norm(&block.ln_2, &x)?;
        let projected = linear(&block.mlp_in, &normalized)?;
        let hidden = b.gelu(&projected)?;
        if index == 0
            && let Some(trace) = trace.as_deref_mut()
        {
            trace.tensor("diagnostic.level_01.second_norm", &normalized)?;
            trace.tensor("diagnostic.level_01.first_mlp_in", &projected)?;
            trace.tensor("diagnostic.level_01.first_mlp_gelu", &hidden)?;
        }
        x = b.add(&x, &linear(&block.mlp_out, &hidden)?)?;
        if let Some(trace) = trace.as_deref_mut() {
            trace.tensor(&format!("diagnostic.level_01.block_{index:02}"), &x)?;
        }
    }
    let x = norm(final_norm, &x)?;
    b.narrow(&x, 1, prefix_len, length)
}

#[cfg(feature = "lm1b-experimental")]
struct Lm1bLevelSevenTrace<'a, T>(&'a mut T);
#[cfg(feature = "lm1b-experimental")]
impl<B: ncm_core::GeneratorBackend, T: crate::TraceSink<B>> crate::TraceSink<B>
    for Lm1bLevelSevenTrace<'_, T>
{
    fn tensor(&mut self, name: &str, value: &B::Tensor) -> Result<()> {
        let suffix = name
            .strip_prefix("diagnostic.level_01.")
            .ok_or_else(|| Error::Backend("unexpected LM1B diagnostic tensor name".into()))?;
        self.0
            .tensor(&format!("diagnostic.level_07.{suffix}"), value)
    }
    fn indices(&mut self, name: &str, value: &B::Indices) -> Result<()> {
        self.0.indices(name, value)
    }
}

/// Fixed-history development path for the pinned WMT14 configuration.
/// It does not implement sampling or qualified translation inference.
#[cfg(feature = "wmt14-experimental")]
pub struct Wmt14Teacher<B: ncm_core::GeneratorBackend> {
    codec: crate::Codec<B>,
    backend: B,
    input: Affine<B::Tensor>,
    level_embedding: B::Tensor,
    final_norm: Affine<B::Tensor>,
    blocks: Box<[GeneratorBlock<B::Tensor>]>,
    heads: Box<[B::Tensor]>,
    geometry: Vec<LevelGeometry<B::Tensor>>,
    cached_prefix_experimental: bool,
    cached_cfg_diagnostic: bool,
    cfg_internals_diagnostic: bool,
    cfg_shape_matched_diagnostic: bool,
    cfg_recompute_margin_diagnostic: Option<f32>,
    deferred_head_check_experimental: bool,
}

/// Host-resident output from explicitly microbatched experimental WMT14 generation.
#[cfg(feature = "wmt14-experimental")]
pub struct Wmt14HostOutput {
    pub codes: Vec<u32>,
    pub ids: Vec<u32>,
}

#[cfg(feature = "wmt14-experimental")]
impl<B: ncm_core::GeneratorBackend + Clone> Wmt14Teacher<B> {
    pub fn from_artifact(backend: B, artifact: &Artifact) -> Result<Self> {
        crate::Wmt14GeneratorInspection::from_artifact(artifact)?;
        let codec = crate::Codec::from_wmt14_artifact_experimental(backend.clone(), artifact)?;
        Self::from_checked_parts(backend, artifact, codec)
    }

    /// Release the validated codec payload before uploading the generator.
    pub fn from_owned_artifact(backend: B, mut artifact: Artifact) -> Result<Self> {
        crate::Wmt14GeneratorInspection::from_artifact(&artifact)?;
        let codec = crate::Codec::from_wmt14_artifact_experimental(backend.clone(), &artifact)?;
        artifact
            .release_file("codec.safetensors")
            .map_err(|e| Error::Weights(e.to_string()))?;
        Self::from_checked_parts(backend, &artifact, codec)
    }

    /// Build a WMT14 model using one reusable, revalidated host weight buffer.
    pub fn from_reusable_artifact(backend: B, artifact: Artifact) -> Result<Self> {
        crate::Wmt14GeneratorInspection::from_artifact(&artifact)?;
        let artifact = artifact
            .with_wmt14_bundle("codec.safetensors")
            .map_err(|e| Error::Weights(e.to_string()))?;
        let codec = crate::Codec::from_wmt14_artifact_experimental(backend.clone(), &artifact)?;
        let artifact = artifact
            .with_wmt14_bundle("generator.safetensors")
            .map_err(|e| Error::Weights(e.to_string()))?;
        Self::from_checked_parts(backend, &artifact, codec)
    }

    fn from_checked_parts(backend: B, artifact: &Artifact, codec: crate::Codec<B>) -> Result<Self> {
        let bytes = artifact
            .file("generator.safetensors")
            .map_err(|e| Error::Weights(e.to_string()))?;
        let tensors = SafeTensors::deserialize(bytes).map_err(|e| Error::Weights(e.to_string()))?;
        let mut weights = BTreeMap::new();
        for name in tensors.names() {
            if name == "lvl_1L" || name == "attn_mask_for_masking" {
                continue;
            }
            let tensor = tensors
                .tensor(name)
                .map_err(|e| Error::Weights(e.to_string()))?;
            weights.insert(
                name.to_owned(),
                backend.upload_f32(&upload_values(tensor.data())?, tensor.shape())?,
            );
        }
        let input = take_affine(&mut weights, "transformer.wte")?;
        let level_embedding = take(&mut weights, "transformer.wle.weight")?;
        let final_norm = take_affine(&mut weights, "transformer.ln_f")?;
        let mut blocks = Vec::with_capacity(12);
        for i in 0..12 {
            let p = format!("transformer.blocks.{i}");
            blocks.push(GeneratorBlock {
                ln_1: take_affine(&mut weights, &format!("{p}.ln_1"))?,
                qkv: take_affine(&mut weights, &format!("{p}.attn.c_attn"))?,
                q_norm: take(&mut weights, &format!("{p}.attn.q_norm.weight"))?,
                k_norm: take(&mut weights, &format!("{p}.attn.k_norm.weight"))?,
                attention_projection: take_affine(&mut weights, &format!("{p}.attn.c_proj"))?,
                ln_2: take_affine(&mut weights, &format!("{p}.ln_2"))?,
                mlp_in: take_affine(&mut weights, &format!("{p}.mlp.c_fc"))?,
                mlp_out: take_affine(&mut weights, &format!("{p}.mlp.c_proj"))?,
            });
        }
        let mut heads = vec![take(&mut weights, "lm_head_l0.weight")?];
        for i in 0..10 {
            heads.push(take(&mut weights, &format!("lm_heads.{i}.weight"))?);
        }
        if !weights.is_empty() {
            return Err(Error::Weights("unbound WMT14 generator tensor".into()));
        }
        let mut lengths = vec![0];
        lengths.extend_from_slice(&codec.config().levels()[1..]);
        let geometry = prepare_geometry(&backend, &lengths, 96, 768, 12, 100000)?;
        Ok(Self {
            codec,
            backend,
            input,
            level_embedding,
            final_norm,
            blocks: blocks.into_boxed_slice(),
            heads: heads.into_boxed_slice(),
            geometry,
            cached_prefix_experimental: false,
            cached_cfg_diagnostic: false,
            cfg_internals_diagnostic: false,
            cfg_shape_matched_diagnostic: false,
            cfg_recompute_margin_diagnostic: None,
            deferred_head_check_experimental: false,
        })
    }

    /// Explicit unqualified cache candidate; the standard constructor stays no-cache.
    pub fn from_artifact_cached_experimental(backend: B, artifact: &Artifact) -> Result<Self> {
        let mut model = Self::from_artifact(backend, artifact)?;
        model.cached_prefix_experimental = true;
        Ok(model)
    }

    /// Capture level-one internals for offline cached/uncached comparison only.
    pub fn from_artifact_cfg_internals_diagnostic(
        backend: B,
        artifact: &Artifact,
        cached: bool,
    ) -> Result<Self> {
        let mut model = Self::from_artifact(backend, artifact)?;
        model.cached_prefix_experimental = cached;
        model.cached_cfg_diagnostic = cached;
        model.cfg_internals_diagnostic = true;
        Ok(model)
    }

    /// Test shape-matched operations in the known-failing guided cache path.
    pub fn from_artifact_cfg_shape_matched_diagnostic(
        backend: B,
        artifact: &Artifact,
    ) -> Result<Self> {
        let mut model = Self::from_artifact_cfg_internals_diagnostic(backend, artifact, true)?;
        model.cfg_shape_matched_diagnostic = true;
        Ok(model)
    }

    /// Experimental CFG: recompute a whole level without cache when its top-two
    /// margin is small. This heuristic has no full-corpus correctness guarantee.
    pub fn from_artifact_cfg_recompute_diagnostic(
        backend: B,
        artifact: &Artifact,
        margin: f32,
    ) -> Result<Self> {
        if !margin.is_finite() || !(0.0..=1.0).contains(&margin) || margin == 0.0 {
            return Err(Error::Input("invalid WMT14 diagnostic margin".into()));
        }
        let mut model = Self::from_artifact_cached_cfg_diagnostic(backend, artifact)?;
        model.cfg_recompute_margin_diagnostic = Some(margin);
        Ok(model)
    }

    /// Investigate the known-failing guided cache path; never a qualified profile.
    pub fn from_artifact_cached_cfg_diagnostic(backend: B, artifact: &Artifact) -> Result<Self> {
        let mut model = Self::from_artifact_cached_experimental(backend, artifact)?;
        model.cached_cfg_diagnostic = true;
        Ok(model)
    }
}

#[cfg(feature = "wmt14-experimental")]
const WMT14_HEAD_REJECTED: &str = "tied or nonfinite WMT14 head maximum is unsupported";

#[cfg(feature = "wmt14-experimental")]
impl<B: ncm_core::GeneratorBackend> Wmt14Teacher<B> {
    /// Keep raw selector sentinels and check them together before decoding.
    /// Intermediate lookups clamp sentinels to valid indices while evaluation
    /// proceeds. An invalid head still rejects the whole request.
    pub fn with_deferred_head_check_experimental(mut self) -> Self {
        self.deferred_head_check_experimental = true;
        self
    }

    pub fn run_traced(
        &self,
        prefix_tokens: &[u32],
        forced_codes: &[u32],
        batch: usize,
        trace: &mut impl crate::TraceSink<B>,
    ) -> Result<()> {
        self.run_internal(prefix_tokens, Some(forced_codes), batch, trace, None)
            .map(|_| ())
    }

    /// Experimental CFG-0 top-1 generation; tied or nonfinite heads fail.
    pub fn generate_top1_traced(
        &self,
        prefix_tokens: &[u32],
        batch: usize,
        trace: &mut impl crate::TraceSink<B>,
    ) -> Result<()> {
        self.run_internal(prefix_tokens, None, batch, trace, None)
            .map(|_| ())
    }

    /// Return experimental CFG-zero top-1 outputs for explicit offline validation.
    pub fn generate_top1(&self, prefix_tokens: &[u32], batch: usize) -> Result<Generated<B>> {
        self.run_internal(prefix_tokens, None, batch, &mut crate::NoTrace, None)
    }

    /// Try the original input shape first. If the WMT14 head rejects a
    /// maximum, repeat rows to one batch of eight and return only the originals.
    /// The second generation uses the same head guard. The flag reports a retry.
    pub fn generate_top1_with_padded_retry(
        &self,
        prefix_tokens: &[u32],
        batch: usize,
        scale: f64,
    ) -> Result<(Generated<B>, bool)> {
        if !(1..=4).contains(&batch) || prefix_tokens.len() != batch * 96 {
            return Err(Error::Input(
                "WMT14 padded retry requires one to four prefixes".into(),
            ));
        }
        let generate = |tokens: &[u32], rows| {
            if scale == 0.0 {
                self.generate_top1(tokens, rows)
            } else {
                self.generate_cfg_top1(tokens, rows, scale)
            }
        };
        match generate(prefix_tokens, batch) {
            Err(Error::Backend(message)) if message == WMT14_HEAD_REJECTED => {}
            result => return result.map(|output| (output, false)),
        }
        let padded: Vec<_> = prefix_tokens.iter().copied().cycle().take(8 * 96).collect();
        let output = generate(&padded, 8)?;
        Ok((
            Generated {
                codes: self.backend.narrow_indices(&output.codes, 0, 0, batch)?,
                logits: self.backend.narrow(&output.logits, 0, 0, batch)?,
                ids: self.backend.narrow_indices(&output.ids, 0, 0, batch)?,
            },
            true,
        ))
    }

    /// Explicit cached base generation in groups of at most four, with host output.
    /// This scheduling changes execution batch shape and remains experimental.
    pub fn generate_top1_microbatched(
        &self,
        prefix_tokens: &[u32],
        batch: usize,
    ) -> Result<Wmt14HostOutput> {
        if !self.cached_prefix_experimental
            || !(1..=32).contains(&batch)
            || prefix_tokens.len() != batch * 96
            || prefix_tokens.iter().any(|&x| x >= 151680)
            || prefix_tokens
                .as_chunks::<96>()
                .0
                .iter()
                .any(|row| row.iter().all(|&x| x == 151643))
        {
            return Err(Error::Input(
                "invalid WMT14 microbatch inputs or backend".into(),
            ));
        }
        let mut codes = Vec::with_capacity(batch * 313);
        let mut ids = Vec::with_capacity(batch * 64);
        for chunk in prefix_tokens.chunks(4 * 96) {
            let result = self.generate_top1(chunk, chunk.len() / 96)?;
            codes.extend(self.backend.read_indices(&result.codes)?);
            ids.extend(self.backend.read_indices(&result.ids)?);
        }
        if codes.len() != batch * 313 || ids.len() != batch * 64 {
            return Err(Error::Backend(
                "invalid WMT14 microbatch output shape".into(),
            ));
        }
        Ok(Wmt14HostOutput { codes, ids })
    }

    /// Explicit unqualified CFG recomputation in groups of at most four.
    /// The heuristic and altered batch shape require independent qualification.
    pub fn generate_cfg_recompute_microbatched(
        &self,
        prefix_tokens: &[u32],
        batch: usize,
        scale: f64,
    ) -> Result<Wmt14HostOutput> {
        if self.cfg_recompute_margin_diagnostic.is_none()
            || !(1..=16).contains(&batch)
            || !scale.is_finite()
            || scale <= 0.0
            || prefix_tokens.len() != batch * 96
            || prefix_tokens.iter().any(|&x| x >= 151680)
            || prefix_tokens
                .as_chunks::<96>()
                .0
                .iter()
                .any(|row| row.iter().all(|&x| x == 151643))
        {
            return Err(Error::Input(
                "invalid WMT14 CFG diagnostic microbatch inputs or backend".into(),
            ));
        }
        let mut codes = Vec::with_capacity(batch * 313);
        let mut ids = Vec::with_capacity(batch * 64);
        for chunk in prefix_tokens.chunks(4 * 96) {
            let result = self.generate_cfg_top1(chunk, chunk.len() / 96, scale)?;
            codes.extend(self.backend.read_indices(&result.codes)?);
            ids.extend(self.backend.read_indices(&result.ids)?);
        }
        if codes.len() != batch * 313 || ids.len() != batch * 64 {
            return Err(Error::Backend(
                "invalid WMT14 CFG diagnostic microbatch output".into(),
            ));
        }
        Ok(Wmt14HostOutput { codes, ids })
    }

    /// Exact uncached CFG in explicit groups of at most four. Unlike the
    /// recomputation diagnostic, this path has no score-margin heuristic.
    pub fn generate_cfg_uncached_microbatched(
        &self,
        prefix_tokens: &[u32],
        batch: usize,
        scale: f64,
    ) -> Result<Wmt14HostOutput> {
        self.generate_cfg_uncached_chunks(prefix_tokens, batch, scale, 4)
    }

    /// Explicit experimental eight-row grouping; independent numerical
    /// qualification is required because it changes the execution shape.
    pub fn generate_cfg_uncached_eight_diagnostic(
        &self,
        prefix_tokens: &[u32],
        batch: usize,
        scale: f64,
    ) -> Result<Wmt14HostOutput> {
        self.generate_cfg_uncached_chunks(prefix_tokens, batch, scale, 8)
    }

    fn generate_cfg_uncached_chunks(
        &self,
        prefix_tokens: &[u32],
        batch: usize,
        scale: f64,
        chunk_size: usize,
    ) -> Result<Wmt14HostOutput> {
        if self.cached_prefix_experimental
            || !(1..=16).contains(&batch)
            || !scale.is_finite()
            || scale <= 0.0
            || prefix_tokens.len() != batch * 96
            || prefix_tokens.iter().any(|&x| x >= 151680)
            || prefix_tokens
                .as_chunks::<96>()
                .0
                .iter()
                .any(|row| row.iter().all(|&x| x == 151643))
        {
            return Err(Error::Input(
                "invalid WMT14 uncached CFG microbatch inputs".into(),
            ));
        }
        let mut codes = Vec::with_capacity(batch * 313);
        let mut ids = Vec::with_capacity(batch * 64);
        for chunk in prefix_tokens.chunks(chunk_size * 96) {
            let result = self.generate_cfg_top1(chunk, chunk.len() / 96, scale)?;
            codes.extend(self.backend.read_indices(&result.codes)?);
            ids.extend(self.backend.read_indices(&result.ids)?);
        }
        if codes.len() != batch * 313 || ids.len() != batch * 64 {
            return Err(Error::Backend(
                "invalid WMT14 uncached CFG microbatch output".into(),
            ));
        }
        Ok(Wmt14HostOutput { codes, ids })
    }

    /// Experimental guided top-1 with the upstream EOS/PAD null prefix.
    pub fn generate_cfg_top1_traced(
        &self,
        prefix_tokens: &[u32],
        batch: usize,
        scale: f64,
        trace: &mut impl crate::TraceSink<B>,
    ) -> Result<()> {
        let combined = wmt14_cfg_prefixes(prefix_tokens, batch, scale)?;
        self.run_internal(&combined, None, batch * 2, trace, Some((batch, scale)))
            .map(|_| ())
    }

    /// Return experimental guided top-1 outputs for explicit offline validation.
    pub fn generate_cfg_top1(
        &self,
        prefix_tokens: &[u32],
        batch: usize,
        scale: f64,
    ) -> Result<Generated<B>> {
        let combined = wmt14_cfg_prefixes(prefix_tokens, batch, scale)?;
        self.run_internal(
            &combined,
            None,
            batch * 2,
            &mut crate::NoTrace,
            Some((batch, scale)),
        )
    }

    fn run_internal(
        &self,
        prefix_tokens: &[u32],
        forced_codes: Option<&[u32]>,
        batch: usize,
        trace: &mut impl crate::TraceSink<B>,
        cfg: Option<(usize, f64)>,
    ) -> Result<Generated<B>> {
        if cfg.is_some() && self.cached_prefix_experimental && !self.cached_cfg_diagnostic {
            return Err(Error::Input(
                "WMT14 cached CFG is unsupported; use the no-cache model".into(),
            ));
        }
        let levels = self.codec.config().levels();
        let total: usize = levels.iter().sum();
        if !(1..=32).contains(&batch)
            || prefix_tokens.len() != batch * 96
            || forced_codes.is_some_and(|codes| codes.len() != batch * total)
            || prefix_tokens.iter().any(|&x| x >= 151680)
            || forced_codes.is_some_and(|codes| codes.iter().any(|&x| x >= 4096))
            || prefix_tokens
                .as_chunks::<96>()
                .0
                .iter()
                .any(|row| row.iter().all(|&x| x == 151643))
        {
            return Err(Error::Input("invalid WMT14 teacher inputs".into()));
        }
        let deferred = self.deferred_head_check_experimental && forced_codes.is_none();
        let b = &self.backend;
        let prefix_ids = b.upload_indices(prefix_tokens, &[batch, 96])?;
        let prefix = b.project_bias(
            &b.embedding(self.codec.weight("encoder.wte.weight")?, &prefix_ids)?,
            &self.input.weight,
            &self.input.bias,
        )?;
        let forced = forced_codes
            .map(|codes| b.upload_indices(codes, &[batch, total]))
            .transpose()?;
        let observed_prefix = if let Some((original, _)) = cfg {
            b.narrow_indices(&prefix_ids, 0, 0, original)?
        } else {
            prefix_ids.clone()
        };
        trace.indices("generator.prefix_ids", &observed_prefix)?;
        if let Some(codes) = &forced {
            trace.indices("generator.forced_codes", codes)?;
        }
        let mut cumulative: Option<B::Tensor> = None;
        let mut selected_levels = Vec::with_capacity(levels.len());
        let mut offset = 0;
        let mut prefix_cache = None;
        for (level, &length) in levels.iter().enumerate() {
            let target_length = if level == 0 { 0 } else { length };
            let seq = 96 + target_length;
            let base = &self.geometry[level];
            let mut padding = Vec::with_capacity(batch * seq);
            for row in prefix_tokens.as_chunks::<96>().0 {
                padding.extend(
                    row.iter()
                        .map(|&x| if x == 151643 { f32::NEG_INFINITY } else { 0.0 }),
                );
                padding.extend(std::iter::repeat_n(0.0, target_length));
            }
            let mask = b.add(&base.mask, &b.upload_f32(&padding, &[batch, 1, 1, seq])?)?;
            let geometry = LevelGeometry {
                mask,
                cosine: base.cosine.clone(),
                sine: base.sine.clone(),
            };
            let mut recompute_target = None;
            let hidden = if level == 0 {
                if self.cached_prefix_experimental {
                    let (hidden, cache) = wmt14_prefix_transformer_cached(
                        b,
                        &self.blocks,
                        &self.final_norm,
                        &geometry,
                        &prefix,
                        if self.cfg_internals_diagnostic {
                            Some(trace as &mut dyn crate::TraceSink<B>)
                        } else {
                            None
                        },
                    )?;
                    prefix_cache = Some(cache);
                    hidden
                } else {
                    wmt14_prefix_transformer(b, &self.blocks, &self.final_norm, &geometry, &prefix)?
                }
            } else {
                let latent = b.area1d(cumulative.as_ref().expect("preceding level"), length)?;
                let target = b.add(
                    &b.project_bias(&latent, &self.input.weight, &self.input.bias)?,
                    &b.narrow(&self.level_embedding, 0, level - 1, 1)?,
                )?;
                if self.cfg_recompute_margin_diagnostic.is_some() && cfg.is_some() {
                    recompute_target = Some(target.clone());
                }
                if let Some(cache) = &prefix_cache {
                    wmt14_cached_target_transformer(
                        b,
                        &self.blocks,
                        &self.final_norm,
                        &geometry,
                        &target,
                        cache,
                        if self.cfg_shape_matched_diagnostic {
                            Some(&prefix)
                        } else {
                            None
                        },
                        if self.cfg_internals_diagnostic && level == 1 {
                            Some(trace as &mut dyn crate::TraceSink<B>)
                        } else {
                            None
                        },
                    )?
                } else {
                    decode_transformer_observed(
                        b,
                        &self.blocks,
                        &self.final_norm,
                        &geometry,
                        &prefix,
                        &target,
                        96,
                        12,
                        768,
                        length,
                        false,
                        if self.cfg_internals_diagnostic && level == 1 {
                            Some(trace as &mut dyn crate::TraceSink<B>)
                        } else {
                            None
                        },
                    )?
                }
            };
            let hidden = if level == 0 {
                let mut rows = Vec::with_capacity(batch);
                for (i, row) in prefix_tokens.as_chunks::<96>().0.iter().enumerate() {
                    let position = wmt14_prefix_position(row);
                    rows.push(b.narrow(&b.narrow(&hidden, 0, i, 1)?, 1, position, 1)?);
                }
                b.concat_f32(&rows, 0)?
            } else {
                hidden
            };
            let head_logits = b.project(&hidden, &self.heads[level])?;
            let mut head_logits = if let Some((original, scale)) = cfg {
                if self.cfg_internals_diagnostic {
                    trace.tensor(
                        &format!("diagnostic.level_{level:02}.branches"),
                        &head_logits,
                    )?;
                }
                let t = scale * level as f64 / (levels.len() - 1) as f64;
                let conditioned = b.narrow(&head_logits, 0, 0, original)?;
                let null = b.narrow(&head_logits, 0, original, original)?;
                b.sub(&b.scale(&conditioned, 1.0 + t)?, &b.scale(&null, t)?)?
            } else {
                head_logits
            };
            if let (Some(margin), Some(target), Some((original, scale))) =
                (self.cfg_recompute_margin_diagnostic, recompute_target, cfg)
            {
                trace.tensor(
                    &format!("diagnostic.cached.level_{level:02}.logits"),
                    &head_logits,
                )?;
                let uncertain = wmt14_margin_uncertain(&b.read_f32(&head_logits)?, margin)?;
                trace.indices(
                    &format!("diagnostic.recomputed.level_{level:02}"),
                    &b.upload_indices(&[u32::from(uncertain)], &[1])?,
                )?;
                if uncertain {
                    let hidden = decode_transformer(
                        b,
                        &self.blocks,
                        &self.final_norm,
                        &geometry,
                        &prefix,
                        &target,
                        96,
                        12,
                        768,
                        length,
                    )?;
                    let logits = b.project(&hidden, &self.heads[level])?;
                    let t = scale * level as f64 / (levels.len() - 1) as f64;
                    let conditioned = b.narrow(&logits, 0, 0, original)?;
                    let null = b.narrow(&logits, 0, original, original)?;
                    head_logits = b.sub(&b.scale(&conditioned, 1.0 + t)?, &b.scale(&null, t)?)?;
                }
            }
            trace.tensor(&format!("generator.level_{level:02}.logits"), &head_logits)?;
            let selected = if let Some(codes) = &forced {
                b.narrow_indices(codes, 1, offset, length)?
            } else {
                let selected = b.unique_max_4096(&head_logits)?;
                if !deferred && b.read_indices(&selected)?.iter().any(|&x| x >= 4096) {
                    return Err(Error::Backend(WMT14_HEAD_REJECTED.into()));
                }
                if cfg.is_some() {
                    b.concat_indices(&[selected.clone(), selected], 0)?
                } else {
                    selected
                }
            };
            offset += length;
            selected_levels.push(selected.clone());
            let book = self
                .codec
                .weight(&format!("vq.embeddings.{level}.weight"))?;
            let embedding = if deferred {
                b.sampled_embedding_4096(book, &selected)?
            } else {
                b.embedding(book, &selected)?
            };
            let contribution = self
                .codec
                .refine(&self.codec.upsample(&embedding, level)?, level)?;
            let sum = if let Some(old) = &cumulative {
                b.add(old, &contribution)?
            } else {
                contribution
            };
            let observed_sum = if let Some((original, _)) = cfg {
                b.narrow(&sum, 0, 0, original)?
            } else {
                sum.clone()
            };
            trace.tensor(&format!("reconstruction.level_{level:02}"), &observed_sum)?;
            cumulative = Some(sum);
        }
        let codes = b.concat_indices(&selected_levels, 1)?;
        let codes = if let Some((original, _)) = cfg {
            b.narrow_indices(&codes, 0, 0, original)?
        } else {
            codes
        };
        if deferred && b.read_indices(&codes)?.iter().any(|&x| x >= 4096) {
            return Err(Error::Backend(WMT14_HEAD_REJECTED.into()));
        }
        if forced.is_none() {
            trace.indices("generator.codes", &codes)?;
        }
        let latent = cumulative.expect("validated levels");
        let latent = if let Some((original, _)) = cfg {
            b.narrow(&latent, 0, 0, original)?
        } else {
            latent
        };
        let (logits, ids) = self
            .codec
            .decode_latent_outputs(&latent, &mut crate::NoTrace)?;
        trace.tensor("generator.final_logits", &logits)?;
        trace.indices("generator.greedy_ids", &ids)?;
        Ok(Generated { codes, logits, ids })
    }
}

#[cfg(feature = "wmt14-experimental")]
fn wmt14_margin_uncertain(scores: &[f32], margin: f32) -> Result<bool> {
    if scores.is_empty()
        || !scores.len().is_multiple_of(4096)
        || !margin.is_finite()
        || margin <= 0.0
    {
        return Err(Error::Input("invalid WMT14 diagnostic scores".into()));
    }
    let mut uncertain = false;
    for row in scores.as_chunks::<4096>().0 {
        let mut first = f32::NEG_INFINITY;
        let mut second = f32::NEG_INFINITY;
        for &value in row {
            if !value.is_finite() {
                return Err(Error::Backend("nonfinite WMT14 diagnostic scores".into()));
            }
            if value > first {
                second = first;
                first = value;
            } else if value > second {
                second = value;
            }
        }
        uncertain |= first - second <= margin;
    }
    Ok(uncertain)
}

#[cfg(all(test, feature = "wmt14-experimental"))]
#[test]
fn wmt14_diagnostic_margin_detects_ties_and_rejects_bad_values() {
    let mut row = vec![0.0; 4096];
    row[0] = 2.0;
    row[1] = 1.5;
    assert!(!wmt14_margin_uncertain(&row, 0.1).unwrap());
    row[1] = 2.0;
    assert!(wmt14_margin_uncertain(&row, 0.1).unwrap());
    row[2] = f32::NAN;
    assert!(wmt14_margin_uncertain(&row, 0.1).is_err());
    assert!(wmt14_margin_uncertain(&row[..1], 0.1).is_err());
    assert!(wmt14_margin_uncertain(&row, f32::INFINITY).is_err());
}

#[cfg(feature = "wmt14-experimental")]
fn wmt14_cfg_prefixes(prefix_tokens: &[u32], batch: usize, scale: f64) -> Result<Vec<u32>> {
    if !(1..=16).contains(&batch)
        || prefix_tokens.len() != batch * 96
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Err(Error::Input("invalid WMT14 CFG inputs".into()));
    }
    let mut combined = Vec::with_capacity(batch * 192);
    combined.extend_from_slice(prefix_tokens);
    for _ in 0..batch {
        combined.push(151645);
        combined.extend(std::iter::repeat_n(151643, 95));
    }
    Ok(combined)
}

#[cfg(feature = "wmt14-experimental")]
fn wmt14_prefix_position(row: &[u32; 96]) -> usize {
    row.iter()
        .filter(|&&x| x != 151643)
        .count()
        .saturating_sub(1)
        .min(95)
}

#[cfg(all(test, feature = "wmt14-experimental"))]
#[test]
fn wmt14_level_zero_uses_upstream_nonpadding_count() {
    let mut row = [151643; 96];
    row[0] = 10;
    row[1] = 11;
    assert_eq!(wmt14_prefix_position(&row), 1);
    row[10] = 12;
    assert_eq!(wmt14_prefix_position(&row), 2);
    assert_eq!(wmt14_prefix_position(&[10; 96]), 95);
}

#[cfg(all(test, feature = "wmt14-experimental"))]
#[test]
fn wmt14_cfg_uses_eos_then_padding_and_rejects_bad_inputs() {
    let original = vec![7; 192];
    let both = wmt14_cfg_prefixes(&original, 2, 1.0).unwrap();
    assert_eq!(&both[..192], original);
    for row in both[192..].as_chunks::<96>().0 {
        assert_eq!(row[0], 151645);
        assert!(row[1..].iter().all(|&x| x == 151643));
    }
    assert!(wmt14_cfg_prefixes(&original, 2, f64::NAN).is_err());
    assert!(wmt14_cfg_prefixes(&original, 2, 0.0).is_err());
    assert!(wmt14_cfg_prefixes(&original, 3, 1.0).is_err());
}

#[cfg(feature = "wmt14-experimental")]
fn wmt14_prefix_transformer<B: ncm_core::GeneratorBackend>(
    b: &B,
    blocks: &[GeneratorBlock<B::Tensor>],
    final_norm: &Affine<B::Tensor>,
    geometry: &LevelGeometry<B::Tensor>,
    prefix: &B::Tensor,
) -> Result<B::Tensor> {
    let batch = b.shape(prefix)[0];
    let linear = |a: &Affine<B::Tensor>, x: &B::Tensor| b.project_bias(x, &a.weight, &a.bias);
    let norm =
        |a: &Affine<B::Tensor>, x: &B::Tensor| b.layer_norm(x, &a.weight, Some(&a.bias), 1e-6);
    let mut x = prefix.clone();
    for block in blocks {
        let qkv = linear(&block.qkv, &norm(&block.ln_1, &x)?)?;
        let [q, k, v] = b.split_qkv(&qkv, batch, 96, 12, 64)?;
        let q = b.rms_rope(&q, &block.q_norm, &geometry.cosine, &geometry.sine, 1e-6)?;
        let k = b.rms_rope(&k, &block.k_norm, &geometry.cosine, &geometry.sine, 1e-6)?;
        let attn = b.merge_heads(
            &b.attention(&q, &k, &v, Some(&geometry.mask))?,
            batch,
            96,
            12,
            64,
        )?;
        x = b.add(&x, &linear(&block.attention_projection, &attn)?)?;
        let hidden = b.gelu(&linear(&block.mlp_in, &norm(&block.ln_2, &x)?)?)?;
        x = b.add(&x, &linear(&block.mlp_out, &hidden)?)?;
    }
    norm(final_norm, &x)
}

#[cfg(feature = "wmt14-experimental")]
type Wmt14PrefixCache<T> = Vec<(T, T)>;

#[cfg(feature = "wmt14-experimental")]
fn wmt14_prefix_transformer_cached<B: ncm_core::GeneratorBackend>(
    b: &B,
    blocks: &[GeneratorBlock<B::Tensor>],
    final_norm: &Affine<B::Tensor>,
    geometry: &LevelGeometry<B::Tensor>,
    prefix: &B::Tensor,
    mut trace: Option<&mut dyn crate::TraceSink<B>>,
) -> Result<(B::Tensor, Wmt14PrefixCache<B::Tensor>)> {
    let batch = b.shape(prefix)[0];
    let linear = |a: &Affine<B::Tensor>, x: &B::Tensor| b.project_bias(x, &a.weight, &a.bias);
    let norm =
        |a: &Affine<B::Tensor>, x: &B::Tensor| b.layer_norm(x, &a.weight, Some(&a.bias), 1e-6);
    let mut x = prefix.clone();
    let mut cache = Vec::with_capacity(blocks.len());
    for (index, block) in blocks.iter().enumerate() {
        let normalized = norm(&block.ln_1, &x)?;
        if index == 0
            && let Some(trace) = trace.as_deref_mut()
        {
            trace.tensor("diagnostic.prefix.first_norm", &normalized)?;
        }
        let qkv = linear(&block.qkv, &normalized)?;
        if index == 0
            && let Some(trace) = trace.as_deref_mut()
        {
            trace.tensor("diagnostic.prefix.first_qkv", &qkv)?;
        }
        let [q, k, v] = b.split_qkv(&qkv, batch, 96, 12, 64)?;
        let q = b.rms_rope(&q, &block.q_norm, &geometry.cosine, &geometry.sine, 1e-6)?;
        let k = b.rms_rope(&k, &block.k_norm, &geometry.cosine, &geometry.sine, 1e-6)?;
        let attn = b.merge_heads(
            &b.attention(&q, &k, &v, Some(&geometry.mask))?,
            batch,
            96,
            12,
            64,
        )?;
        if index == 0
            && let Some(trace) = trace.as_deref_mut()
        {
            trace.tensor("diagnostic.prefix.first_attention", &attn)?;
        }
        cache.push((k, v));
        x = b.add(&x, &linear(&block.attention_projection, &attn)?)?;
        let normalized = norm(&block.ln_2, &x)?;
        let projected = linear(&block.mlp_in, &normalized)?;
        let hidden = b.gelu(&projected)?;
        if index == 0
            && let Some(trace) = trace.as_deref_mut()
        {
            trace.tensor("diagnostic.prefix.first_residual", &x)?;
            trace.tensor("diagnostic.prefix.second_norm", &normalized)?;
            trace.tensor("diagnostic.prefix.first_mlp_in", &projected)?;
            trace.tensor("diagnostic.prefix.first_mlp_gelu", &hidden)?;
        }
        x = b.add(&x, &linear(&block.mlp_out, &hidden)?)?;
        if let Some(trace) = trace.as_deref_mut() {
            trace.tensor(&format!("diagnostic.prefix.block_{index:02}"), &x)?;
        }
    }
    Ok((norm(final_norm, &x)?, cache))
}

#[cfg(feature = "wmt14-experimental")]
#[allow(clippy::too_many_arguments)]
fn wmt14_cached_target_transformer<B: ncm_core::GeneratorBackend>(
    b: &B,
    blocks: &[GeneratorBlock<B::Tensor>],
    final_norm: &Affine<B::Tensor>,
    geometry: &LevelGeometry<B::Tensor>,
    target: &B::Tensor,
    cache: &Wmt14PrefixCache<B::Tensor>,
    projection_padding: Option<&B::Tensor>,
    mut trace: Option<&mut dyn crate::TraceSink<B>>,
) -> Result<B::Tensor> {
    let shape = b.shape(target);
    let [batch, length, 768] = shape.as_slice() else {
        return Err(Error::Input("invalid WMT14 cached target shape".into()));
    };
    if cache.len() != blocks.len() || *batch == 0 || *length == 0 {
        return Err(Error::Input("invalid WMT14 prefix cache".into()));
    }
    let cosine = b.narrow(&geometry.cosine, 0, 96, *length)?;
    let sine = b.narrow(&geometry.sine, 0, 96, *length)?;
    let mask = b.narrow(&geometry.mask, 2, 96, *length)?;
    let linear = |a: &Affine<B::Tensor>, x: &B::Tensor| {
        if let Some(padding) = projection_padding {
            let width = b.shape(x)[2];
            let filler = if width == 768 {
                padding.clone()
            } else {
                b.upload_f32(&vec![0.0; *batch * 96 * width], &[*batch, 96, width])?
            };
            let full = b.concat_f32(&[filler, x.clone()], 1)?;
            b.narrow(&b.project_bias(&full, &a.weight, &a.bias)?, 1, 96, *length)
        } else {
            b.project_bias(x, &a.weight, &a.bias)
        }
    };
    let norm = |a: &Affine<B::Tensor>, x: &B::Tensor| {
        if *batch < 16 {
            b.layer_norm(x, &a.weight, Some(&a.bias), 1e-6)
        } else {
            b.layer_norm_accelerated(x, &a.weight, Some(&a.bias), 1e-6)
        }
    };
    let mut x = target.clone();
    if let Some(trace) = trace.as_deref_mut() {
        trace.tensor("diagnostic.level_01.input", &x)?;
    }
    for (index, (block, (prefix_k, prefix_v))) in blocks.iter().zip(cache).enumerate() {
        let normalized = norm(&block.ln_1, &x)?;
        if index == 0
            && let Some(trace) = trace.as_deref_mut()
        {
            trace.tensor("diagnostic.level_01.first_norm", &normalized)?;
        }
        let qkv = linear(&block.qkv, &normalized)?;
        if index == 0
            && let Some(trace) = trace.as_deref_mut()
        {
            trace.tensor("diagnostic.level_01.first_qkv", &qkv)?;
        }
        let [q, k, v] = b.split_qkv(&qkv, *batch, *length, 12, 64)?;
        let q = b.rms_rope(&q, &block.q_norm, &cosine, &sine, 1e-6)?;
        let k = b.rms_rope(&k, &block.k_norm, &cosine, &sine, 1e-6)?;
        let keys = b.concat_f32(&[prefix_k.clone(), k], 2)?;
        let values = b.concat_f32(&[prefix_v.clone(), v], 2)?;
        let attn = if projection_padding.is_some() {
            let full_q = b.concat_f32(&[prefix_k.clone(), q], 2)?;
            let full = b.attention(&full_q, &keys, &values, Some(&geometry.mask))?;
            b.merge_heads(&b.narrow(&full, 2, 96, *length)?, *batch, *length, 12, 64)?
        } else {
            b.merge_heads(
                &b.attention(&q, &keys, &values, Some(&mask))?,
                *batch,
                *length,
                12,
                64,
            )?
        };
        if index == 0
            && let Some(trace) = trace.as_deref_mut()
        {
            trace.tensor("diagnostic.level_01.first_attention", &attn)?;
        }
        x = b.add(&x, &linear(&block.attention_projection, &attn)?)?;
        if index == 0
            && let Some(trace) = trace.as_deref_mut()
        {
            trace.tensor("diagnostic.level_01.first_residual", &x)?;
        }
        let hidden = b.gelu(&linear(&block.mlp_in, &norm(&block.ln_2, &x)?)?)?;
        x = b.add(&x, &linear(&block.mlp_out, &hidden)?)?;
        if let Some(trace) = trace.as_deref_mut() {
            trace.tensor(&format!("diagnostic.level_01.block_{index:02}"), &x)?;
        }
    }
    norm(final_norm, &x)
}

/// TinyStories teacher-forced execution and seeded stochastic generation.
/// Arbitrary padded prefixes are not yet implemented.
pub struct TinyStoriesTeacher<B: ncm_core::GeneratorBackend> {
    codec: crate::Codec<B>,
    backend: B,
    input: Affine<B::Tensor>,
    level_embedding: B::Tensor,
    final_norm: Affine<B::Tensor>,
    blocks: Box<[GeneratorBlock<B::Tensor>]>,
    heads: Box<[TinyHead<B::Tensor>]>,
    geometry: Vec<LevelGeometry<B::Tensor>>,
    initial_distribution: Vec<f32>,
    request_lock: std::sync::Mutex<()>,
}
struct TinyHead<T> {
    projection: T,
    squared_norm: T,
    mean_squared_norm: f32,
    inverse_temperature: f32,
}
impl<B: ncm_core::GeneratorBackend + Clone> TinyStoriesTeacher<B> {
    pub fn from_artifact(backend: B, artifact: &Artifact) -> Result<Self> {
        // Check the complete generator and inference inventory before any upload.
        let inspection = TinyStoriesGeneratorInspection::from_artifact(artifact)?;
        let codec = crate::Codec::from_artifact(backend.clone(), artifact)?;
        let bytes = artifact
            .file("generator.safetensors")
            .map_err(|e| Error::Weights(e.to_string()))?;
        let tensors = SafeTensors::deserialize(bytes).map_err(|e| Error::Weights(e.to_string()))?;
        let mut weights = BTreeMap::new();
        let mut scales = Vec::with_capacity(11);
        for i in 0..11 {
            let t = tensors
                .tensor(&format!("lm_heads.{i}.log_inv_T"))
                .map_err(|e| Error::Weights(e.to_string()))?;
            let log_scale = f32::from_le_bytes(
                t.data()
                    .try_into()
                    .map_err(|_| Error::Weights("invalid head scale".into()))?,
            );
            let scale = log_scale.exp();
            if !scale.is_finite() || scale <= 0.0 {
                return Err(Error::Weights("invalid head temperature".into()));
            }
            scales.push(scale);
        }
        for name in tensors.names() {
            if name == "lvl_1L" || name == "attn_mask_for_masking" || name.ends_with(".log_inv_T") {
                continue;
            }
            let t = tensors
                .tensor(name)
                .map_err(|e| Error::Weights(e.to_string()))?;
            let values: Vec<f32> = t
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|x| f32::from_le_bytes(*x))
                .collect();
            weights.insert(name.to_owned(), backend.upload_f32(&values, t.shape())?);
        }
        let input = take_affine(&mut weights, "transformer.wte")?;
        let level_embedding = take(&mut weights, "transformer.wle.weight")?;
        let final_norm = take_affine(&mut weights, "transformer.ln_f")?;
        let mut blocks = Vec::with_capacity(12);
        for i in 0..12 {
            let p = format!("transformer.blocks.{i}");
            blocks.push(GeneratorBlock {
                ln_1: take_affine(&mut weights, &format!("{p}.ln_1"))?,
                qkv: take_affine(&mut weights, &format!("{p}.attn.c_attn"))?,
                q_norm: take(&mut weights, &format!("{p}.attn.q_norm.weight"))?,
                k_norm: take(&mut weights, &format!("{p}.attn.k_norm.weight"))?,
                attention_projection: take_affine(&mut weights, &format!("{p}.attn.c_proj"))?,
                ln_2: take_affine(&mut weights, &format!("{p}.ln_2"))?,
                mlp_in: take_affine(&mut weights, &format!("{p}.mlp.c_fc"))?,
                mlp_out: take_affine(&mut weights, &format!("{p}.mlp.c_proj"))?,
            });
        }
        let mut heads = Vec::with_capacity(11);
        for (i, inverse_temperature) in scales.into_iter().enumerate() {
            let book = codec.weight(&format!("vq.embeddings.{}.weight", i + 1))?;
            let norm = backend.sum_last(&backend.square(book)?)?;
            let total = backend.sum_last(&backend.reshape(&norm, &[1, 4096])?)?;
            let mean = backend.read_f32(&backend.scale(&total, 1.0 / 4096.0)?)?[0].max(1e-12);
            if !mean.is_finite() {
                return Err(Error::Weights("invalid codebook norm".into()));
            }
            heads.push(TinyHead {
                projection: take(&mut weights, &format!("lm_heads.{i}.proj.weight"))?,
                squared_norm: backend.reshape(&norm, &[1, 1, 4096])?,
                mean_squared_norm: mean,
                inverse_temperature,
            });
        }
        if !weights.is_empty() {
            return Err(Error::Weights(
                "unbound TinyStories generator tensor".into(),
            ));
        }
        let geometry =
            prepare_geometry(&backend, &codec.config().levels()[1..], 1, 768, 12, 100000)?;
        Ok(Self {
            codec,
            backend,
            input,
            level_embedding,
            final_norm,
            blocks: blocks.into_boxed_slice(),
            heads: heads.into_boxed_slice(),
            geometry,
            initial_distribution: inspection.initial_distribution,
            request_lock: std::sync::Mutex::new(()),
        })
    }
}
impl<B: ncm_core::GeneratorBackend> TinyStoriesTeacher<B> {
    pub fn initial_distribution(&self) -> &[f32] {
        &self.initial_distribution
    }

    /// Experimental TinyStories generation at temperature 1, top-k 0, top-p 1
    /// and CFG 0. SplitMix64 supplies independent controlled draws; its seed
    /// and the inverse-CDF sampler are not PyTorch's random-number stream.
    /// Prefix padding and other generation settings are explicitly unsupported.
    pub fn sample_seeded(&self, prefix: &[u32], batch: usize, seed: u64) -> Result<Generated<B>> {
        let _request = self
            .request_lock
            .lock()
            .map_err(|_| Error::Backend("generator request lock poisoned".into()))?;
        if !(1..=128).contains(&batch)
            || prefix.len() != batch
            || prefix.iter().any(|&p| p >= 8192 || p == 1)
        {
            return Err(Error::Input("invalid TinyStories prefix or batch".into()));
        }
        let b = &self.backend;
        let levels = self.codec.config().levels();
        let mut random = SplitMix64(seed);
        let sum: f64 = self
            .initial_distribution
            .iter()
            .map(|&x| f64::from(x))
            .sum();
        let mut initial = Vec::with_capacity(batch);
        for _ in 0..batch {
            let threshold = ((random.next() >> 11) as f64 / 9007199254740992.0) * sum;
            let mut cumulative = 0.0f64;
            let selected = self
                .initial_distribution
                .iter()
                .position(|&p| {
                    cumulative += f64::from(p);
                    threshold < cumulative
                })
                .ok_or_else(|| Error::Weights("invalid initial distribution".into()))?;
            initial.push(selected as u32);
        }
        let prefix_ids = b.upload_indices(prefix, &[batch, 1])?;
        let raw = b.embedding(self.codec.weight("encoder.wte.weight")?, &prefix_ids)?;
        let prefix = b.project_bias(&raw, &self.input.weight, &self.input.bias)?;
        let mut selected = b.upload_indices(&initial, &[batch, 1])?;
        let mut codes = vec![selected.clone()];
        let mut cumulative: Option<B::Tensor> = None;
        for level in 0..levels.len() - 1 {
            let book = self
                .codec
                .weight(&format!("vq.embeddings.{level}.weight"))?;
            let embedding = b.sampled_embedding_4096(book, &selected)?;
            let up = self.codec.upsample(&embedding, level)?;
            let contribution = self.codec.refine(&up, level)?;
            let sum = match cumulative {
                Some(ref old) => b.add(old, &contribution)?,
                None => contribution,
            };
            let length = levels[level + 1];
            let latent = b.area1d(&sum, length)?;
            let target = b.project_bias(&latent, &self.input.weight, &self.input.bias)?;
            let target = b.add(&target, &b.narrow(&self.level_embedding, 0, level, 1)?)?;
            let hidden = decode_transformer(
                b,
                &self.blocks,
                &self.final_norm,
                &self.geometry[level],
                &prefix,
                &target,
                1,
                12,
                768,
                length,
            )?;
            let head = &self.heads[level];
            let z = b.project(&hidden, &head.projection)?;
            let book = self
                .codec
                .weight(&format!("vq.embeddings.{}.weight", level + 1))?;
            let score = b.sub(&b.scale(&b.project(&z, book)?, 2.0)?, &head.squared_norm)?;
            let score = b.scale(&score, 1.0 / f64::from(head.mean_squared_norm))?;
            let logits = b.scale(&score, f64::from(head.inverse_temperature))?;
            let words: Vec<u32> = (0..batch * length)
                .map(|_| (random.next() >> 32) as u32)
                .collect();
            let words = b.upload_indices(&words, &[batch, length])?;
            selected = b.categorical_4096(&logits, &words)?;
            codes.push(selected.clone());
            cumulative = Some(sum);
        }
        let codes = b.concat_indices(&codes, 1)?;
        // The native sampler marks invalid logits with u32::MAX. Candle treats
        // that value specially in embedding, so validate before returning any
        // result rather than allowing it to masquerade as an ordinary token.
        if b.read_indices(&codes)?.iter().any(|&v| v >= 4096) {
            return Err(Error::Backend(
                "categorical sampling encountered invalid logits".into(),
            ));
        }
        let last = levels.len() - 1;
        let embedding = b.embedding(
            self.codec.weight(&format!("vq.embeddings.{last}.weight"))?,
            &selected,
        )?;
        let contribution = self
            .codec
            .refine(&self.codec.upsample(&embedding, last)?, last)?;
        let latent = b.add(
            cumulative.as_ref().expect("validated levels"),
            &contribution,
        )?;
        let (logits, ids) = self
            .codec
            .decode_latent_outputs(&latent, &mut crate::NoTrace)?;
        Ok(Generated { codes, logits, ids })
    }

    /// Run fixed codes through each transformer level and trace logits. Prefix
    /// padding is rejected until the upstream additive padding mask is supported.
    pub fn run_traced(
        &self,
        prefix: &[u32],
        forced_codes: &[u32],
        batch: usize,
        trace: &mut impl crate::TraceSink<B>,
    ) -> Result<Generated<B>> {
        let _request = self
            .request_lock
            .lock()
            .map_err(|_| Error::Backend("generator request lock poisoned".into()))?;
        let levels = self.codec.config().levels();
        let total: usize = levels.iter().sum();
        if !(1..=128).contains(&batch)
            || prefix.len() != batch
            || prefix.iter().any(|&p| p >= 8192 || p == 1)
            || batch.checked_mul(total) != Some(forced_codes.len())
            || forced_codes.iter().any(|&c| c >= 4096)
        {
            return Err(Error::Input(
                "invalid TinyStories prefix, forced codes or batch".into(),
            ));
        }
        let b = &self.backend;
        let prefix_ids = b.upload_indices(prefix, &[batch, 1])?;
        let raw = b.embedding(self.codec.weight("encoder.wte.weight")?, &prefix_ids)?;
        let prefix = b.project_bias(&raw, &self.input.weight, &self.input.bias)?;
        let codes = b.upload_indices(forced_codes, &[batch, total])?;
        let mut cumulative: Option<B::Tensor> = None;
        let mut offset = 0;
        for (level, &length) in levels.iter().enumerate() {
            let selected = b.narrow_indices(&codes, 1, offset, length)?;
            offset += length;
            let book = self
                .codec
                .weight(&format!("vq.embeddings.{level}.weight"))?;
            let embedding = b.embedding(book, &selected)?;
            let up = self.codec.upsample(&embedding, level)?;
            let contribution = self.codec.refine(&up, level)?;
            let sum = match cumulative {
                Some(ref old) => b.add(old, &contribution)?,
                None => contribution,
            };
            if level + 1 < levels.len() {
                let next_length = levels[level + 1];
                let latent = b.area1d(&sum, next_length)?;
                let target = b.project_bias(&latent, &self.input.weight, &self.input.bias)?;
                let encoding = b.narrow(&self.level_embedding, 0, level, 1)?;
                let target = b.add(&target, &encoding)?;
                let hidden = decode_transformer(
                    b,
                    &self.blocks,
                    &self.final_norm,
                    &self.geometry[level],
                    &prefix,
                    &target,
                    1,
                    12,
                    768,
                    next_length,
                )?;
                let head = &self.heads[level];
                let z = b.project(&hidden, &head.projection)?;
                let book = self
                    .codec
                    .weight(&format!("vq.embeddings.{}.weight", level + 1))?;
                let score = b.sub(&b.scale(&b.project(&z, book)?, 2.0)?, &head.squared_norm)?;
                let score = b.scale(&score, 1.0 / f64::from(head.mean_squared_norm))?;
                let logits = b.scale(&score, f64::from(head.inverse_temperature))?;
                trace.tensor(&format!("generator.level_{:02}.logits", level + 1), &logits)?;
            }
            trace.tensor(&format!("reconstruction.level_{level:02}"), &sum)?;
            cumulative = Some(sum);
        }
        let (logits, ids) = self
            .codec
            .decode_latent_outputs(&cumulative.expect("validated levels"), &mut crate::NoTrace)?;
        trace.tensor("generator.final_logits", &logits)?;
        trace.indices("generator.output_ids", &ids)?;
        Ok(Generated { codes, logits, ids })
    }
}

/// Fixed-code XSum teacher trace. This does not implement sampling or final penalties.
pub struct XsumTeacher<B: ncm_core::GeneratorBackend> {
    codec: crate::Codec<B>,
    backend: B,
    input: Affine<B::Tensor>,
    query: B::Tensor,
    level_embedding: B::Tensor,
    final_norm: Affine<B::Tensor>,
    blocks: Box<[GeneratorBlock<B::Tensor>]>,
    heads: Box<[TinyHead<B::Tensor>]>,
    geometry: Vec<LevelGeometry<B::Tensor>>,
    token_allowance: Vec<f32>,
    device_allowance: B::Tensor,
    reference_ties: bool,
    request_lock: std::sync::Mutex<()>,
}

/// Raw teacher-forced output; `greedy_ids` do not include XSum penalties.
pub struct XsumTeacherOutput<B: ncm_core::GeneratorBackend> {
    pub codes: B::Indices,
    pub logits: B::Tensor,
    pub greedy_ids: B::Indices,
}

/// XSum generated codes and final IDs after the explicit CPU FP64 penalties.
pub struct XsumGenerated<B: ncm_core::GeneratorBackend> {
    pub codes: B::Indices,
    pub ids: Vec<u32>,
}
struct XsumRun<B: ncm_core::GeneratorBackend> {
    codes: B::Indices,
    logits: B::Tensor,
    greedy_ids: Option<B::Indices>,
}

struct XsumSampling {
    cfg: Option<(usize, f64)>,
    random: Option<SplitMix64>,
}

impl<B: ncm_core::GeneratorBackend + Clone> XsumTeacher<B> {
    pub fn from_artifact(backend: B, artifact: &Artifact) -> Result<Self> {
        let inspection = crate::XsumGeneratorInspection::from_artifact(artifact)?;
        let reference_ties = inspection.rope_tables.is_some();
        let backend = if reference_ties {
            backend.xsum_reference_backend()?
        } else {
            backend
        };
        let codec = crate::Codec::from_artifact(backend.clone(), artifact)?;
        Self::from_checked_parts(backend, artifact, inspection, reference_ties, codec)
    }

    /// Build from an owned artifact, releasing codec host bytes after upload.
    pub fn from_owned_artifact(backend: B, mut artifact: Artifact) -> Result<Self> {
        let inspection = crate::XsumGeneratorInspection::from_artifact(&artifact)?;
        let reference_ties = inspection.rope_tables.is_some();
        let backend = if reference_ties {
            backend.xsum_reference_backend()?
        } else {
            backend
        };
        let codec = crate::Codec::from_artifact(backend.clone(), &artifact)?;
        artifact
            .release_file("codec.safetensors")
            .map_err(|e| Error::Weights(e.to_string()))?;
        Self::from_checked_parts(backend, &artifact, inspection, reference_ties, codec)
    }

    fn from_checked_parts(
        backend: B,
        artifact: &Artifact,
        inspection: crate::XsumGeneratorInspection,
        reference_ties: bool,
        codec: crate::Codec<B>,
    ) -> Result<Self> {
        let bytes = artifact
            .file("generator.safetensors")
            .map_err(|e| Error::Weights(e.to_string()))?;
        let tensors = SafeTensors::deserialize(bytes).map_err(|e| Error::Weights(e.to_string()))?;
        let mut scales = Vec::with_capacity(9);
        for i in 0..9 {
            let tensor = tensors
                .tensor(&format!("lm_heads.{i}.log_inv_T"))
                .map_err(|e| Error::Weights(e.to_string()))?;
            let raw: [u8; 4] = tensor
                .data()
                .try_into()
                .map_err(|_| Error::Weights("invalid head scale".into()))?;
            scales.push(f32::from_le_bytes(raw).exp());
        }
        let mut weights = BTreeMap::new();
        for name in tensors.names() {
            if name == "lvl_1L" || name == "attn_mask_for_masking" || name.ends_with(".log_inv_T") {
                continue;
            }
            let tensor = tensors
                .tensor(name)
                .map_err(|e| Error::Weights(e.to_string()))?;
            let values = upload_values(tensor.data())?;
            weights.insert(
                name.to_owned(),
                backend.upload_f32(&values, tensor.shape())?,
            );
        }
        let input = take_affine(&mut weights, "transformer.wte")?;
        let query = take(&mut weights, "l0_queries")?;
        let level_embedding = take(&mut weights, "transformer.wle.weight")?;
        let final_norm = take_affine(&mut weights, "transformer.ln_f")?;
        let mut blocks = Vec::with_capacity(12);
        for i in 0..12 {
            let p = format!("transformer.blocks.{i}");
            blocks.push(GeneratorBlock {
                ln_1: take_affine(&mut weights, &format!("{p}.ln_1"))?,
                qkv: take_affine(&mut weights, &format!("{p}.attn.c_attn"))?,
                q_norm: take(&mut weights, &format!("{p}.attn.q_norm.weight"))?,
                k_norm: take(&mut weights, &format!("{p}.attn.k_norm.weight"))?,
                attention_projection: take_affine(&mut weights, &format!("{p}.attn.c_proj"))?,
                ln_2: take_affine(&mut weights, &format!("{p}.ln_2"))?,
                mlp_in: take_affine(&mut weights, &format!("{p}.mlp.c_fc"))?,
                mlp_out: take_affine(&mut weights, &format!("{p}.mlp.c_proj"))?,
            });
        }
        let mut heads = Vec::with_capacity(9);
        for (i, inverse_temperature) in scales.into_iter().enumerate() {
            let book = codec.weight(&format!("vq.embeddings.{i}.weight"))?;
            let norm = backend.sum_last(&backend.square(book)?)?;
            let total = backend.sum_last(&backend.reshape(&norm, &[1, 4096])?)?;
            let mean = backend.read_f32(&backend.scale(&total, 1.0 / 4096.0)?)?[0].max(1e-12);
            if !mean.is_finite() {
                return Err(Error::Weights("invalid XSum codebook norm".into()));
            }
            heads.push(TinyHead {
                projection: take(&mut weights, &format!("lm_heads.{i}.proj.weight"))?,
                squared_norm: backend.reshape(&norm, &[1, 1, 4096])?,
                mean_squared_norm: mean,
                inverse_temperature,
            });
        }
        if !weights.is_empty() {
            return Err(Error::Weights("unbound XSum generator tensor".into()));
        }
        let reference_rope = inspection
            .rope_tables
            .as_ref()
            .map(|(cos, sin)| (cos.as_slice(), sin.as_slice()));
        let geometry = prepare_geometry_with_rope(
            &backend,
            codec.config().levels(),
            512,
            768,
            12,
            100000,
            reference_rope,
        )?;
        let device_allowance = backend.upload_f32(&inspection.token_allowance, &[50304])?;
        Ok(Self {
            codec,
            backend,
            input,
            query,
            level_embedding,
            final_norm,
            blocks: blocks.into_boxed_slice(),
            heads: heads.into_boxed_slice(),
            geometry,
            token_allowance: inspection.token_allowance,
            device_allowance,
            reference_ties,
            request_lock: std::sync::Mutex::new(()),
        })
    }
}

impl<B: ncm_core::GeneratorBackend> XsumTeacher<B> {
    pub fn token_allowance(&self) -> &[f32] {
        &self.token_allowance
    }

    /// Run fixed code IDs through all nine levels with explicit prefix padding.
    pub fn run_traced(
        &self,
        prefix_tokens: &[u32],
        forced_codes: &[u32],
        batch: usize,
        trace: &mut impl crate::TraceSink<B>,
    ) -> Result<XsumTeacherOutput<B>> {
        let _request = self
            .request_lock
            .lock()
            .map_err(|_| Error::Backend("generator request lock poisoned".into()))?;
        let result =
            self.run_internal(prefix_tokens, Some(forced_codes), batch, trace, true, None)?;
        Ok(XsumTeacherOutput {
            codes: result.codes,
            logits: result.logits,
            greedy_ids: result
                .greedy_ids
                .ok_or_else(|| Error::Backend("missing XSum greedy trace".into()))?,
        })
    }

    /// Generate with the pinned temperature-1, top-k-1, top-p-0, CFG-0
    /// configuration. The pinned CUDA reference profile uses its documented
    /// first-index tie ordering; the earlier profile rejects exact ties. The
    /// backend must explicitly implement selection; there is no fallback.
    pub fn generate_top1_traced(
        &self,
        prefix_tokens: &[u32],
        batch: usize,
        trace: &mut impl crate::TraceSink<B>,
    ) -> Result<XsumGenerated<B>> {
        let _request = self
            .request_lock
            .lock()
            .map_err(|_| Error::Backend("generator request lock poisoned".into()))?;
        let output = self.run_internal(prefix_tokens, None, batch, trace, true, None)?;
        self.finish_cpu(output, batch)
    }

    /// Untraced top-1 generation with explicitly requested CPU FP64 final selection.
    /// This avoids relying on device FP64 and does not silently change backends.
    pub fn generate_top1_cpu(
        &self,
        prefix_tokens: &[u32],
        batch: usize,
    ) -> Result<XsumGenerated<B>> {
        let _request = self
            .request_lock
            .lock()
            .map_err(|_| Error::Backend("generator request lock poisoned".into()))?;
        let output =
            self.run_internal(prefix_tokens, None, batch, &mut crate::NoTrace, false, None)?;
        self.finish_cpu(output, batch)
    }

    /// Seeded CFG-0 sampling with explicitly requested CPU FP64 final selection.
    /// The request-local random stream is the same as `sample_seeded_device`.
    pub fn sample_seeded_cpu(
        &self,
        prefix_tokens: &[u32],
        batch: usize,
        seed: u64,
    ) -> Result<XsumGenerated<B>> {
        let _request = self
            .request_lock
            .lock()
            .map_err(|_| Error::Backend("generator request lock poisoned".into()))?;
        let output = self.run_internal(
            prefix_tokens,
            None,
            batch,
            &mut crate::NoTrace,
            false,
            Some(XsumSampling {
                cfg: None,
                random: Some(SplitMix64(seed)),
            }),
        )?;
        self.finish_cpu(output, batch)
    }

    fn finish_cpu(&self, output: XsumRun<B>, batch: usize) -> Result<XsumGenerated<B>> {
        // Selection is independent between rows. Limit the explicit FP32 host
        // readback to four rows instead of materializing the full batch.
        let mut ids = Vec::with_capacity(batch * 48);
        for start in (0..batch).step_by(4) {
            let count = (batch - start).min(4);
            let chunk = self.backend.narrow(&output.logits, 0, start, count)?;
            ids.extend(crate::xsum_decode::select_xsum_chunk_parallel(
                &self.backend.read_f32(&chunk)?,
                &self.token_allowance,
                count,
                48,
            )?);
        }
        Ok(XsumGenerated {
            codes: output.codes,
            ids,
        })
    }

    /// Generate with the restricted top-1 settings and explicitly use the
    /// backend's device FP64 penalty implementation. Unsupported backends fail;
    /// this method never silently switches to the CPU selector.
    pub fn generate_top1_device_traced(
        &self,
        prefix_tokens: &[u32],
        batch: usize,
        trace: &mut impl crate::TraceSink<B>,
    ) -> Result<XsumGenerated<B>> {
        let _request = self
            .request_lock
            .lock()
            .map_err(|_| Error::Backend("generator request lock poisoned".into()))?;
        let output = self.run_internal(prefix_tokens, None, batch, trace, true, None)?;
        self.finish_device(output)
    }

    /// Untraced device generation; skips the unused raw-greedy output.
    pub fn generate_top1_device(
        &self,
        prefix_tokens: &[u32],
        batch: usize,
    ) -> Result<XsumGenerated<B>> {
        let _request = self
            .request_lock
            .lock()
            .map_err(|_| Error::Backend("generator request lock poisoned".into()))?;
        let output =
            self.run_internal(prefix_tokens, None, batch, &mut crate::NoTrace, false, None)?;
        self.finish_device(output)
    }

    /// Generate top-1 codes with the pinned upstream classifier-free guidance
    /// schedule. A positive scale uses a doubled document/null batch. Tie
    /// behavior follows the selected profile. Scale zero uses ordinary generation.
    pub fn generate_top1_cfg_device(
        &self,
        prefix_tokens: &[u32],
        batch: usize,
        scale: f64,
    ) -> Result<XsumGenerated<B>> {
        if scale == 0.0 {
            return self.generate_top1_device(prefix_tokens, batch);
        }
        self.generate_guided_top1(prefix_tokens, batch, scale, |output| {
            self.finish_device(output)
        })
    }

    /// Use the same guided top-1 schedule with explicitly requested CPU FP64
    /// final selection. A request at scale zero uses the ordinary CPU-final path.
    pub fn generate_top1_cfg_cpu(
        &self,
        prefix_tokens: &[u32],
        batch: usize,
        scale: f64,
    ) -> Result<XsumGenerated<B>> {
        if scale == 0.0 {
            return self.generate_top1_cpu(prefix_tokens, batch);
        }
        self.generate_guided_top1(prefix_tokens, batch, scale, |output| {
            self.finish_cpu(output, batch)
        })
    }

    fn generate_guided_top1(
        &self,
        prefix_tokens: &[u32],
        batch: usize,
        scale: f64,
        finish: impl FnOnce(XsumRun<B>) -> Result<XsumGenerated<B>>,
    ) -> Result<XsumGenerated<B>> {
        if !scale.is_finite() || scale < 0.0 || !(scale as f32).is_finite() {
            return Err(Error::Input("invalid XSum CFG scale".into()));
        }
        if !(1..=64).contains(&batch) || batch.checked_mul(512) != Some(prefix_tokens.len()) {
            return Err(Error::Input("XSum CFG expects 1..=64 prefixes".into()));
        }
        let mut prefixes = Vec::with_capacity(batch * 1024);
        prefixes.extend_from_slice(prefix_tokens);
        for _ in 0..batch {
            prefixes.push(50256);
            prefixes.extend(std::iter::repeat_n(50257, 511));
        }
        let _request = self
            .request_lock
            .lock()
            .map_err(|_| Error::Backend("generator request lock poisoned".into()))?;
        let output = self.run_internal(
            &prefixes,
            None,
            batch * 2,
            &mut crate::NoTrace,
            false,
            Some(XsumSampling {
                cfg: Some((batch, scale)),
                random: None,
            }),
        )?;
        finish(output)
    }

    /// Experimental sampling at temperature 1, top-k 0 and top-p 1 using request-local
    /// SplitMix64 draws. This does not reproduce PyTorch's random stream.
    /// Positive CFG follows the same upstream document/null schedule as the
    /// top-1 path; other sampling settings are not implemented.
    pub fn sample_seeded_device(
        &self,
        prefix_tokens: &[u32],
        batch: usize,
        scale: f64,
        seed: u64,
    ) -> Result<XsumGenerated<B>> {
        if !scale.is_finite() || scale < 0.0 || !(scale as f32).is_finite() {
            return Err(Error::Input("invalid XSum CFG scale".into()));
        }
        let cfg = if scale > 0.0 {
            if !(1..=64).contains(&batch) || batch.checked_mul(512) != Some(prefix_tokens.len()) {
                return Err(Error::Input("XSum CFG expects 1..=64 prefixes".into()));
            }
            Some((batch, scale))
        } else {
            None
        };
        let mut doubled = Vec::new();
        let (prefixes, internal_batch) = if cfg.is_some() {
            doubled.reserve(batch * 1024);
            doubled.extend_from_slice(prefix_tokens);
            for _ in 0..batch {
                doubled.push(50256);
                doubled.extend(std::iter::repeat_n(50257, 511));
            }
            (doubled.as_slice(), batch * 2)
        } else {
            (prefix_tokens, batch)
        };
        let _request = self
            .request_lock
            .lock()
            .map_err(|_| Error::Backend("generator request lock poisoned".into()))?;
        let output = self.run_internal(
            prefixes,
            None,
            internal_batch,
            &mut crate::NoTrace,
            false,
            Some(XsumSampling {
                cfg,
                random: Some(SplitMix64(seed)),
            }),
        )?;
        self.finish_device(output)
    }

    fn finish_device(&self, output: XsumRun<B>) -> Result<XsumGenerated<B>> {
        let selected = self
            .backend
            .xsum_penalty_fp64(&output.logits, &self.device_allowance)?;
        let ids = self.backend.read_indices(&selected)?;
        if ids.iter().any(|&id| id >= 50304) {
            return Err(Error::Backend("invalid XSum device penalty input".into()));
        }
        Ok(XsumGenerated {
            codes: output.codes,
            ids,
        })
    }

    fn run_internal(
        &self,
        prefix_tokens: &[u32],
        forced_codes: Option<&[u32]>,
        batch: usize,
        trace: &mut impl crate::TraceSink<B>,
        need_greedy: bool,
        mut sampling: Option<XsumSampling>,
    ) -> Result<XsumRun<B>> {
        let cfg = sampling.as_ref().and_then(|options| options.cfg);
        let levels = self.codec.config().levels();
        let total: usize = levels.iter().sum();
        if !(1..=128).contains(&batch)
            || batch.checked_mul(512) != Some(prefix_tokens.len())
            || forced_codes.is_some_and(|codes| batch.checked_mul(total) != Some(codes.len()))
            || prefix_tokens.iter().any(|&v| v >= 50304)
            || prefix_tokens
                .as_chunks::<512>()
                .0
                .iter()
                .any(|row| row.iter().all(|&v| v == 50257))
            || forced_codes.is_some_and(|codes| codes.iter().any(|&v| v >= 4096))
        {
            return Err(Error::Input(
                "invalid XSum prefix, forced codes or batch".into(),
            ));
        }
        let b = &self.backend;
        let prefix_ids = b.upload_indices(prefix_tokens, &[batch, 512])?;
        let raw = b.embedding(self.codec.weight("encoder.wte.weight")?, &prefix_ids)?;
        let prefix = b.project_bias(&raw, &self.input.weight, &self.input.bias)?;
        let forced = forced_codes
            .map(|codes| b.upload_indices(codes, &[batch, total]))
            .transpose()?;
        let has_padding = prefix_tokens.contains(&50257);
        let mut selected_levels = Vec::with_capacity(levels.len());
        let mut cumulative: Option<B::Tensor> = None;
        let mut offset = 0;
        for (level, &length) in levels.iter().enumerate() {
            let latent = if level == 0 {
                let zeros =
                    b.upload_f32(&vec![0.0; batch * length * 384], &[batch, length, 384])?;
                b.add(&self.query, &zeros)?
            } else {
                b.area1d(cumulative.as_ref().expect("preceding levels"), length)?
            };
            if level == 1 && forced.is_some() {
                trace.tensor("diagnostic.level_01.latent", &latent)?;
            }
            let target = b.project_bias(&latent, &self.input.weight, &self.input.bias)?;
            if level == 1 && forced.is_some() {
                trace.tensor("diagnostic.level_01.target_projection", &target)?;
            }
            let target = b.add(&target, &b.narrow(&self.level_embedding, 0, level, 1)?)?;
            let seq = 512 + length;
            let base = &self.geometry[level];
            let mask = if has_padding {
                let mut padding = Vec::with_capacity(batch * seq);
                for row in prefix_tokens.as_chunks::<512>().0 {
                    padding.extend(
                        row.iter()
                            .map(|&v| if v == 50257 { f32::NEG_INFINITY } else { 0.0 }),
                    );
                    padding.extend(std::iter::repeat_n(0.0, length));
                }
                let padding = b.upload_f32(&padding, &[batch, 1, 1, seq])?;
                b.add(&base.mask, &padding)?
            } else {
                base.mask.clone()
            };
            let geometry = LevelGeometry {
                mask,
                cosine: base.cosine.clone(),
                sine: base.sine.clone(),
            };
            let hidden = decode_transformer_observed(
                b,
                &self.blocks,
                &self.final_norm,
                &geometry,
                &prefix,
                &target,
                512,
                12,
                768,
                length,
                false,
                if level == 1 && forced.is_some() {
                    Some(trace)
                } else {
                    None
                },
            )?;
            let head = &self.heads[level];
            if level == 1 && forced.is_some() {
                trace.tensor("diagnostic.level_01.hidden", &hidden)?;
            }
            let z = b.project(&hidden, &head.projection)?;
            if level == 1 && forced.is_some() {
                trace.tensor("diagnostic.level_01.projection", &z)?;
            }
            let book = self
                .codec
                .weight(&format!("vq.embeddings.{level}.weight"))?;
            let score = b.sub(&b.scale(&b.project(&z, book)?, 2.0)?, &head.squared_norm)?;
            let score = b.scale(&score, 1.0 / f64::from(head.mean_squared_norm))?;
            let logits = b.scale(&score, f64::from(head.inverse_temperature))?;
            trace.tensor(&format!("generator.level_{level:02}.logits"), &logits)?;
            let selected = if let Some(codes) = &forced {
                b.narrow_indices(codes, 1, offset, length)?
            } else {
                let guided = if let Some((original_batch, scale)) = cfg {
                    let t = scale * (level as f64 / (levels.len() - 1).max(1) as f64);
                    let conditioned = b.narrow(&logits, 0, 0, original_batch)?;
                    let null = b.narrow(&logits, 0, original_batch, original_batch)?;
                    Some(b.sub(&b.scale(&conditioned, 1.0 + t)?, &b.scale(&null, t)?)?)
                } else {
                    None
                };
                let scores = guided.as_ref().unwrap_or(&logits);
                let selected = if let Some(random) =
                    sampling.as_mut().and_then(|s| s.random.as_mut())
                {
                    let rows = cfg.map_or(batch, |(original_batch, _)| original_batch);
                    let words: Vec<u32> = (0..rows * length)
                        .map(|_| (random.next() >> 32) as u32)
                        .collect();
                    b.xsum_categorical_4096(scores, &b.upload_indices(&words, &[rows, length])?)?
                } else {
                    if self.reference_ties {
                        b.xsum_reference_top1_4096(scores)?
                    } else {
                        b.unique_max_4096(scores)?
                    }
                };
                if cfg.is_some() {
                    b.concat_indices(&[selected.clone(), selected], 0)?
                } else {
                    selected
                }
            };
            offset += length;
            selected_levels.push(selected.clone());
            let embedding = if forced.is_some() {
                b.embedding(book, &selected)?
            } else {
                b.sampled_embedding_4096(book, &selected)?
            };
            let upsampled = if forced.is_some() && level == 0 {
                trace.tensor("diagnostic.level_01.codebook", &embedding)?;
                self.codec.xsum_cascade_diagnostic(&embedding, trace)?
            } else {
                self.codec.upsample(&embedding, level)?
            };
            let contribution = self.codec.refine(&upsampled, level)?;
            let sum = match cumulative {
                Some(ref old) => b.add(old, &contribution)?,
                None => contribution,
            };
            if level == 0 && forced.is_some() {
                trace.tensor("diagnostic.level_01.initial_reconstruction", &sum)?;
            }
            trace.tensor(&format!("reconstruction.level_{level:02}"), &sum)?;
            cumulative = Some(sum);
        }
        let codes = b.concat_indices(&selected_levels, 1)?;
        if forced.is_none() && b.read_indices(&codes)?.iter().any(|&value| value >= 4096) {
            let reason = if sampling.as_ref().is_some_and(|s| s.random.is_some()) {
                "invalid XSum categorical sampling logits"
            } else if self.reference_ties {
                "nonfinite XSum reference head logits are unsupported"
            } else {
                "tied or nonfinite XSum head maximum is unsupported"
            };
            return Err(Error::Backend(reason.into()));
        }
        let latent = cumulative.expect("validated levels");
        let (codes, latent) = if let Some((original_batch, _)) = cfg {
            (
                b.narrow_indices(&codes, 0, 0, original_batch)?,
                b.narrow(&latent, 0, 0, original_batch)?,
            )
        } else {
            (codes, latent)
        };
        let (logits, greedy_ids) = if need_greedy {
            let (logits, ids) = self
                .codec
                .decode_latent_outputs(&latent, &mut crate::NoTrace)?;
            (logits, Some(ids))
        } else {
            (
                self.codec
                    .decode_latent_logits(&latent, &mut crate::NoTrace)?,
                None,
            )
        };
        trace.tensor("generator.final_logits", &logits)?;
        if let Some(ids) = &greedy_ids {
            trace.indices("generator.greedy_ids", ids)?;
        }
        Ok(XsumRun {
            codes,
            logits,
            greedy_ids,
        })
    }
}

// Explicit portable request-local RNG, not a claim of PyTorch seed equivalence.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

/// Validates the exact TinyStories generator and its supplied initial distribution
/// on the host. This is not a generator implementation or a parity qualification.
pub struct TinyStoriesGeneratorInspection {
    pub tensor_count: usize,
    pub initial_distribution: Vec<f32>,
}

impl TinyStoriesGeneratorInspection {
    pub fn from_artifact(artifact: &Artifact) -> Result<Self> {
        let config_bytes = artifact
            .file("config.json")
            .map_err(|e| Error::Weights(e.to_string()))?;
        crate::validate_artifact_profile(artifact, config_bytes)?;
        let codec = artifact_config(config_bytes)?;
        let config: serde_json::Value = serde_json::from_slice(config_bytes)?;
        let reference: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/tinystories-config-phase4.json"
        ))?;
        if config["model"] != reference["model"] || config["task"] != "tinystories" {
            return Err(Error::Config(
                "unsupported TinyStories generator configuration".into(),
            ));
        }
        let mut shapes = GeneratorConfig {
            attn_pattern: "block_diagonal".into(),
            bias: true,
            context_length: 129,
            dropout: 0.1,
            n_embd: 768,
            n_head: 12,
            n_layer: 12,
            prefix_len: 1,
            query_level0: true,
            raw_token_prefix: true,
            rope_base: 100000,
            tie_heads_to_codebook: false,
            use_level_encoding: true,
            use_positional_encoding: true,
            use_qk_norm: true,
            use_rope: true,
            vocab_size: vec![4096; 11],
        }
        .shapes(&codec);
        shapes.remove("l0_queries");
        shapes.insert("transformer.wle.weight".into(), vec![11, 768]);
        for i in 0..11 {
            shapes.remove(&format!("lm_heads.{i}.weight"));
            shapes.remove(&format!("lm_heads.{i}.bias"));
            shapes.insert(format!("lm_heads.{i}.proj.weight"), vec![384, 768]);
            shapes.insert(format!("lm_heads.{i}.log_inv_T"), vec![]);
        }
        let bytes = artifact
            .file("generator.safetensors")
            .map_err(|e| Error::Weights(e.to_string()))?;
        let tensors = SafeTensors::deserialize(bytes).map_err(|e| Error::Weights(e.to_string()))?;
        if tensors.len() != shapes.len() + 2 {
            return Err(Error::Weights(
                "TinyStories generator tensor inventory differs".into(),
            ));
        }
        for (name, shape) in &shapes {
            let t = tensors
                .tensor(name)
                .map_err(|e| Error::Weights(e.to_string()))?;
            if t.dtype() != Dtype::F32 || t.shape() != shape || has_nonfinite_f32_le(t.data()) {
                return Err(Error::Weights(format!(
                    "invalid TinyStories generator tensor {name}"
                )));
            }
        }
        for i in 0..11 {
            let scale = tensors
                .tensor(&format!("lm_heads.{i}.log_inv_T"))
                .map_err(|e| Error::Weights(e.to_string()))?;
            let raw: [u8; 4] = scale
                .data()
                .try_into()
                .map_err(|_| Error::Weights("invalid head scale".into()))?;
            let temperature = f32::from_le_bytes(raw).exp();
            if !temperature.is_finite() || temperature <= 0.0 {
                return Err(Error::Weights("invalid head temperature".into()));
            }
        }
        let levels = &codec.levels()[1..];
        let total: usize = levels.iter().sum();
        let level = tensors
            .tensor("lvl_1L")
            .map_err(|e| Error::Weights(e.to_string()))?;
        let expected = levels
            .iter()
            .enumerate()
            .flat_map(|(i, &len)| std::iter::repeat_n(i as i64, len));
        if level.dtype() != Dtype::I64
            || level.shape() != [1, total]
            || !level
                .data()
                .as_chunks::<8>()
                .0
                .iter()
                .map(|x| i64::from_le_bytes(*x))
                .eq(expected)
        {
            return Err(Error::Weights("incorrect TinyStories level buffer".into()));
        }
        let seq = 1 + total;
        let mask = tensors
            .tensor("attn_mask_for_masking")
            .map_err(|e| Error::Weights(e.to_string()))?;
        if mask.dtype() != Dtype::F32 || mask.shape() != [1, 1, seq, seq] {
            return Err(Error::Weights("invalid TinyStories attention mask".into()));
        }
        let mut positions = vec![usize::MAX];
        for (i, &len) in levels.iter().enumerate() {
            positions.extend(std::iter::repeat_n(i, len));
        }
        for (i, raw) in mask.data().as_chunks::<4>().0.iter().enumerate() {
            let (row, col) = (i / seq, i % seq);
            let allowed = col == 0 || (row > 0 && positions[row] == positions[col]);
            let expected = if allowed {
                0f32.to_bits()
            } else {
                f32::NEG_INFINITY.to_bits()
            };
            if u32::from_le_bytes(*raw) != expected {
                return Err(Error::Weights(
                    "incorrect TinyStories attention mask".into(),
                ));
            }
        }
        let bytes = artifact
            .file("inference.safetensors")
            .map_err(|e| Error::Weights(e.to_string()))?;
        let inference =
            SafeTensors::deserialize(bytes).map_err(|e| Error::Weights(e.to_string()))?;
        if inference.len() != 1 {
            return Err(Error::Weights(
                "invalid TinyStories inference inventory".into(),
            ));
        }
        let unigram = inference
            .tensor("level0_unigram")
            .map_err(|e| Error::Weights(e.to_string()))?;
        if unigram.dtype() != Dtype::F32 || unigram.shape() != [4096] {
            return Err(Error::Weights("invalid TinyStories unigram shape".into()));
        }
        let initial_distribution: Vec<f32> = unigram
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| f32::from_le_bytes(*x))
            .collect();
        let sum: f64 = initial_distribution.iter().map(|&p| f64::from(p)).sum();
        if initial_distribution
            .iter()
            .any(|p| !p.is_finite() || *p < 0.0)
            || (sum - 1.0).abs() > 1e-6
        {
            return Err(Error::Weights(
                "invalid TinyStories unigram distribution".into(),
            ));
        }
        Ok(Self {
            tensor_count: shapes.len(),
            initial_distribution,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{TensorView, serialize};
    use std::{cell::Cell, rc::Rc};

    #[test]
    fn reference_rotary_tables_validate_longer_target_and_finite_values() {
        let cos = vec![1.0; 128 * 32];
        let mut sin = vec![0.0; 128 * 32];
        assert!(validate_reference_rope(&[1, 64, 128], 1, 64, &cos, &sin).is_ok());
        assert!(validate_reference_rope(&[1, 64, 128], 1, 64, &cos[..32], &sin).is_err());
        sin[13] = f32::NAN;
        assert!(validate_reference_rope(&[1, 64, 128], 1, 64, &cos, &sin).is_err());
    }

    #[derive(Clone, Default)]
    struct Counter(Rc<Cell<usize>>);
    impl WeightBackend for Counter {
        type Tensor = Vec<usize>;
        fn upload_f32(&self, values: &[f32], shape: &[usize]) -> Result<Self::Tensor> {
            assert_eq!(values.len(), shape.iter().product::<usize>());
            self.0.set(self.0.get() + 1);
            Ok(shape.to_vec())
        }
    }
    fn tiny() -> (GeneratorConfig, CodecConfig) {
        let f: serde_json::Value =
            serde_json::from_str(include_str!("../../../tests/fixtures/codec-v1.json")).unwrap();
        let codec = CodecConfig::from_json(&serde_json::to_vec(&f["config"]).unwrap()).unwrap();
        let cfg = GeneratorConfig {
            attn_pattern: "block_diagonal".into(),
            bias: true,
            context_length: 10,
            dropout: 0.2,
            n_embd: 8,
            n_head: 2,
            n_layer: 1,
            prefix_len: 3,
            query_level0: true,
            raw_token_prefix: true,
            rope_base: 10000,
            tie_heads_to_codebook: false,
            use_level_encoding: true,
            use_positional_encoding: true,
            use_qk_norm: true,
            use_rope: true,
            vocab_size: codec.raw().vqvae_vocab_size.clone(),
        };
        (cfg, codec)
    }
    fn make_bytes(cfg: &GeneratorConfig, codec: &CodecConfig, corruption: usize) -> Vec<u8> {
        let mut tensors: BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)> = cfg
            .shapes(codec)
            .into_iter()
            .map(|(n, s)| {
                let count = s.iter().product::<usize>();
                (n, (Dtype::F32, s, vec![0; count * 4]))
            })
            .collect();
        let mut level_bytes = Vec::new();
        let mut levels = vec![usize::MAX; cfg.prefix_len];
        for (i, &n) in codec.levels().iter().enumerate() {
            for _ in 0..n {
                level_bytes.extend_from_slice(&(i as i64).to_le_bytes());
                levels.push(i);
            }
        }
        let seq = levels.len();
        let mut mask = Vec::new();
        for row in 0..seq {
            for col in 0..seq {
                let allow =
                    col < cfg.prefix_len || (row >= cfg.prefix_len && levels[row] == levels[col]);
                mask.extend_from_slice(
                    &(if allow { 0.0f32 } else { f32::NEG_INFINITY }).to_le_bytes(),
                );
            }
        }
        let total = seq - cfg.prefix_len;
        tensors.insert("lvl_1L".into(), (Dtype::I64, vec![1, total], level_bytes));
        tensors.insert(
            "attn_mask_for_masking".into(),
            (Dtype::F32, vec![1, 1, seq, seq], mask),
        );
        match corruption {
            1 => {
                tensors.get_mut("lvl_1L").unwrap().2[0] = 9;
            }
            2 => {
                tensors.get_mut("attn_mask_for_masking").unwrap().2
                    [4 * cfg.prefix_len..4 * (cfg.prefix_len + 1)]
                    .copy_from_slice(&0f32.to_le_bytes());
            }
            3 => {
                tensors.get_mut("transformer.wte.weight").unwrap().2[..4]
                    .copy_from_slice(&f32::NAN.to_le_bytes());
            }
            4 => {
                tensors.remove("lm_heads.0.weight");
            }
            _ => {}
        }
        serialize(
            tensors
                .iter()
                .map(|(n, (d, s, b))| (n, TensorView::new(*d, s.clone(), b).unwrap())),
            None,
        )
        .unwrap()
    }
    #[test]
    fn generator_preflight_checks_complete_inventory_and_buffers_before_upload() {
        let (cfg, codec) = tiny();
        let expected = cfg.shapes(&codec).len();
        let counter = Counter::default();
        let loaded = GeneratorWeights::from_safetensors(
            counter.clone(),
            cfg.clone(),
            &codec,
            &make_bytes(&cfg, &codec, 0),
        )
        .unwrap();
        assert_eq!(loaded.tensor_count(), expected);
        assert_eq!(counter.0.get(), expected);
        for mutation in 1..=4 {
            let counter = Counter::default();
            assert!(
                GeneratorWeights::from_safetensors(
                    counter.clone(),
                    cfg.clone(),
                    &codec,
                    &make_bytes(&cfg, &codec, mutation)
                )
                .is_err(),
                "mutation {mutation}"
            );
            assert_eq!(counter.0.get(), 0, "mutation {mutation} allocated");
        }
        let mut unsupported = cfg.clone();
        unsupported.query_level0 = false;
        let counter = Counter::default();
        assert!(
            GeneratorWeights::from_safetensors(
                counter.clone(),
                unsupported,
                &codec,
                &make_bytes(&cfg, &codec, 0)
            )
            .is_err()
        );
        assert_eq!(counter.0.get(), 0);
    }
    #[test]
    #[ignore = "requires the local canonical Sudoku artifact"]
    fn real_artifact_loads_if_explicitly_requested() {
        let path =
            std::env::var_os("NCM_PHASE3_ARTIFACT").expect("NCM_PHASE3_ARTIFACT is required");
        let artifact = Artifact::load(path).unwrap();
        let counter = Counter::default();
        let weights = GeneratorWeights::from_artifact(counter.clone(), &artifact).unwrap();
        assert_eq!(weights.tensor_count(), 142);
        assert_eq!(counter.0.get(), 142);
    }
}

/// Experimental LM1B output with native codes and explicit host-selected IDs.
#[cfg(feature = "lm1b-experimental")]
pub struct Lm1bSample<B: ncm_core::GeneratorBackend> {
    pub codes: B::Indices,
    pub ids: Vec<u32>,
}

/// Experimental LM1B-small teacher and sampler; not a qualified runtime.
#[cfg(feature = "lm1b-experimental")]
pub struct Lm1bSmallTeacher<B: ncm_core::GeneratorBackend> {
    codec: crate::Codec<B>,
    backend: B,
    level_seven_diagnostic: bool,
    input: Affine<B::Tensor>,
    levels: B::Tensor,
    final_norm: Affine<B::Tensor>,
    blocks: Box<[GeneratorBlock<B::Tensor>]>,
    head: B::Tensor,
    head_bias: Box<[B::Tensor]>,
    post_conv: Box<[Affine<B::Tensor>]>,
    geometry: Vec<LevelGeometry<B::Tensor>>,
    allowance: Vec<f32>,
    device_allowance: B::Tensor,
    initial_distribution: Vec<f32>,
}

#[cfg(feature = "lm1b-experimental")]
impl<B: ncm_core::GeneratorBackend + Clone> Lm1bSmallTeacher<B> {
    pub fn from_artifact(backend: B, artifact: &Artifact) -> Result<Self> {
        crate::Lm1bSmallGeneratorInspection::from_preparation_bytes(
            artifact
                .file("config.json")
                .map_err(|e| Error::Weights(e.to_string()))?,
            artifact
                .file("generator.safetensors")
                .map_err(|e| Error::Weights(e.to_string()))?,
            artifact
                .file("inference.safetensors")
                .map_err(|e| Error::Weights(e.to_string()))?,
        )?;
        let codec = crate::Codec::from_lm1b_small_artifact_experimental(backend.clone(), artifact)?;
        let bytes = artifact
            .file("generator.safetensors")
            .map_err(|e| Error::Weights(e.to_string()))?;
        let tensors = SafeTensors::deserialize(bytes).map_err(|e| Error::Weights(e.to_string()))?;
        let mut weights = BTreeMap::new();
        for name in tensors.names() {
            if name == "lvl_1L" || name == "attn_mask_for_masking" {
                continue;
            }
            let t = tensors
                .tensor(name)
                .map_err(|e| Error::Weights(e.to_string()))?;
            weights.insert(
                name.to_owned(),
                backend.upload_f32(&upload_values(t.data())?, t.shape())?,
            );
        }
        let input = take_affine(&mut weights, "transformer.wte")?;
        let levels = take(&mut weights, "transformer.wle.weight")?;
        let final_norm = take_affine(&mut weights, "transformer.ln_f")?;
        let mut blocks = Vec::with_capacity(12);
        for i in 0..12 {
            let p = format!("transformer.blocks.{i}");
            blocks.push(GeneratorBlock {
                ln_1: take_affine(&mut weights, &format!("{p}.ln_1"))?,
                qkv: take_affine(&mut weights, &format!("{p}.attn.c_attn"))?,
                q_norm: take(&mut weights, &format!("{p}.attn.q_norm.weight"))?,
                k_norm: take(&mut weights, &format!("{p}.attn.k_norm.weight"))?,
                attention_projection: take_affine(&mut weights, &format!("{p}.attn.c_proj"))?,
                ln_2: take_affine(&mut weights, &format!("{p}.ln_2"))?,
                mlp_in: take_affine(&mut weights, &format!("{p}.mlp.c_fc"))?,
                mlp_out: take_affine(&mut weights, &format!("{p}.mlp.c_proj"))?,
            });
        }
        let head = take(&mut weights, "lm_heads.head.weight")?;
        let mut head_bias = Vec::with_capacity(12);
        let mut post_conv = Vec::with_capacity(12);
        for i in 0..12 {
            head_bias.push(take(&mut weights, &format!("lm_heads.biases.{i}"))?);
            post_conv.push(take_affine(
                &mut weights,
                &format!("post_upsample_convs.{i}"),
            )?);
        }
        let _unused_query = take(&mut weights, "l0_queries")?;
        if !weights.is_empty() {
            return Err(Error::Weights("unbound LM1B generator tensor".into()));
        }
        let (cosine, sine) = lm1b_reference_order_rope();
        let geometry = prepare_geometry_with_rope(
            &backend,
            codec.config().levels(),
            1,
            768,
            12,
            10000,
            Some((&cosine, &sine)),
        )?;
        let data = artifact
            .file("inference.safetensors")
            .map_err(|e| Error::Weights(e.to_string()))?;
        let tensors = SafeTensors::deserialize(data).map_err(|e| Error::Weights(e.to_string()))?;
        let t = tensors
            .tensor("token_unigram")
            .map_err(|e| Error::Weights(e.to_string()))?;
        let allowance = upload_values(t.data())?.into_owned();
        let t = tensors
            .tensor("level0_unigram")
            .map_err(|e| Error::Weights(e.to_string()))?;
        let initial_distribution = upload_values(t.data())?.into_owned();
        let device_allowance = backend.upload_f32(&allowance, &[30528])?;
        Ok(Self {
            codec,
            backend,
            level_seven_diagnostic: false,
            input,
            levels,
            final_norm,
            blocks: blocks.into_boxed_slice(),
            head,
            head_bias: head_bias.into_boxed_slice(),
            post_conv: post_conv.into_boxed_slice(),
            geometry,
            allowance,
            device_allowance,
            initial_distribution,
        })
    }

    /// Explicit same-input diagnostic using externally supplied reference rotary tables.
    pub fn with_reference_rope_diagnostic(mut self, cosine: &[f32], sine: &[f32]) -> Result<Self> {
        self.geometry = prepare_geometry_with_rope(
            &self.backend,
            self.codec.config().levels(),
            1,
            768,
            12,
            10000,
            Some((cosine, sine)),
        )?;
        Ok(self)
    }

    /// Capture the level-seven transformer internals for an explicit diagnostic.
    pub fn from_artifact_level_seven_diagnostic(backend: B, artifact: &Artifact) -> Result<Self> {
        let mut model = Self::from_artifact(backend, artifact)?;
        model.level_seven_diagnostic = true;
        Ok(model)
    }
}

#[cfg(feature = "lm1b-experimental")]
impl<B: ncm_core::GeneratorBackend> Lm1bSmallTeacher<B> {
    pub fn run_traced(
        &self,
        prefix: &[u32],
        codes: &[u32],
        batch: usize,
        trace: &mut impl crate::TraceSink<B>,
    ) -> Result<()> {
        self.run_impl(prefix, Some(codes), &[], batch, trace, None, false, false)
            .map(|_| ())
    }

    /// Uncached greedy continuation from an explicitly supplied initial code.
    pub fn generate_greedy_traced(
        &self,
        prefix: &[u32],
        initial: &[u32],
        batch: usize,
        trace: &mut impl crate::TraceSink<B>,
    ) -> Result<()> {
        self.run_impl(prefix, None, initial, batch, trace, None, false, false)
            .map(|_| ())
    }

    /// Controlled full-distribution sampling, with a request-local SplitMix64 stream.
    /// This explicit CPU sampler does not reproduce PyTorch's random stream.
    pub fn sample_seeded_traced(
        &self,
        prefix: &[u32],
        batch: usize,
        seed: u64,
        trace: &mut impl crate::TraceSink<B>,
    ) -> Result<Lm1bSample<B>> {
        self.sample_seeded_impl(prefix, batch, seed, trace, false, false)
    }

    /// Explicit experimental CUDA sampling; unsupported backends return an error.
    pub fn sample_seeded_device_traced(
        &self,
        prefix: &[u32],
        batch: usize,
        seed: u64,
        trace: &mut impl crate::TraceSink<B>,
    ) -> Result<Lm1bSample<B>> {
        self.sample_seeded_impl(prefix, batch, seed, trace, true, false)
    }

    /// Explicit device categorical sampling with host final penalties.
    pub fn sample_seeded_device(
        &self,
        prefix: &[u32],
        batch: usize,
        seed: u64,
    ) -> Result<Lm1bSample<B>> {
        self.sample_seeded_device_traced(prefix, batch, seed, &mut crate::NoTrace)
    }

    /// Explicit diagnostic with CUDA sampling and CUDA FP64 final penalties.
    pub fn sample_seeded_fully_device(
        &self,
        prefix: &[u32],
        batch: usize,
        seed: u64,
    ) -> Result<Lm1bSample<B>> {
        self.sample_seeded_impl(prefix, batch, seed, &mut crate::NoTrace, true, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_seeded_impl(
        &self,
        prefix: &[u32],
        batch: usize,
        seed: u64,
        trace: &mut impl crate::TraceSink<B>,
        device: bool,
        device_penalty: bool,
    ) -> Result<Lm1bSample<B>> {
        if !(1..=32).contains(&batch) {
            return Err(Error::Input("invalid LM1B sample batch".into()));
        }
        let mut random = SplitMix64(seed);
        let weights: Vec<f64> = self
            .initial_distribution
            .iter()
            .map(|&x| f64::from(x))
            .collect();
        let initial: Vec<u32> = (0..batch)
            .map(|_| lm1b_choose(&weights, &mut random))
            .collect::<Result<_>>()?;
        self.run_impl(
            prefix,
            None,
            &initial,
            batch,
            trace,
            Some(&mut random),
            device,
            device_penalty,
        )
    }

    /// Explicit CPU sampling and penalty selection, without retaining trace tensors.
    pub fn sample_seeded(&self, prefix: &[u32], batch: usize, seed: u64) -> Result<Lm1bSample<B>> {
        self.sample_seeded_traced(prefix, batch, seed, &mut crate::NoTrace)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_impl(
        &self,
        prefix: &[u32],
        codes: Option<&[u32]>,
        initial: &[u32],
        batch: usize,
        trace: &mut impl crate::TraceSink<B>,
        mut random: Option<&mut SplitMix64>,
        device: bool,
        device_penalty: bool,
    ) -> Result<Lm1bSample<B>> {
        validate_lm1b_request(prefix, codes, initial, batch)?;
        let b = &self.backend;
        let prefix_ids = b.upload_indices(prefix, &[batch, 1])?;
        let codes = codes
            .map(|values| b.upload_indices(values, &[batch, 641]))
            .transpose()?;
        let initial = if let Some(codes) = &codes {
            trace.indices("generator.forced_codes", codes)?;
            b.narrow_indices(codes, 1, 0, 1)?
        } else {
            b.upload_indices(initial, &[batch, 1])?
        };
        trace.indices("generator.prefix_ids", &prefix_ids)?;
        trace.indices("generator.initial_codes", &initial)?;
        let raw = b.embedding(self.codec.weight("encoder.wte.weight")?, &prefix_ids)?;
        let prefix = b.project_bias(&raw, &self.input.weight, &self.input.bias)?;
        let mut cumulative = None;
        let mut offset = 0;
        let mut selected_levels = Vec::with_capacity(12);
        for (level, &length) in self.codec.config().levels().iter().enumerate() {
            let mut generated = None;
            if level > 0 {
                let latent = b.area1d(cumulative.as_ref().expect("previous level"), length)?;
                trace.tensor(
                    &format!("generator.level_{level:02}.post_conv_input"),
                    &b.permute(&latent, &[0, 2, 1])?,
                )?;
                let conv = &self.post_conv[level];
                let delta = b.conv1d(&latent, &conv.weight, &conv.bias, 1, 1, false)?;
                trace.tensor(
                    &format!("generator.level_{level:02}.post_conv_output"),
                    &b.permute(&delta, &[0, 2, 1])?,
                )?;
                let target = b.project_bias(
                    &b.add(&latent, &delta)?,
                    &self.input.weight,
                    &self.input.bias,
                )?;
                let target = b.add(&target, &b.narrow(&self.levels, 0, level, 1)?)?;
                let hidden = if self.level_seven_diagnostic && level == 7 {
                    trace.tensor("diagnostic.level_07.target", &target)?;
                    let mut observed = Lm1bLevelSevenTrace(&mut *trace);
                    decode_transformer_observed(
                        b,
                        &self.blocks,
                        &self.final_norm,
                        &self.geometry[level],
                        &prefix,
                        &target,
                        1,
                        12,
                        768,
                        length,
                        true,
                        Some(&mut observed),
                    )?
                } else {
                    decode_transformer(
                        b,
                        &self.blocks,
                        &self.final_norm,
                        &self.geometry[level],
                        &prefix,
                        &target,
                        1,
                        12,
                        768,
                        length,
                    )?
                };
                if self.level_seven_diagnostic && level == 7 {
                    trace.tensor("diagnostic.level_07.hidden", &hidden)?;
                }
                let logits = b.add(&b.project(&hidden, &self.head)?, &self.head_bias[level])?;
                trace.tensor(&format!("generator.level_{level:02}.logits"), &logits)?;
                if codes.is_none() {
                    generated = if device {
                        let rng = random
                            .as_deref_mut()
                            .ok_or_else(|| Error::Input("device sampling requires RNG".into()))?;
                        let words: Vec<u32> = (0..batch * length)
                            .map(|_| (rng.next() >> 32) as u32)
                            .collect();
                        let selected = b.categorical_8192(
                            &logits,
                            &b.upload_indices(&words, &[batch, length])?,
                        )?;
                        if b.read_indices(&selected)?
                            .iter()
                            .any(|&value| value >= 8192)
                        {
                            return Err(Error::Backend(
                                "invalid LM1B device categorical logits".into(),
                            ));
                        }
                        Some(selected)
                    } else {
                        let values = b.read_f32(&logits)?;
                        if values.len() != batch * length * 8192
                            || values.iter().any(|x| !x.is_finite())
                        {
                            return Err(Error::Backend("invalid or nonfinite LM1B logits".into()));
                        }
                        if let Some(rng) = random.as_deref_mut() {
                            let selected: Vec<u32> = values
                                .as_chunks::<8192>()
                                .0
                                .iter()
                                .map(|row| lm1b_sample_logits(row, rng))
                                .collect::<Result<_>>()?;
                            Some(b.upload_indices(&selected, &[batch, length])?)
                        } else {
                            Some(b.select_last(&logits, true)?)
                        }
                    };
                }
            }
            let selected = if let Some(codes) = &codes {
                b.narrow_indices(codes, 1, offset, length)?
            } else if level == 0 {
                initial.clone()
            } else {
                generated.expect("generated level")
            };
            selected_levels.push(selected.clone());
            offset += length;
            let embedding = b.embedding(
                self.codec
                    .weight(&format!("vq.embeddings.{level}.weight"))?,
                &selected,
            )?;
            let up = self.codec.upsample(&embedding, level)?;
            let contribution = self.codec.refine(&up, level)?;
            let sum = if let Some(old) = cumulative {
                b.add(&old, &contribution)?
            } else {
                contribution
            };
            trace.tensor(&format!("reconstruction.level_{level:02}"), &sum)?;
            cumulative = Some(sum);
        }
        let result_codes = b.concat_indices(&selected_levels, 1)?;
        if codes.is_none() {
            trace.indices("generator.generated_codes", &result_codes)?;
        }
        let (logits, greedy) = self
            .codec
            .decode_latent_outputs(&cumulative.expect("levels"), &mut crate::NoTrace)?;
        trace.tensor("generator.final_logits", &logits)?;
        trace.indices("generator.greedy_ids", &greedy)?;
        let penalized = if device_penalty {
            let selected = b.lm1b_penalty_fp64(&logits, &self.device_allowance)?;
            let ids = b.read_indices(&selected)?;
            if ids.len() != batch * 128 || ids.iter().any(|&x| x >= 30528) {
                return Err(Error::Backend("invalid LM1B device penalty output".into()));
            }
            ids
        } else {
            lm1b_penalized_parallel(&b.read_f32(&logits)?, &self.allowance, batch)?
        };
        trace.indices(
            "generator.penalized_ids",
            &b.upload_indices(&penalized, &[batch, 128])?,
        )?;
        Ok(Lm1bSample {
            codes: result_codes,
            ids: penalized,
        })
    }
}

#[cfg(feature = "lm1b-experimental")]
fn lm1b_reference_order_rope() -> (Vec<f32>, Vec<f32>) {
    let inverse: Vec<f32> = (0..32)
        .map(|i| 1.0_f32 / 10000.0_f32.powf(i as f32 / 32.0))
        .collect();
    let mut cosine = Vec::with_capacity(128 * 32);
    let mut sine = Vec::with_capacity(128 * 32);
    for pos in 0..128 {
        for &frequency in &inverse {
            let angle = pos as f32 * frequency;
            cosine.push(angle.cos());
            sine.push(angle.sin());
        }
    }
    (cosine, sine)
}

#[cfg(feature = "lm1b-experimental")]
fn lm1b_choose(weights: &[f64], random: &mut SplitMix64) -> Result<u32> {
    if weights.is_empty()
        || weights.len() > u32::MAX as usize
        || weights.iter().any(|x| !x.is_finite() || *x < 0.0)
    {
        return Err(Error::Input("invalid LM1B categorical weights".into()));
    }
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(Error::Input("invalid LM1B categorical total".into()));
    }
    let uniform = ((random.next() >> 32) as f64 + 0.5) / 4294967296.0;
    let threshold = uniform * total;
    let mut cumulative = 0.0;
    for (index, &weight) in weights.iter().enumerate() {
        cumulative += weight;
        if cumulative > threshold {
            return Ok(index as u32);
        }
    }
    Err(Error::Backend("LM1B categorical sum mismatch".into()))
}

#[cfg(feature = "lm1b-experimental")]
fn lm1b_sample_logits(values: &[f32], random: &mut SplitMix64) -> Result<u32> {
    if values.len() != 8192 || values.iter().any(|v| !v.is_finite()) {
        return Err(Error::Input("invalid LM1B categorical logits".into()));
    }
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f64> = values
        .iter()
        .map(|&x| (f64::from(x) - f64::from(maximum)).exp())
        .collect();
    lm1b_choose(&weights, random)
}

#[cfg(feature = "lm1b-experimental")]
fn validate_lm1b_request(
    prefix: &[u32],
    codes: Option<&[u32]>,
    initial: &[u32],
    batch: usize,
) -> Result<()> {
    if !(1..=32).contains(&batch)
        || prefix.len() != batch
        || prefix.iter().any(|&x| x != 101)
        || codes
            .is_some_and(|values| values.len() != batch * 641 || values.iter().any(|&x| x >= 8192))
        || (codes.is_none() && (initial.len() != batch || initial.iter().any(|&x| x >= 8192)))
    {
        return Err(Error::Input("invalid LM1B generator inputs".into()));
    }
    Ok(())
}

#[cfg(feature = "lm1b-experimental")]
fn lm1b_penalized_parallel(logits: &[f32], allowance: &[f32], batch: usize) -> Result<Vec<u32>> {
    let row_len = 128 * 30528;
    if !(1..=32).contains(&batch) || logits.len() != batch * row_len || allowance.len() != 30528 {
        return Err(Error::Input("invalid LM1B penalty dimensions".into()));
    }
    let mut result = Vec::with_capacity(batch * 128);
    for chunk in logits.chunks(4 * row_len) {
        let rows = std::thread::scope(|scope| {
            let handles: Vec<_> = chunk
                .chunks_exact(row_len)
                .map(|row| scope.spawn(move || lm1b_penalized_ids(row, allowance, 1, 128)))
                .collect();
            let mut output = Vec::with_capacity(handles.len() * 128);
            for handle in handles {
                output.extend(
                    handle
                        .join()
                        .map_err(|_| Error::Backend("LM1B penalty worker failed".into()))??,
                );
            }
            Ok::<_, Error>(output)
        })?;
        result.extend(rows);
    }
    Ok(result)
}

#[cfg(feature = "lm1b-experimental")]
fn lm1b_penalized_ids(
    logits: &[f32],
    allowance: &[f32],
    batch: usize,
    time: usize,
) -> Result<Vec<u32>> {
    let vocab = allowance.len();
    if vocab != 30528
        || !(1..=32).contains(&batch)
        || time != 128
        || logits.len() != batch * time * vocab
        || allowance.iter().any(|x| !x.is_finite() || *x < 0.0)
        || logits.iter().any(|x| !x.is_finite())
    {
        return Err(Error::Input("invalid LM1B penalty input".into()));
    }
    let allow: Vec<f64> = allowance
        .iter()
        .map(|&x| f64::from(x) * time as f64)
        .collect();
    let mut output = Vec::with_capacity(batch * time);
    for row in logits.chunks_exact(time * vocab) {
        let mut counts = vec![0_u32; vocab];
        let start = output.len();
        for (t, values) in row.chunks_exact(vocab).enumerate() {
            let mut best = f64::NEG_INFINITY;
            let mut chosen = 0;
            for (id, &value) in values.iter().enumerate() {
                let count = f64::from(counts[id]);
                let mut score = f64::from(value) - 11.5 * (count - allow[id]).max(0.0);
                score += 2.0 * ((time - t) as f64 / time as f64) * (allow[id] - count).max(0.0);
                for (j, &penalty) in [10.0, 5.0, 2.5, 1.25].iter().enumerate().take(t.min(4)) {
                    if output[start + t - j - 1] as usize == id {
                        score -= penalty;
                    }
                }
                if score > best {
                    best = score;
                    chosen = id;
                }
            }
            counts[chosen] += 1;
            output.push(chosen as u32);
        }
    }
    Ok(output)
}

#[cfg(all(test, feature = "lm1b-experimental"))]
mod lm1b_teacher_tests {
    use super::{
        SplitMix64, lm1b_choose, lm1b_penalized_ids, lm1b_penalized_parallel,
        lm1b_reference_order_rope, lm1b_sample_logits, validate_lm1b_request,
    };

    #[test]
    fn reference_order_rope_matches_pinned_values() {
        let (cosine, sine) = lm1b_reference_order_rope();
        assert_eq!(cosine.len(), 128 * 32);
        assert_eq!(sine.len(), cosine.len());
        assert_eq!(cosine[0], 1.0);
        assert_eq!(sine[0], 0.0);
        for (position, column, c, s) in [
            (127, 1, 0.549_705_2_f32, 0.835_358_74_f32),
            (127, 5, 0.267_998_87_f32, -0.963_419_2_f32),
            (127, 31, 0.999_856_6_f32, 0.016_934_913_f32),
        ] {
            assert!((cosine[position * 32 + column] - c).abs() <= 1e-7);
            assert!((sine[position * 32 + column] - s).abs() <= 1e-7);
        }
    }

    #[test]
    fn controlled_sampler_repeats_and_rejects_invalid_distributions() {
        let mut rng = SplitMix64(11);
        let result: Vec<_> = (0..12)
            .map(|_| lm1b_choose(&[1.0, 2.0, 3.0], &mut rng).unwrap())
            .collect();
        assert_eq!(result, [1, 1, 2, 2, 0, 2, 0, 2, 1, 2, 0, 1]);
        assert!(lm1b_choose(&[0.0, 0.0], &mut rng).is_err());
        assert!(lm1b_choose(&[-1.0, 2.0], &mut rng).is_err());
        assert!(lm1b_sample_logits(&[0.0], &mut rng).is_err());
        assert!(lm1b_sample_logits(&vec![f32::NAN; 8192], &mut rng).is_err());
    }

    #[test]
    fn validates_greedy_and_forced_inputs_before_execution() {
        assert!(validate_lm1b_request(&[101], None, &[8191], 1).is_ok());
        assert!(validate_lm1b_request(&[101], None, &[8192], 1).is_err());
        assert!(validate_lm1b_request(&[0], None, &[0], 1).is_err());
        assert!(validate_lm1b_request(&[101], None, &[], 1).is_err());
        assert!(validate_lm1b_request(&[101], Some(&vec![0; 641]), &[], 1).is_ok());
        assert!(validate_lm1b_request(&[101], Some(&[0]), &[], 1).is_err());
        assert!(validate_lm1b_request(&[], None, &[], 0).is_err());
    }

    #[test]
    fn penalty_selects_first_available_id_and_rejects_invalid_inputs() {
        let allowance = vec![0.0; 30528];
        let mut logits = vec![0.0; 128 * 30528];
        let ids = lm1b_penalized_ids(&logits, &allowance, 1, 128).unwrap();
        assert_eq!(ids, (0..128).collect::<Vec<_>>());
        logits[0] = f32::NAN;
        assert!(lm1b_penalized_ids(&logits, &allowance, 1, 128).is_err());
        assert!(lm1b_penalized_ids(&logits[..1], &allowance, 1, 128).is_err());
    }

    #[test]
    fn parallel_penalties_preserve_order_and_reject_bad_input() {
        let allowance = vec![0.0; 30528];
        let logits = vec![0.0; 5 * 128 * 30528];
        let expected = lm1b_penalized_ids(&logits, &allowance, 5, 128).unwrap();
        assert_eq!(
            lm1b_penalized_parallel(&logits, &allowance, 5).unwrap(),
            expected
        );
        assert!(lm1b_penalized_parallel(&logits[..1], &allowance, 5).is_err());
    }
}
