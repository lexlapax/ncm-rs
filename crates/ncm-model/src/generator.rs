//! Checked weights and eager greedy generation for the Sudoku profile.
use crate::{artifact_config, has_nonfinite_f32_le};
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

    fn shapes(&self, codec: &CodecConfig) -> BTreeMap<String, Vec<usize>> {
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
        let b = &model.backend;
        let cfg = &model.config;
        let head = cfg.n_embd / cfg.n_head;
        let mut result = Vec::with_capacity(levels.len());
        for &length in levels {
            let seq = cfg.prefix_len + length;
            let mut mask = vec![0.0f32; seq * seq];
            for row in 0..cfg.prefix_len {
                for col in cfg.prefix_len..seq {
                    mask[row * seq + col] = f32::NEG_INFINITY;
                }
            }
            let mask = b.upload_f32(&mask, &[1, 1, seq, seq])?;
            let mut cosine = Vec::with_capacity(seq * head / 2);
            let mut sine = Vec::with_capacity(seq * head / 2);
            for pos in (0..cfg.prefix_len).chain(0..length) {
                for i in 0..head / 2 {
                    let freq =
                        (pos as f32) / (cfg.rope_base as f32).powf(i as f32 / (head / 2) as f32);
                    cosine.push(freq.cos());
                    sine.push(freq.sin());
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
    fn linear(&self, affine: &Affine<B::Tensor>, x: &B::Tensor) -> Result<B::Tensor> {
        self.model
            .backend
            .project_bias(x, &affine.weight, &affine.bias)
    }
    fn norm(&self, affine: &Affine<B::Tensor>, x: &B::Tensor) -> Result<B::Tensor> {
        self.model
            .backend
            .layer_norm(x, &affine.weight, Some(&affine.bias), 1e-6)
    }
    fn decode_level(
        &self,
        prefix: &B::Tensor,
        target: &B::Tensor,
        length: usize,
        level: usize,
    ) -> Result<B::Tensor> {
        let b = &self.model.backend;
        let cfg = &self.model.config;
        let batch = b.shape(prefix)[0];
        let seq = cfg.prefix_len + length;
        let head = cfg.n_embd / cfg.n_head;
        let geometry = self
            .geometry
            .get(level)
            .ok_or_else(|| Error::Input("invalid generator level".into()))?;
        let mask = &geometry.mask;
        let cos = &geometry.cosine;
        let sin = &geometry.sine;
        let mut x = b.concat_f32(&[prefix.clone(), target.clone()], 1)?;
        for block in &self.model.parameters.blocks {
            let norm = self.norm(&block.ln_1, &x)?;
            let qkv = self.linear(&block.qkv, &norm)?;
            // Materialize Q/K/V together in the head-major order required by
            // normalization and attention, instead of three separate copies.
            let packed = b.pack_qkv(&qkv, batch, seq, cfg.n_head, head)?;
            let make = |i| -> Result<B::Tensor> {
                b.reshape(
                    &b.narrow(&packed, 0, i, 1)?,
                    &[batch, cfg.n_head, seq, head],
                )
            };
            let q = b.rms_norm(&make(0)?, &block.q_norm, 1e-6)?;
            let k = b.rms_norm(&make(1)?, &block.k_norm, 1e-6)?;
            let v = make(2)?;
            let q = b.rope(&q, cos, sin)?;
            let k = b.rope(&k, cos, sin)?;
            let attn = b.attention(&q, &k, &v, Some(mask))?;
            let attn = b.merge_heads(&attn, batch, seq, cfg.n_head, head)?;
            x = b.add(&x, &self.linear(&block.attention_projection, &attn)?)?;
            let norm = self.norm(&block.ln_2, &x)?;
            let hidden = b.gelu(&self.linear(&block.mlp_in, &norm)?)?;
            x = b.add(&x, &self.linear(&block.mlp_out, &hidden)?)?;
        }
        let x = self.norm(&self.model.parameters.final_norm, &x)?;
        b.narrow(&x, 1, cfg.prefix_len, length)
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

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{TensorView, serialize};
    use std::{cell::Cell, rc::Rc};

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
