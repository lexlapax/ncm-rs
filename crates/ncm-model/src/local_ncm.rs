//! Generic config-driven local NCM model loader and generator (`local-ncm-v1`).
//!
//! Shares the core codec, transformer block, and prefix K/V caching primitives with the
//! pinned upstream profiles while allowing user-exported local NCM checkpoints to specify
//! their own dimensions, level schedules, head layout, optional padding/EOS IDs, and
//! optional HuggingFace tokenizer without hardcoding per-checkpoint Rust structs.
use crate::{
    Codec, Generated, NoTrace, TraceSink,
    generator::{
        Affine, GeneratorBlock, LevelGeometry, decode_target_with_prefix_cache, decode_transformer,
        prefill_prefix_cache, prepare_geometry, take, take_affine,
    },
    has_nonfinite_f32_le, upload_values, xsum_generator,
};
use ncm_core::{
    CodecConfig, Error, GeneratorBackend, LocalGeneratorConfig, LocalHeadKind, LocalNcmConfig,
    Result,
};
use ncm_io::Artifact;
use safetensors::{Dtype, SafeTensors};
use std::collections::BTreeMap;

const LOCAL_CODEC_LIMIT: usize = 2 * 1024 * 1024 * 1024 + 1024 * 1024;
const LOCAL_GENERATOR_LIMIT: u64 = 4 * 1024 * 1024 * 1024;

enum LocalHeads<T> {
    UntiedAffine(Box<[Affine<T>]>),
    UntiedLinear(Box<[T]>),
    SharedHeadBiases { weight: T, biases: Box<[T]> },
}

/// Config-driven NCM codec + generator for `local-ncm-v1` artifacts.
pub struct LocalNcmModel<B: GeneratorBackend> {
    codec: Codec<B>,
    backend: B,
    config: LocalGeneratorConfig,
    input: Affine<B::Tensor>,
    query: B::Tensor,
    level_embedding: B::Tensor,
    final_norm: Affine<B::Tensor>,
    blocks: Box<[GeneratorBlock<B::Tensor>]>,
    heads: LocalHeads<B::Tensor>,
    geometry: Vec<LevelGeometry<B::Tensor>>,
    prefix_cache: bool,
    lora: Option<crate::RuntimeLoraAdapter<B::Tensor>>,
    request_lock: std::sync::Mutex<()>,
}

impl<B: GeneratorBackend + Clone> LocalNcmModel<B> {
    /// Validate and load a `local-ncm-v1` artifact onto `backend`.
    /// Performs complete host preflight on both codec and generator bundles
    /// before the first device allocation.
    pub fn from_artifact(backend: B, artifact: &Artifact) -> Result<Self> {
        if artifact.manifest().profile != "local-ncm-v1" {
            return Err(Error::Config(format!(
                "LocalNcmModel requires profile local-ncm-v1, got {}",
                artifact.manifest().profile
            )));
        }
        let config_bytes = artifact
            .file("config.json")
            .map_err(|e| Error::Weights(e.to_string()))?;
        let local_cfg = LocalNcmConfig::from_json(config_bytes)?;
        let codec_cfg = local_cfg.codec().clone();
        let gen_cfg = local_cfg.model().clone();

        // Preflight generator inventory before any backend allocations.
        let expected_gen = gen_cfg.weight_shapes(&codec_cfg);
        preflight_generator(artifact, &codec_cfg, &gen_cfg, &expected_gen)?;

        // Load codec (which runs its own complete host preflight before uploading).
        let codec = Codec::from_artifact_with_limit(
            backend.clone(),
            codec_cfg.clone(),
            artifact,
            LOCAL_CODEC_LIMIT,
        )?;

        // Upload generator tensors (streaming or in-memory).
        let mut weights = BTreeMap::new();
        if artifact.is_streaming_bundle("generator.safetensors") {
            crate::stream_bundle_weights(
                &backend,
                artifact,
                "generator.safetensors",
                |_, _| Ok(false),
                |name, tensor| {
                    weights.insert(name.to_owned(), tensor);
                    Ok(())
                },
            )?;
        } else {
            let bytes = artifact
                .file("generator.safetensors")
                .map_err(|e| Error::Weights(e.to_string()))?;
            let tensors =
                SafeTensors::deserialize(bytes).map_err(|e| Error::Weights(e.to_string()))?;
            for (name, shape) in &expected_gen {
                let t = tensors
                    .tensor(name)
                    .map_err(|e| Error::Weights(e.to_string()))?;
                let values = upload_values(t.data())?;
                weights.insert(
                    name.clone(),
                    crate::upload_named_weight(&backend, name, &values, shape)?,
                );
            }
        }

        let input = take_affine(&mut weights, "transformer.wte")?;
        let query = take(&mut weights, "l0_queries")?;
        let level_embedding = take(&mut weights, "transformer.wle.weight")?;
        let final_norm = take_affine(&mut weights, "transformer.ln_f")?;
        let mut blocks = Vec::with_capacity(gen_cfg.n_layer);
        for i in 0..gen_cfg.n_layer {
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
        let heads = match gen_cfg.head_kind {
            LocalHeadKind::UntiedAffine => {
                let mut list = Vec::with_capacity(gen_cfg.vocab_size.len());
                for i in 0..gen_cfg.vocab_size.len() {
                    list.push(take_affine(&mut weights, &format!("lm_heads.{i}"))?);
                }
                LocalHeads::UntiedAffine(list.into_boxed_slice())
            }
            LocalHeadKind::UntiedLinear => {
                let mut list = Vec::with_capacity(gen_cfg.vocab_size.len());
                for i in 0..gen_cfg.vocab_size.len() {
                    list.push(take(&mut weights, &format!("lm_heads.{i}.weight"))?);
                }
                LocalHeads::UntiedLinear(list.into_boxed_slice())
            }
            LocalHeadKind::SharedHeadBiases => {
                let weight = take(&mut weights, "lm_head.weight")?;
                let mut biases = Vec::with_capacity(gen_cfg.vocab_size.len());
                for i in 0..gen_cfg.vocab_size.len() {
                    biases.push(take(&mut weights, &format!("head_biases.{i}"))?);
                }
                LocalHeads::SharedHeadBiases {
                    weight,
                    biases: biases.into_boxed_slice(),
                }
            }
        };
        if !weights.is_empty() {
            return Err(Error::Weights("unbound local generator tensor".into()));
        }
        let geometry = prepare_geometry(
            &backend,
            codec_cfg.levels(),
            gen_cfg.prefix_len,
            gen_cfg.n_embd,
            gen_cfg.n_head,
            gen_cfg.rope_base,
        )?;
        let prefix_cache = gen_cfg.prefix_len > 1;
        Ok(Self {
            codec,
            backend,
            config: gen_cfg,
            input,
            query,
            level_embedding,
            final_norm,
            blocks: blocks.into_boxed_slice(),
            heads,
            geometry,
            prefix_cache,
            lora: None,
            request_lock: std::sync::Mutex::new(()),
        })
    }
}

impl<B: GeneratorBackend> LocalNcmModel<B> {
    /// Enable or disable per-layer prefix K/V caching across levels.
    /// Defaults to `true` when `prefix_len > 1`, `false` when `prefix_len == 1`.
    pub fn with_prefix_cache(mut self, enabled: bool) -> Self {
        self.prefix_cache = enabled;
        self
    }

    /// Expected `(n_layer, n_embd)` dimensions for compatible LoRA adapters.
    pub fn lora_dimensions(&self) -> (usize, usize) {
        (self.config.n_layer, self.config.n_embd)
    }

    /// Attach or detach a runtime LoRA adapter overlay (instant refcounted swap).
    pub fn set_lora_adapter(
        &mut self,
        adapter: Option<crate::RuntimeLoraAdapter<B::Tensor>>,
    ) -> Result<()> {
        if let Some(ref a) = adapter
            && (a.n_layer() != self.config.n_layer || a.n_embd() != self.config.n_embd)
        {
            return Err(Error::Config(format!(
                "LoRA adapter ({}, {}) does not match local model ({}, {})",
                a.n_layer(),
                a.n_embd(),
                self.config.n_layer,
                self.config.n_embd
            )));
        }
        self.lora = adapter;
        Ok(())
    }

    pub fn config(&self) -> &LocalGeneratorConfig {
        &self.config
    }

    pub fn codec_config(&self) -> &CodecConfig {
        self.codec.config()
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Generate codes and decoded tokens for `batch` prefixes of length `config.prefix_len`.
    /// `seed = None` runs greedy argmax; `seed = Some(s)` runs deterministic SplitMix64
    /// inverse-CDF sampling scaled by `1.0 / config.temperature`.
    pub fn generate(
        &self,
        prefix_tokens: &[u32],
        batch: usize,
        seed: Option<u64>,
    ) -> Result<Generated<B>> {
        self.generate_traced(prefix_tokens, batch, seed, &mut NoTrace)
    }

    pub fn generate_traced(
        &self,
        prefix_tokens: &[u32],
        batch: usize,
        seed: Option<u64>,
        trace: &mut impl TraceSink<B>,
    ) -> Result<Generated<B>> {
        let _request = self
            .request_lock
            .lock()
            .map_err(|_| Error::Backend("local generator request lock poisoned".into()))?;
        let cfg = &self.config;
        let codec_cfg = self.codec.config();
        if !(1..=128).contains(&batch)
            || batch.checked_mul(cfg.prefix_len) != Some(prefix_tokens.len())
            || prefix_tokens
                .iter()
                .any(|&t| t as usize >= codec_cfg.raw().in_vocab_size)
        {
            return Err(Error::Input(
                "invalid local generator prefix or batch".into(),
            ));
        }
        if let Some(pad) = cfg.pad_token_id {
            for row in prefix_tokens.chunks_exact(cfg.prefix_len) {
                if row.iter().all(|&t| t == pad) {
                    return Err(Error::Input("local generator prefix row is all pad".into()));
                }
            }
        }

        let b = &self.backend;
        let ids = b.upload_indices(prefix_tokens, &[batch, cfg.prefix_len])?;
        let raw = b.embedding(self.codec.weight("encoder.wte.weight")?, &ids)?;
        let prefix = b.project_bias(&raw, &self.input.weight, &self.input.bias)?;
        trace.stage_tensor("prefix_done", &prefix)?;

        let lora_blocks = self.lora.as_ref().map(|a| a.blocks());
        let prefix_cache = if self.prefix_cache {
            let prefix_cos = b.narrow(&self.geometry[0].cosine, 0, 0, cfg.prefix_len)?;
            let prefix_sin = b.narrow(&self.geometry[0].sine, 0, 0, cfg.prefix_len)?;
            let prefix_mask = if let Some(pad) = cfg.pad_token_id {
                Some(upload_prefix_pad_mask(
                    b,
                    prefix_tokens,
                    batch,
                    cfg.prefix_len,
                    pad,
                )?)
            } else {
                None
            };
            Some(prefill_prefix_cache(
                b,
                &self.blocks,
                lora_blocks,
                &prefix,
                &prefix_cos,
                &prefix_sin,
                prefix_mask.as_ref(),
                cfg.prefix_len,
                cfg.n_head,
                cfg.n_embd,
            )?)
        } else {
            None
        };

        let mut rng_state = seed.map(splitmix64);
        let mut cumulative: Option<B::Tensor> = None;
        let mut codes = Vec::with_capacity(codec_cfg.levels().len());
        let mut previous: Option<B::Indices> = None;

        for (level, &length) in codec_cfg.levels().iter().enumerate() {
            let latent = if level == 0 {
                let zeros = b.upload_f32(
                    &vec![0.0; batch * length * codec_cfg.raw().embed_dim],
                    &[batch, length, codec_cfg.raw().embed_dim],
                )?;
                b.add(&self.query, &zeros)?
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
            let target = b.project_bias(&latent, &self.input.weight, &self.input.bias)?;
            let embedding = b.narrow(&self.level_embedding, 0, level, 1)?;
            let target = b.add(&target, &embedding)?;
            let geom = &self.geometry[level];
            let hidden = if let Some(ref cache) = prefix_cache {
                let target_cos = b.narrow(&geom.cosine, 0, cfg.prefix_len, length)?;
                let target_sin = b.narrow(&geom.sine, 0, cfg.prefix_len, length)?;
                let target_mask = if let Some(pad) = cfg.pad_token_id {
                    Some(upload_target_pad_mask(
                        b,
                        prefix_tokens,
                        batch,
                        cfg.prefix_len,
                        length,
                        pad,
                    )?)
                } else {
                    None
                };
                decode_target_with_prefix_cache(
                    b,
                    &self.blocks,
                    lora_blocks,
                    &self.final_norm,
                    &target,
                    cache,
                    &target_cos,
                    &target_sin,
                    target_mask.as_ref(),
                    cfg.n_head,
                    cfg.n_embd,
                    length,
                )?
            } else if let Some(pad) = cfg.pad_token_id {
                let padded_mask =
                    upload_full_pad_mask(b, prefix_tokens, batch, cfg.prefix_len, length, pad)?;
                let padded_geom = LevelGeometry {
                    mask: padded_mask,
                    cosine: geom.cosine.clone(),
                    sine: geom.sine.clone(),
                };
                decode_transformer(
                    b,
                    &self.blocks,
                    lora_blocks,
                    &self.final_norm,
                    &padded_geom,
                    &prefix,
                    &target,
                    cfg.prefix_len,
                    cfg.n_head,
                    cfg.n_embd,
                    length,
                )?
            } else {
                decode_transformer(
                    b,
                    &self.blocks,
                    lora_blocks,
                    &self.final_norm,
                    geom,
                    &prefix,
                    &target,
                    cfg.prefix_len,
                    cfg.n_head,
                    cfg.n_embd,
                    length,
                )?
            };
            let logits = self.project_head(level, &hidden)?;
            trace.tensor(&format!("generator.level_{level:02}.logits"), &logits)?;
            trace.stage_tensor(&format!("level_{level:02}_logits"), &logits)?;
            let next = if let Some(state) = rng_state.as_mut() {
                sample_categorical_cpu(
                    b,
                    &logits,
                    batch,
                    length,
                    cfg.vocab_size[level],
                    cfg.temperature,
                    state,
                )?
            } else {
                b.select_last(&logits, true)?
            };
            trace.indices(&format!("generator.level_{level:02}.codes"), &next)?;
            trace.stage_indices(&format!("level_{level:02}_sampled"), &next)?;
            codes.push(next.clone());
            previous = Some(next);
        }

        // Release prefix activations and cached K/V before final vocabulary projection.
        drop(prefix_cache);
        drop(prefix);

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
        let (logits, ids) = self.codec.decode_latent_outputs(&latent, &mut NoTrace)?;
        trace.stage_indices("codec_decode_done", &ids)?;
        Ok(Generated { codes, logits, ids })
    }

    fn project_head(&self, level: usize, hidden: &B::Tensor) -> Result<B::Tensor> {
        let b = &self.backend;
        match &self.heads {
            LocalHeads::UntiedAffine(heads) => {
                b.project_bias(hidden, &heads[level].weight, &heads[level].bias)
            }
            LocalHeads::UntiedLinear(weights) => b.project(hidden, &weights[level]),
            LocalHeads::SharedHeadBiases { weight, biases } => {
                b.project_bias(hidden, weight, &biases[level])
            }
        }
    }
}

fn preflight_generator(
    artifact: &Artifact,
    codec_cfg: &CodecConfig,
    gen_cfg: &LocalGeneratorConfig,
    expected: &BTreeMap<String, Vec<usize>>,
) -> Result<()> {
    let file_bytes = artifact
        .file_bytes("generator.safetensors")
        .map_err(|e| Error::Weights(e.to_string()))?;
    if file_bytes > LOCAL_GENERATOR_LIMIT {
        return Err(Error::Weights("local generator exceeds 4 GiB limit".into()));
    }
    if artifact.is_streaming_bundle("generator.safetensors") {
        if !artifact.bundle_f32_weights_verified_finite("generator.safetensors") {
            return Err(Error::Weights(
                "streaming local generator missing finite-value preflight".into(),
            ));
        }
        let entries = artifact
            .bundle_tensors("generator.safetensors")
            .map_err(|e| Error::Weights(e.to_string()))?;
        let by_name: BTreeMap<_, _> = entries
            .iter()
            .map(|entry| (entry.name.as_str(), entry))
            .collect();
        let mut optional_count = 0usize;
        if let Some(level_entry) = by_name.get("lvl_1L") {
            optional_count += 1;
            let bytes = artifact
                .read_bundle_tensor("generator.safetensors", "lvl_1L")
                .map_err(|e| Error::Weights(e.to_string()))?;
            xsum_generator::validate_levels_raw(
                level_entry.dtype,
                &level_entry.shape,
                &bytes,
                codec_cfg.levels(),
            )?;
        }
        if let Some(mask_entry) = by_name.get("attn_mask_for_masking") {
            optional_count += 1;
            let bytes = artifact
                .read_bundle_tensor("generator.safetensors", "attn_mask_for_masking")
                .map_err(|e| Error::Weights(e.to_string()))?;
            xsum_generator::validate_mask_raw(
                mask_entry.dtype,
                &mask_entry.shape,
                &bytes,
                codec_cfg.levels(),
                gen_cfg.prefix_len,
            )?;
        }
        if entries.len() != expected.len() + optional_count {
            return Err(Error::Weights(
                "local generator tensor inventory differs".into(),
            ));
        }
        for (name, shape) in expected {
            let entry = by_name
                .get(name.as_str())
                .ok_or_else(|| Error::Weights(format!("missing local generator tensor {name}")))?;
            if !matches!(entry.dtype, ncm_io::TensorDtype::F32)
                || entry.shape != *shape
                || entry.bytes != shape.iter().product::<usize>() * 4
            {
                return Err(Error::Weights(format!(
                    "invalid local generator tensor {name}"
                )));
            }
        }
        return Ok(());
    }

    let bytes = artifact
        .file("generator.safetensors")
        .map_err(|e| Error::Weights(e.to_string()))?;
    let tensors = SafeTensors::deserialize(bytes).map_err(|e| Error::Weights(e.to_string()))?;
    let mut optional_count = 0usize;
    if let Ok(t) = tensors.tensor("lvl_1L") {
        optional_count += 1;
        let dtype = match t.dtype() {
            Dtype::F32 => ncm_io::TensorDtype::F32,
            Dtype::I64 => ncm_io::TensorDtype::I64,
            _ => return Err(Error::Weights("invalid local lvl_1L dtype".into())),
        };
        xsum_generator::validate_levels_raw(dtype, t.shape(), t.data(), codec_cfg.levels())?;
    }
    if let Ok(t) = tensors.tensor("attn_mask_for_masking") {
        optional_count += 1;
        let dtype = match t.dtype() {
            Dtype::F32 => ncm_io::TensorDtype::F32,
            Dtype::I64 => ncm_io::TensorDtype::I64,
            _ => {
                return Err(Error::Weights(
                    "invalid local attn_mask_for_masking dtype".into(),
                ));
            }
        };
        xsum_generator::validate_mask_raw(
            dtype,
            t.shape(),
            t.data(),
            codec_cfg.levels(),
            gen_cfg.prefix_len,
        )?;
    }
    if tensors.len() != expected.len() + optional_count {
        return Err(Error::Weights(
            "local generator tensor inventory differs".into(),
        ));
    }
    for (name, shape) in expected {
        let t = tensors
            .tensor(name)
            .map_err(|e| Error::Weights(e.to_string()))?;
        if t.dtype() != Dtype::F32
            || t.shape() != shape
            || t.data().len() != shape.iter().product::<usize>() * 4
            || has_nonfinite_f32_le(t.data())
        {
            return Err(Error::Weights(format!(
                "invalid local generator tensor {name}"
            )));
        }
    }
    Ok(())
}

fn upload_prefix_pad_mask<B: GeneratorBackend>(
    b: &B,
    prefix_tokens: &[u32],
    batch: usize,
    prefix_len: usize,
    pad: u32,
) -> Result<B::Tensor> {
    let mut mask = vec![0.0f32; batch * prefix_len];
    for (i, &tok) in prefix_tokens.iter().enumerate() {
        if tok == pad {
            mask[i] = f32::NEG_INFINITY;
        }
    }
    b.upload_f32(&mask, &[batch, 1, 1, prefix_len])
}

fn upload_target_pad_mask<B: GeneratorBackend>(
    b: &B,
    prefix_tokens: &[u32],
    batch: usize,
    prefix_len: usize,
    target_len: usize,
    pad: u32,
) -> Result<B::Tensor> {
    let seq = prefix_len + target_len;
    let mut mask = vec![0.0f32; batch * seq];
    for (row_idx, row) in prefix_tokens.chunks_exact(prefix_len).enumerate() {
        let base = row_idx * seq;
        for (col, &tok) in row.iter().enumerate() {
            if tok == pad {
                mask[base + col] = f32::NEG_INFINITY;
            }
        }
    }
    b.upload_f32(&mask, &[batch, 1, 1, seq])
}

fn upload_full_pad_mask<B: GeneratorBackend>(
    b: &B,
    prefix_tokens: &[u32],
    batch: usize,
    prefix_len: usize,
    target_len: usize,
    pad: u32,
) -> Result<B::Tensor> {
    let seq = prefix_len + target_len;
    let mut mask = vec![0.0f32; batch * seq * seq];
    for (row_idx, row) in prefix_tokens.chunks_exact(prefix_len).enumerate() {
        let base = row_idx * seq * seq;
        for r in 0..seq {
            for (c, &tok) in row.iter().enumerate() {
                if tok == pad {
                    mask[base + r * seq + c] = f32::NEG_INFINITY;
                }
            }
            if r < prefix_len {
                for c in prefix_len..seq {
                    mask[base + r * seq + c] = f32::NEG_INFINITY;
                }
            }
        }
    }
    b.upload_f32(&mask, &[batch, 1, seq, seq])
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn sample_categorical_cpu<B: GeneratorBackend>(
    b: &B,
    logits: &B::Tensor,
    batch: usize,
    length: usize,
    vocab: usize,
    temperature: f64,
    rng_state: &mut u64,
) -> Result<B::Indices> {
    let values = b.read_f32(logits)?;
    let rows = batch * length;
    if values.len() != rows * vocab {
        return Err(Error::Backend(
            "local categorical logits shape mismatch".into(),
        ));
    }
    let inv_temp = 1.0 / temperature;
    let mut picks = Vec::with_capacity(rows);
    for row in values.chunks_exact(vocab) {
        let mut max = f64::NEG_INFINITY;
        for &v in row {
            if !v.is_finite() {
                return Err(Error::Backend("nonfinite local generator logit".into()));
            }
            let scaled = f64::from(v) * inv_temp;
            if scaled > max {
                max = scaled;
            }
        }
        let mut total = 0.0f64;
        for &v in row {
            total += (f64::from(v) * inv_temp - max).exp();
        }
        if !total.is_finite() || total <= 0.0 {
            return Err(Error::Backend(
                "degenerate local categorical distribution".into(),
            ));
        }
        *rng_state = splitmix64(*rng_state);
        // Uniform in (0, 1) using top 53 bits with midpoint offset.
        let u = (((*rng_state >> 11) as f64) + 0.5) / ((1u64 << 53) as f64);
        let threshold = u * total;
        let mut acc = 0.0f64;
        let mut chosen = (vocab - 1) as u32;
        for (idx, &v) in row.iter().enumerate() {
            acc += (f64::from(v) * inv_temp - max).exp();
            if acc >= threshold {
                chosen = idx as u32;
                break;
            }
        }
        picks.push(chosen);
    }
    b.upload_indices(&picks, &[batch, length])
}
