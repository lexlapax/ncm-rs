//! Verified `local-ncm-v1` artifact export/import and atomic training checkpoint save/resume.
use crate::{Result, TrainError, TrainStage, ema::EmaCodebookState, schedule::TensorMap};
use ncm_core::LocalNcmConfig;
use ncm_io::{Artifact, FileRecord, Manifest, TensorDtype, TensorRecord, sha256};
use safetensors::{
    SafeTensors,
    tensor::{Dtype, TensorView},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

/// Serializable trainer state saved alongside checkpoint safetensors for deterministic resume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub schema: String,
    pub stage: TrainStage,
    pub step: usize,
    pub total_steps: usize,
    pub rng_state: u64,
    pub last_loss: f64,
    #[serde(default)]
    pub lora_rank: usize,
    #[serde(default)]
    pub lora_alpha: f64,
}

/// Complete loaded training checkpoint state.
#[derive(Debug, Clone)]
pub struct LoadedCheckpoint {
    pub meta: CheckpointMeta,
    pub weights: TensorMap,
    pub optim_m: TensorMap,
    pub optim_v: TensorMap,
    pub ema: Option<EmaCodebookState>,
}

/// Serialize a `TensorMap` into safetensors bytes and compute per-tensor SHA-256 records.
/// Rejects any non-finite (`NaN`/`Inf`) weights.
pub fn serialize_tensor_map(map: &TensorMap) -> Result<(Vec<u8>, Vec<TensorRecord>)> {
    let mut raw: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::with_capacity(map.len());
    let mut records = Vec::with_capacity(map.len());
    for (name, (shape, vals)) in map {
        let expected: usize = shape.iter().product();
        if vals.len() != expected {
            return Err(TrainError::Artifact(format!(
                "tensor `{name}` length {} != shape product {expected}",
                vals.len()
            )));
        }
        let mut bytes = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            if !v.is_finite() {
                return Err(TrainError::Artifact(format!(
                    "non-finite FP32 value in tensor `{name}`"
                )));
            }
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        records.push(TensorRecord {
            name: name.clone(),
            shape: shape.clone(),
            dtype: TensorDtype::F32,
            bytes: bytes.len(),
            sha256: sha256(&bytes),
        });
        raw.push((name.clone(), shape.clone(), bytes));
    }
    let views: Vec<(&str, TensorView<'_>)> = raw
        .iter()
        .map(|(name, shape, bytes)| {
            let tv = TensorView::new(Dtype::F32, shape.clone(), bytes)
                .map_err(|e| TrainError::Artifact(format!("safetensors view `{name}`: {e}")))?;
            Ok((name.as_str(), tv))
        })
        .collect::<Result<Vec<_>>>()?;
    let serialized = safetensors::tensor::serialize(views, None)
        .map_err(|e| TrainError::Artifact(format!("safetensors serialize failed: {e}")))?;
    Ok((serialized, records))
}

/// Deserialize an FP32 safetensors file into a `TensorMap`, verifying finite values.
pub fn read_tensor_map(bytes: &[u8]) -> Result<TensorMap> {
    let st = SafeTensors::deserialize(bytes)
        .map_err(|e| TrainError::Artifact(format!("safetensors deserialize: {e}")))?;
    let mut map = TensorMap::new();
    for (name, view) in st.tensors() {
        if view.dtype() != Dtype::F32 {
            return Err(TrainError::Artifact(format!(
                "expected F32 tensor `{name}`, got {:?}",
                view.dtype()
            )));
        }
        let data = view.data();
        if !data.len().is_multiple_of(4) {
            return Err(TrainError::Artifact(format!(
                "misaligned F32 byte length for `{name}`"
            )));
        }
        let mut vals = Vec::with_capacity(data.len() / 4);
        for chunk in data.as_chunks::<4>().0 {
            let v = f32::from_le_bytes(*chunk);
            if !v.is_finite() {
                return Err(TrainError::Artifact(format!(
                    "non-finite value in `{name}`"
                )));
            }
            vals.push(v);
        }
        map.insert(name, (view.shape().to_vec(), vals));
    }
    Ok(map)
}

/// Loaded `local-ncm-v1` artifact contents for fine-tuning.
pub struct LoadedArtifactWeights {
    pub config_bytes: Vec<u8>,
    pub config: LocalNcmConfig,
    pub codec_weights: TensorMap,
    pub generator_weights: TensorMap,
    pub tokenizer_bytes: Option<Vec<u8>>,
}

/// Load validated codec + generator weights (plus raw `config.json` and optional `tokenizer.json`)
/// from an existing verified `local-ncm-v1` or upstream `tinystories-inference-v1` artifact
/// directory (for fine-tuning).
pub fn load_local_artifact_weights(artifact_dir: &Path) -> Result<LoadedArtifactWeights> {
    let artifact = Artifact::load_inference_streaming(artifact_dir).map_err(|e| {
        TrainError::Artifact(format!("load artifact {}: {e}", artifact_dir.display()))
    })?;
    match artifact.manifest().profile.as_str() {
        "local-ncm-v1" => load_native_local_weights(artifact_dir, &artifact),
        "tinystories-inference-v1" => load_upstream_tinystories_weights(artifact_dir, &artifact),
        other => Err(TrainError::Artifact(format!(
            "unsupported --init-from profile `{other}` (expected local-ncm-v1 or tinystories-inference-v1)"
        ))),
    }
}

fn load_native_local_weights(
    artifact_dir: &Path,
    artifact: &Artifact,
) -> Result<LoadedArtifactWeights> {
    let config_bytes = artifact
        .file("config.json")
        .map_err(|e| TrainError::Artifact(e.to_string()))?
        .to_vec();
    let config =
        LocalNcmConfig::from_json(&config_bytes).map_err(|e| TrainError::Config(e.to_string()))?;
    let codec_bytes = fs::read(artifact_dir.join("codec.safetensors"))
        .map_err(|e| TrainError::Io(format!("read codec.safetensors: {e}")))?;
    let gen_bytes = fs::read(artifact_dir.join("generator.safetensors"))
        .map_err(|e| TrainError::Io(format!("read generator.safetensors: {e}")))?;
    let codec_weights = read_tensor_map(&codec_bytes)?;
    let generator_weights = read_tensor_map(&gen_bytes)?;
    let tokenizer_bytes = artifact.file("tokenizer.json").ok().map(|b| b.to_vec());
    Ok(LoadedArtifactWeights {
        config_bytes,
        config,
        codec_weights,
        generator_weights,
        tokenizer_bytes,
    })
}

fn read_f32_tensor_view(
    name: &str,
    view: &safetensors::tensor::TensorView<'_>,
) -> Result<Vec<f32>> {
    if view.dtype() != Dtype::F32 {
        return Err(TrainError::Artifact(format!(
            "expected F32 tensor `{name}`, got {:?}",
            view.dtype()
        )));
    }
    let data = view.data();
    if !data.len().is_multiple_of(4) {
        return Err(TrainError::Artifact(format!(
            "misaligned F32 byte length for `{name}`"
        )));
    }
    let mut vals = Vec::with_capacity(data.len() / 4);
    for &chunk in data.as_chunks::<4>().0 {
        let v = f32::from_le_bytes(chunk);
        if !v.is_finite() {
            return Err(TrainError::Artifact(format!(
                "non-finite value in `{name}`"
            )));
        }
        vals.push(v);
    }
    Ok(vals)
}

/// Admit a verified upstream `tinystories-inference-v1` artifact (15k EMA or legacy) and
/// convert its tied-codebook generator + level-0 unigram distribution into a mathematically
/// equivalent `local-ncm-v1` (`UntiedAffine`) representation ready for fine-tuning and export.
fn load_upstream_tinystories_weights(
    artifact_dir: &Path,
    artifact: &Artifact,
) -> Result<LoadedArtifactWeights> {
    let raw_cfg_bytes = artifact
        .file("config.json")
        .map_err(|e| TrainError::Artifact(e.to_string()))?;
    let raw_cfg: serde_json::Value = serde_json::from_slice(raw_cfg_bytes)
        .map_err(|e| TrainError::Config(format!("parse TinyStories config.json: {e}")))?;
    let vqvae = raw_cfg
        .get("tokenizer")
        .and_then(|t| t.get("vqvae"))
        .ok_or_else(|| TrainError::Config("TinyStories config missing tokenizer.vqvae".into()))?;
    let model_raw = raw_cfg
        .get("model")
        .ok_or_else(|| TrainError::Config("TinyStories config missing model".into()))?;

    let vq_vocab = vqvae
        .get("vqvae_vocab_size")
        .and_then(|v| v.as_array())
        .ok_or_else(|| TrainError::Config("missing vqvae_vocab_size".into()))?;
    let n_levels = vq_vocab.len();
    let n_embd = model_raw
        .get("n_embd")
        .and_then(|v| v.as_u64())
        .unwrap_or(768) as usize;
    let n_head = model_raw
        .get("n_head")
        .and_then(|v| v.as_u64())
        .unwrap_or(12) as usize;
    let n_layer = model_raw
        .get("n_layer")
        .and_then(|v| v.as_u64())
        .unwrap_or(12) as usize;
    let prefix_len = model_raw
        .get("prefix_len")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as usize;
    let context_length = model_raw
        .get("context_length")
        .and_then(|v| v.as_u64())
        .unwrap_or(129) as usize;
    let rope_base = model_raw
        .get("rope_base")
        .and_then(|v| v.as_u64())
        .unwrap_or(100_000) as usize;
    let pad_token_id = model_raw
        .get("prefix_pad_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as u32;
    let temperature = raw_cfg
        .get("generation")
        .and_then(|g| g.get("temperature"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);

    let local_cfg_val = serde_json::json!({
        "tokenizer": vqvae,
        "model": {
            "attn_pattern": "block_diagonal",
            "bias": true,
            "context_length": context_length,
            "dropout": 0.0,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "prefix_len": prefix_len,
            "query_level0": true,
            "raw_token_prefix": true,
            "rope_base": rope_base,
            "tie_heads_to_codebook": false,
            "use_level_encoding": true,
            "use_positional_encoding": true,
            "use_qk_norm": true,
            "use_rope": true,
            "vocab_size": vq_vocab,
            "head_kind": "untied_affine",
            "pad_token_id": pad_token_id,
            "eos_token_id": 0,
            "temperature": temperature,
        }
    });
    let config_bytes = serde_json::to_vec_pretty(&local_cfg_val)
        .map_err(|e| TrainError::Config(format!("serialize converted local config: {e}")))?;
    let config =
        LocalNcmConfig::from_json(&config_bytes).map_err(|e| TrainError::Config(e.to_string()))?;

    let codec_bytes = fs::read(artifact_dir.join("codec.safetensors"))
        .map_err(|e| TrainError::Io(format!("read codec.safetensors: {e}")))?;
    let codec_weights = read_tensor_map(&codec_bytes)?;

    let gen_bytes = fs::read(artifact_dir.join("generator.safetensors"))
        .map_err(|e| TrainError::Io(format!("read generator.safetensors: {e}")))?;
    let gen_st = SafeTensors::deserialize(&gen_bytes)
        .map_err(|e| TrainError::Artifact(format!("parse generator.safetensors: {e}")))?;

    let inf_bytes = artifact
        .file("inference.safetensors")
        .map_err(|e| TrainError::Artifact(format!("missing inference.safetensors: {e}")))?;
    let inf_st = SafeTensors::deserialize(inf_bytes)
        .map_err(|e| TrainError::Artifact(format!("parse inference.safetensors: {e}")))?;
    let l0_unigram_view = inf_st
        .tensor("level0_unigram")
        .map_err(|e| TrainError::Artifact(format!("missing level0_unigram: {e}")))?;
    let l0_unigram = read_f32_tensor_view("level0_unigram", &l0_unigram_view)?;

    let embed_dim = config.codec().raw().embed_dim;
    let l0_len = config.codec().levels()[0];
    let l0_vocab = config.model().vocab_size[0];
    if l0_unigram.len() != l0_vocab {
        return Err(TrainError::Artifact(format!(
            "level0_unigram length {} != expected {l0_vocab}",
            l0_unigram.len()
        )));
    }

    let mut generator_weights = TensorMap::new();
    let mut head_projs: Vec<(usize, Vec<f32>, f32)> = Vec::with_capacity(n_levels - 1);

    for (name, view) in gen_st.tensors() {
        if name == "lvl_1L" || name == "attn_mask_for_masking" {
            continue;
        }
        if name == "transformer.wle.weight" {
            let old_wle = read_f32_tensor_view(&name, &view)?;
            if view.shape() != [n_levels - 1, n_embd] {
                return Err(TrainError::Artifact(format!(
                    "unexpected TinyStories transformer.wle.weight shape {:?}",
                    view.shape()
                )));
            }
            // Prepend a zero row for level 0 (which predicts the unigram distribution).
            let mut new_wle = vec![0.0f32; n_levels * n_embd];
            new_wle[n_embd..].copy_from_slice(&old_wle);
            generator_weights.insert(name, (vec![n_levels, n_embd], new_wle));
            continue;
        }
        if name.starts_with("lm_heads.") {
            continue;
        }
        let vals = read_f32_tensor_view(&name, &view)?;
        generator_weights.insert(name, (view.shape().to_vec(), vals));
    }

    // Collect upstream tied-head parameters for levels 1..n_levels.
    for i in 0..(n_levels - 1) {
        let proj_name = format!("lm_heads.{i}.proj.weight");
        let temp_name = format!("lm_heads.{i}.log_inv_T");
        let proj_view = gen_st
            .tensor(&proj_name)
            .map_err(|e| TrainError::Artifact(format!("missing {proj_name}: {e}")))?;
        let temp_view = gen_st
            .tensor(&temp_name)
            .map_err(|e| TrainError::Artifact(format!("missing {temp_name}: {e}")))?;
        if proj_view.shape() != [embed_dim, n_embd] {
            return Err(TrainError::Artifact(format!(
                "unexpected {proj_name} shape {:?}",
                proj_view.shape()
            )));
        }
        let proj_vals = read_f32_tensor_view(&proj_name, &proj_view)?;
        let temp_vals = read_f32_tensor_view(&temp_name, &temp_view)?;
        if temp_vals.len() != 1 {
            return Err(TrainError::Artifact(format!(
                "expected scalar {temp_name}, got len {}",
                temp_vals.len()
            )));
        }
        head_projs.push((i + 1, proj_vals, temp_vals[0]));
    }

    // Level 0: l0_queries = 0, lm_heads.0.weight = 0, lm_heads.0.bias = log(p_unigram).
    generator_weights.insert(
        "l0_queries".into(),
        (vec![1, l0_len, embed_dim], vec![0.0f32; l0_len * embed_dim]),
    );
    generator_weights.insert(
        "lm_heads.0.weight".into(),
        (vec![l0_vocab, n_embd], vec![0.0f32; l0_vocab * n_embd]),
    );
    let unigram_sum: f64 = l0_unigram.iter().map(|&v| f64::from(v).max(0.0)).sum();
    if unigram_sum <= 0.0 || !unigram_sum.is_finite() {
        return Err(TrainError::Artifact(
            "invalid level0_unigram probability sum".into(),
        ));
    }
    let l0_bias: Vec<f32> = l0_unigram
        .iter()
        .map(|&p| ((f64::from(p).max(0.0) / unigram_sum).max(1e-30)).ln() as f32)
        .collect();
    generator_weights.insert("lm_heads.0.bias".into(), (vec![l0_vocab], l0_bias));

    // Expand tied codebook heads for levels 1..n_levels into exact UntiedAffine heads in parallel:
    //   logits(h) = (inv_T / mean(||E||^2)) * (2 * (h @ P^T) @ E^T - ||E||^2)
    //             = h @ W_eff^T + b_eff
    //   W_eff = (2 * inv_T / mean(||E||^2)) * (E @ P)  [vocab, n_embd]
    //   b_eff = -(inv_T / mean(||E||^2)) * ||E||^2     [vocab]
    type ConvertedHead = (usize, Vec<f32>, Vec<f32>);
    let converted_heads: Result<Vec<ConvertedHead>> =
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(head_projs.len());
            for (level, proj, log_inv_t) in &head_projs {
                let level = *level;
                let vocab = config.model().vocab_size[level];
                let book_name = format!("vq.embeddings.{level}.weight");
                let (_, book) = codec_weights
                    .get(&book_name)
                    .ok_or_else(|| TrainError::Artifact(format!("missing codec `{book_name}`")))?;
                if book.len() != vocab * embed_dim {
                    return Err(TrainError::Artifact(format!(
                        "unexpected `{book_name}` length {}",
                        book.len()
                    )));
                }
                let log_inv_t = *log_inv_t;
                handles.push(scope.spawn(move || -> Result<ConvertedHead> {
                    let mut sq_norms = Vec::with_capacity(vocab);
                    let mut sum_sq = 0.0f64;
                    for k in 0..vocab {
                        let row = &book[k * embed_dim..(k + 1) * embed_dim];
                        let mut n2 = 0.0f64;
                        for &v in row {
                            n2 += f64::from(v) * f64::from(v);
                        }
                        sq_norms.push(n2);
                        sum_sq += n2;
                    }
                    let mean_sq = (sum_sq / vocab as f64).max(1e-12);
                    let inv_t = f64::from(log_inv_t).exp();
                    let scale = inv_t / mean_sq;
                    let w_scale = 2.0 * scale;

                    let mut w_eff = vec![0.0f32; vocab * n_embd];
                    let mut b_eff = Vec::with_capacity(vocab);
                    for k in 0..vocab {
                        let b_val = (-scale * sq_norms[k]) as f32;
                        if !b_val.is_finite() {
                            return Err(TrainError::Artifact(format!(
                                "non-finite converted bias in lm_heads.{level}"
                            )));
                        }
                        b_eff.push(b_val);

                        let e_row = &book[k * embed_dim..(k + 1) * embed_dim];
                        let out_row = &mut w_eff[k * n_embd..(k + 1) * n_embd];
                        // Accumulate out_row[j] += e_row[d] * proj[d * n_embd + j] (cache-friendly row order)
                        for (d, &e_kd) in e_row.iter().enumerate() {
                            let e_scaled = (f64::from(e_kd) * w_scale) as f32;
                            let p_row = &proj[d * n_embd..(d + 1) * n_embd];
                            for (dst, &p_dj) in out_row.iter_mut().zip(p_row.iter()) {
                                *dst += e_scaled * p_dj;
                            }
                        }
                        if out_row.iter().any(|v| !v.is_finite()) {
                            return Err(TrainError::Artifact(format!(
                                "non-finite converted weight in lm_heads.{level}"
                            )));
                        }
                    }
                    Ok((level, w_eff, b_eff))
                }));
            }
            let mut out = Vec::with_capacity(handles.len());
            for h in handles {
                out.push(h.join().map_err(|_| {
                    TrainError::Artifact("head conversion thread panicked".into())
                })??);
            }
            Ok(out)
        });

    for (level, w_eff, b_eff) in converted_heads? {
        let vocab = config.model().vocab_size[level];
        generator_weights.insert(
            format!("lm_heads.{level}.weight"),
            (vec![vocab, n_embd], w_eff),
        );
        generator_weights.insert(format!("lm_heads.{level}.bias"), (vec![vocab], b_eff));
    }

    // Validate complete converted inventories before returning.
    validate_inventory(
        "codec.safetensors",
        &codec_weights,
        &config.codec().weight_shapes(),
    )?;
    validate_inventory(
        "generator.safetensors",
        &generator_weights,
        &config.model().weight_shapes(config.codec()),
    )?;

    let tokenizer_bytes = artifact.file("tokenizer.json").ok().map(|b| b.to_vec());
    Ok(LoadedArtifactWeights {
        config_bytes,
        config,
        codec_weights,
        generator_weights,
        tokenizer_bytes,
    })
}

/// Export a verified `local-ncm-v1` artifact directory directly loadable by `ncm inspect`,
/// `ncm sample`, `ncm run`, and `ncm serve`.
///
/// Writes to a staging directory and atomically replaces/renames into `output_dir` (if
/// `overwrite` is true, replaces an existing directory; otherwise fails if it exists).
#[allow(clippy::too_many_arguments)]
pub fn export_local_ncm_artifact(
    output_dir: &Path,
    config_json_bytes: &[u8],
    config: &LocalNcmConfig,
    codec_weights: &TensorMap,
    generator_weights: &TensorMap,
    tokenizer_json_bytes: Option<&[u8]>,
    provenance_json: &serde_json::Value,
    overwrite: bool,
) -> Result<PathBuf> {
    if output_dir.exists() && !overwrite {
        return Err(TrainError::Artifact(format!(
            "output directory {} already exists (pass --overwrite to replace)",
            output_dir.display()
        )));
    }

    // Validate complete expected inventories before writing anything.
    let exp_codec = config.codec().weight_shapes();
    let exp_gen = config.model().weight_shapes(config.codec());
    validate_inventory("codec.safetensors", codec_weights, &exp_codec)?;
    validate_inventory("generator.safetensors", generator_weights, &exp_gen)?;

    let (codec_bytes, codec_records) = serialize_tensor_map(codec_weights)?;
    let (gen_bytes, gen_records) = serialize_tensor_map(generator_weights)?;
    let prov_bytes = serde_json::to_vec_pretty(provenance_json)
        .map_err(|e| TrainError::Artifact(format!("provenance json: {e}")))?;

    let parent = output_dir.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .map_err(|e| TrainError::Io(format!("create parent {}: {e}", parent.display())))?;
    let stage_dir = parent.join(format!(
        ".ncm-train-export-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    ));
    fs::create_dir_all(&stage_dir)
        .map_err(|e| TrainError::Io(format!("create staging dir: {e}")))?;

    let write_res: Result<()> = (|| {
        let mut files = BTreeMap::<String, FileRecord>::new();
        let mut write_file = |name: &str, data: &[u8]| -> Result<()> {
            fs::write(stage_dir.join(name), data)
                .map_err(|e| TrainError::Io(format!("write {name}: {e}")))?;
            files.insert(
                name.to_owned(),
                FileRecord {
                    bytes: data.len() as u64,
                    sha256: sha256(data),
                },
            );
            Ok(())
        };

        write_file("config.json", config_json_bytes)?;
        write_file("codec.safetensors", &codec_bytes)?;
        write_file("generator.safetensors", &gen_bytes)?;
        write_file("provenance.json", &prov_bytes)?;
        if let Some(tok) = tokenizer_json_bytes {
            write_file("tokenizer.json", tok)?;
        }

        let mut tensors = BTreeMap::new();
        tensors.insert("codec.safetensors".to_owned(), codec_records);
        tensors.insert("generator.safetensors".to_owned(), gen_records);

        let manifest = Manifest {
            schema_version: 1,
            profile: "local-ncm-v1".to_owned(),
            upstream_commit: "ncm-train-native".to_owned(),
            checkpoint_repository: "local/ncm-train".to_owned(),
            checkpoint_revision: "local-v1".to_owned(),
            files,
            tensors,
        };
        let manifest_bytes = serde_json::to_vec_pretty(&manifest)
            .map_err(|e| TrainError::Artifact(format!("manifest json: {e}")))?;
        fs::write(stage_dir.join("manifest.json"), manifest_bytes)
            .map_err(|e| TrainError::Io(format!("write manifest.json: {e}")))?;
        Ok(())
    })();

    if let Err(e) = write_res {
        let _ = fs::remove_dir_all(&stage_dir);
        return Err(e);
    }

    if output_dir.exists() {
        fs::remove_dir_all(output_dir)
            .map_err(|e| TrainError::Io(format!("remove existing output: {e}")))?;
    }
    fs::rename(&stage_dir, output_dir).map_err(|e| {
        let _ = fs::remove_dir_all(&stage_dir);
        TrainError::Io(format!("rename export into {}: {e}", output_dir.display()))
    })?;
    Ok(output_dir.to_path_buf())
}

fn validate_inventory(
    bundle: &str,
    actual: &TensorMap,
    expected: &BTreeMap<String, Vec<usize>>,
) -> Result<()> {
    if actual.len() != expected.len() {
        return Err(TrainError::Artifact(format!(
            "{bundle}: expected {} tensors, got {}",
            expected.len(),
            actual.len()
        )));
    }
    for (name, exp_shape) in expected {
        let Some((shape, _)) = actual.get(name) else {
            return Err(TrainError::Artifact(format!(
                "{bundle}: missing required tensor `{name}`"
            )));
        };
        if shape != exp_shape {
            return Err(TrainError::Artifact(format!(
                "{bundle}/{name}: expected shape {exp_shape:?}, got {shape:?}"
            )));
        }
    }
    Ok(())
}

/// Save a deterministic training checkpoint (weights + AdamW moments + optional EMA state + metadata).
pub fn save_checkpoint(
    dir: &Path,
    meta: &CheckpointMeta,
    weights: &TensorMap,
    optim_m: &TensorMap,
    optim_v: &TensorMap,
    ema: Option<&EmaCodebookState>,
) -> Result<()> {
    fs::create_dir_all(dir)
        .map_err(|e| TrainError::Io(format!("create checkpoint dir {}: {e}", dir.display())))?;
    let (w_bytes, _) = serialize_tensor_map(weights)?;
    let (m_bytes, _) = serialize_tensor_map(optim_m)?;
    let (v_bytes, _) = serialize_tensor_map(optim_v)?;
    let meta_bytes = serde_json::to_vec_pretty(meta)
        .map_err(|e| TrainError::Artifact(format!("checkpoint meta: {e}")))?;

    write_atomic(dir, "weights.safetensors", &w_bytes)?;
    write_atomic(dir, "optim_m.safetensors", &m_bytes)?;
    write_atomic(dir, "optim_v.safetensors", &v_bytes)?;
    if let Some(ema_state) = ema {
        let (ema_bytes, _) = serialize_tensor_map(&ema_state.to_tensor_map())?;
        write_atomic(dir, "ema.safetensors", &ema_bytes)?;
    }
    write_atomic(dir, "trainer_state.json", &meta_bytes)?;
    Ok(())
}

/// Load a training checkpoint written by [`save_checkpoint`].
pub fn load_checkpoint(dir: &Path, config: &LocalNcmConfig) -> Result<LoadedCheckpoint> {
    let meta_bytes = fs::read(dir.join("trainer_state.json"))
        .map_err(|e| TrainError::Io(format!("read trainer_state.json: {e}")))?;
    let meta: CheckpointMeta = serde_json::from_slice(&meta_bytes)
        .map_err(|e| TrainError::Artifact(format!("parse trainer_state.json: {e}")))?;
    let weights = read_tensor_map(
        &fs::read(dir.join("weights.safetensors"))
            .map_err(|e| TrainError::Io(format!("read weights.safetensors: {e}")))?,
    )?;
    let optim_m = read_tensor_map(
        &fs::read(dir.join("optim_m.safetensors"))
            .map_err(|e| TrainError::Io(format!("read optim_m.safetensors: {e}")))?,
    )?;
    let optim_v = read_tensor_map(
        &fs::read(dir.join("optim_v.safetensors"))
            .map_err(|e| TrainError::Io(format!("read optim_v.safetensors: {e}")))?,
    )?;
    let ema_path = dir.join("ema.safetensors");
    let ema = if ema_path.is_file() {
        let ema_map = read_tensor_map(
            &fs::read(&ema_path)
                .map_err(|e| TrainError::Io(format!("read ema.safetensors: {e}")))?,
        )?;
        EmaCodebookState::from_tensor_map(config.codec(), &ema_map)
    } else {
        None
    };
    Ok(LoadedCheckpoint {
        meta,
        weights,
        optim_m,
        optim_v,
        ema,
    })
}

fn write_atomic(dir: &Path, name: &str, bytes: &[u8]) -> Result<()> {
    let tmp = dir.join(format!(".{name}.tmp.{}", std::process::id()));
    fs::write(&tmp, bytes).map_err(|e| TrainError::Io(format!("write {}: {e}", tmp.display())))?;
    fs::rename(&tmp, dir.join(name)).map_err(|e| TrainError::Io(format!("rename {name}: {e}")))?;
    Ok(())
}
