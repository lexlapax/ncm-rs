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
/// from an existing verified `local-ncm-v1` artifact directory (for fine-tuning).
pub fn load_local_artifact_weights(artifact_dir: &Path) -> Result<LoadedArtifactWeights> {
    let artifact = Artifact::load_inference_streaming(artifact_dir).map_err(|e| {
        TrainError::Artifact(format!(
            "load local-ncm-v1 artifact {}: {e}",
            artifact_dir.display()
        ))
    })?;
    if artifact.manifest().profile != "local-ncm-v1" {
        return Err(TrainError::Artifact(format!(
            "expected local-ncm-v1 artifact, got `{}`",
            artifact.manifest().profile
        )));
    }
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
