//! Verified unmerged LoRA adapter artifact format (`ncm-lora-adapter-v1`).
//!
//! Stores only the trained low-rank `lora_a` and `lora_b` matrices in `adapter.safetensors`
//! alongside a SHA-256-pinned `adapter_manifest.json` bound to the base model's
//! `manifest.json` SHA-256. Fails closed on any file tamper, tensor tamper, non-finite value,
//! symlink, or base-model mismatch.

use crate::{Error, FileRecord, Result, TensorDtype, TensorRecord, bounded_read, invalid, sha256};
use safetensors::{
    Dtype, SafeTensors,
    tensor::{TensorView, serialize},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

pub const LORA_ADAPTER_SCHEMA: &str = "ncm-lora-adapter-v1";
pub const MAX_ADAPTER_MANIFEST_BYTES: u64 = 4 * 1024 * 1024;
pub const MAX_ADAPTER_WEIGHTS_BYTES: u64 = 512 * 1024 * 1024;

/// Verified manifest metadata for an unmerged `ncm-lora-adapter-v1` artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoraAdapterManifest {
    pub schema_version: u32,
    pub schema: String,
    pub base_profile: String,
    pub base_manifest_sha256: String,
    pub n_layer: usize,
    pub n_embd: usize,
    pub rank: usize,
    pub alpha: f64,
    pub adapter_file: FileRecord,
    pub tensors: Vec<TensorRecord>,
}

/// Verified in-memory LoRA adapter artifact ready for upload onto any backend.
#[derive(Debug, Clone)]
pub struct LoraAdapterArtifact {
    pub manifest: LoraAdapterManifest,
    pub manifest_sha256: String,
    /// Map from tensor name (e.g. `transformer.blocks.0.attn.c_attn.weight.lora_a`) to `(shape, fp32_values)`.
    pub tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
}

/// Metadata required when writing a new `ncm-lora-adapter-v1` artifact.
#[derive(Debug, Clone)]
pub struct LoraAdapterWriteSpec {
    pub base_profile: String,
    pub base_manifest_sha256: String,
    pub n_layer: usize,
    pub n_embd: usize,
    pub rank: usize,
    pub alpha: f64,
}

fn check_hash_hex(h: &str) -> Result<()> {
    if h.len() != 64 || !h.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
        return Err(invalid("invalid SHA-256 hex digest in adapter manifest"));
    }
    Ok(())
}

/// Expected tensor names and exact shapes for a LoRA adapter at `(n_layer, n_embd, rank)`.
pub fn expected_lora_shapes(
    n_layer: usize,
    n_embd: usize,
    rank: usize,
) -> BTreeMap<String, Vec<usize>> {
    let mut out = BTreeMap::new();
    for i in 0..n_layer {
        let p = format!("transformer.blocks.{i}");
        for (suffix, out_dim, in_dim) in [
            ("attn.c_attn.weight", 3 * n_embd, n_embd),
            ("attn.c_proj.weight", n_embd, n_embd),
            ("mlp.c_fc.weight", 4 * n_embd, n_embd),
            ("mlp.c_proj.weight", n_embd, 4 * n_embd),
        ] {
            out.insert(format!("{p}.{suffix}.lora_a"), vec![rank, in_dim]);
            out.insert(format!("{p}.{suffix}.lora_b"), vec![out_dim, rank]);
        }
    }
    out
}

/// Atomically write a verified `ncm-lora-adapter-v1` directory (`adapter.safetensors` +
/// `adapter_manifest.json`). Fails if `output_dir` already exists unless `overwrite` is true.
pub fn write_lora_adapter(
    output_dir: &Path,
    spec: &LoraAdapterWriteSpec,
    tensors: &BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    overwrite: bool,
) -> Result<PathBuf> {
    if spec.rank == 0 || spec.rank > 256 {
        return Err(invalid("LoRA adapter rank must be in 1..=256"));
    }
    if !spec.alpha.is_finite() || spec.alpha <= 0.0 {
        return Err(invalid("LoRA adapter alpha must be finite and > 0"));
    }
    if spec.n_layer == 0 || spec.n_embd == 0 {
        return Err(invalid("LoRA adapter n_layer and n_embd must be > 0"));
    }
    check_hash_hex(&spec.base_manifest_sha256)?;
    if output_dir.exists() && !overwrite {
        return Err(invalid(format!(
            "adapter output directory {} already exists (pass --overwrite to replace)",
            output_dir.display()
        )));
    }

    let expected = expected_lora_shapes(spec.n_layer, spec.n_embd, spec.rank);
    if tensors.len() != expected.len() {
        return Err(invalid(format!(
            "LoRA adapter tensor count {} != expected {}",
            tensors.len(),
            expected.len()
        )));
    }

    let mut raw: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::with_capacity(tensors.len());
    let mut records = Vec::with_capacity(tensors.len());
    for (name, exp_shape) in &expected {
        let Some((shape, vals)) = tensors.get(name) else {
            return Err(invalid(format!("missing LoRA adapter tensor `{name}`")));
        };
        if shape != exp_shape {
            return Err(invalid(format!(
                "LoRA adapter tensor `{name}` shape {shape:?} != expected {exp_shape:?}"
            )));
        }
        let exp_elems: usize = exp_shape.iter().product();
        if vals.len() != exp_elems {
            return Err(invalid(format!(
                "LoRA adapter tensor `{name}` length {} != {exp_elems}",
                vals.len()
            )));
        }
        let mut bytes = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            if !v.is_finite() {
                return Err(invalid(format!(
                    "non-finite FP32 value in LoRA adapter tensor `{name}`"
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
            TensorView::new(Dtype::F32, shape.clone(), bytes)
                .map(|tv| (name.as_str(), tv))
                .map_err(|e| invalid(format!("adapter safetensors view `{name}`: {e}")))
        })
        .collect::<Result<Vec<_>>>()?;
    let safetensors_bytes = serialize(views, None)
        .map_err(|e| invalid(format!("adapter safetensors serialize: {e}")))?;

    let manifest = LoraAdapterManifest {
        schema_version: 1,
        schema: LORA_ADAPTER_SCHEMA.to_owned(),
        base_profile: spec.base_profile.clone(),
        base_manifest_sha256: spec.base_manifest_sha256.clone(),
        n_layer: spec.n_layer,
        n_embd: spec.n_embd,
        rank: spec.rank,
        alpha: spec.alpha,
        adapter_file: FileRecord {
            bytes: safetensors_bytes.len() as u64,
            sha256: sha256(&safetensors_bytes),
        },
        tensors: records,
    };
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|e| invalid(format!("serialize adapter_manifest.json: {e}")))?;

    let parent = output_dir.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let stage_dir = parent.join(format!(
        ".ncm-lora-adapter-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    ));
    fs::create_dir_all(&stage_dir)?;

    let res: Result<()> = (|| {
        fs::write(stage_dir.join("adapter.safetensors"), &safetensors_bytes)?;
        fs::write(stage_dir.join("adapter_manifest.json"), &manifest_bytes)?;
        Ok(())
    })();
    if let Err(err) = res {
        let _ = fs::remove_dir_all(&stage_dir);
        return Err(err);
    }
    if output_dir.exists() {
        fs::remove_dir_all(output_dir)?;
    }
    if let Err(err) = fs::rename(&stage_dir, output_dir) {
        let _ = fs::remove_dir_all(&stage_dir);
        return Err(Error::Io(err));
    }
    Ok(output_dir.to_path_buf())
}

/// Inspect just `adapter_manifest.json` (lightweight metadata check, no weight load).
pub fn inspect_lora_adapter_manifest(dir: &Path) -> Result<(LoraAdapterManifest, String)> {
    let root = dir.components().collect::<PathBuf>();
    if !fs::symlink_metadata(&root)?.file_type().is_dir() {
        return Err(invalid("LoRA adapter root must be a real directory"));
    }
    let manifest_bytes = bounded_read(
        &root.join("adapter_manifest.json"),
        MAX_ADAPTER_MANIFEST_BYTES,
    )?;
    let manifest_sha256 = sha256(&manifest_bytes);
    let manifest: LoraAdapterManifest = serde_json::from_slice(&manifest_bytes)?;
    validate_manifest_fields(&manifest)?;
    Ok((manifest, manifest_sha256))
}

fn validate_manifest_fields(m: &LoraAdapterManifest) -> Result<()> {
    if m.schema_version != 1 || m.schema != LORA_ADAPTER_SCHEMA {
        return Err(invalid("unsupported LoRA adapter schema or version"));
    }
    if m.base_profile.is_empty() || m.base_profile.len() > 128 {
        return Err(invalid("invalid base_profile in adapter manifest"));
    }
    check_hash_hex(&m.base_manifest_sha256)?;
    check_hash_hex(&m.adapter_file.sha256)?;
    if m.rank == 0 || m.rank > 256 || !m.alpha.is_finite() || m.alpha <= 0.0 {
        return Err(invalid("invalid rank or alpha in adapter manifest"));
    }
    if m.n_layer == 0 || m.n_layer > 256 || m.n_embd == 0 || m.n_embd > 16_384 {
        return Err(invalid("invalid model dimensions in adapter manifest"));
    }
    if m.adapter_file.bytes == 0 || m.adapter_file.bytes > MAX_ADAPTER_WEIGHTS_BYTES {
        return Err(invalid("adapter.safetensors size outside bounds"));
    }
    Ok(())
}

impl LoraAdapterArtifact {
    /// Strictly load and verify an unmerged `ncm-lora-adapter-v1` artifact.
    /// If `expected_base_manifest_sha256` is `Some`, fails closed if the adapter was
    /// trained against a different base artifact.
    pub fn load(
        dir: impl AsRef<Path>,
        expected_base_manifest_sha256: Option<&str>,
    ) -> Result<Self> {
        let root = dir.as_ref().components().collect::<PathBuf>();
        let (manifest, manifest_sha256) = inspect_lora_adapter_manifest(&root)?;
        if let Some(expected_sha) = expected_base_manifest_sha256
            && manifest.base_manifest_sha256 != expected_sha
        {
            return Err(invalid(format!(
                "LoRA adapter base_manifest_sha256 mismatch: adapter targets {} ({}), base is {}",
                manifest.base_profile,
                &manifest.base_manifest_sha256[..12],
                &expected_sha[..expected_sha.len().min(12)]
            )));
        }

        let expected_shapes =
            expected_lora_shapes(manifest.n_layer, manifest.n_embd, manifest.rank);
        if manifest.tensors.len() != expected_shapes.len() {
            return Err(invalid(format!(
                "adapter manifest tensor count {} != expected {}",
                manifest.tensors.len(),
                expected_shapes.len()
            )));
        }

        let st_bytes = bounded_read(
            &root.join("adapter.safetensors"),
            manifest.adapter_file.bytes,
        )?;
        if st_bytes.len() as u64 != manifest.adapter_file.bytes {
            return Err(invalid("adapter.safetensors byte size mismatch"));
        }
        if sha256(&st_bytes) != manifest.adapter_file.sha256 {
            return Err(invalid("adapter.safetensors SHA-256 mismatch"));
        }

        let st = SafeTensors::deserialize(&st_bytes)
            .map_err(|e| invalid(format!("adapter.safetensors deserialize: {e}")))?;
        if st.len() != expected_shapes.len() {
            return Err(invalid("adapter.safetensors tensor inventory mismatch"));
        }

        let mut tensors = BTreeMap::new();
        for rec in &manifest.tensors {
            check_hash_hex(&rec.sha256)?;
            if rec.dtype != TensorDtype::F32 {
                return Err(invalid(format!(
                    "adapter tensor `{}` must be F32",
                    rec.name
                )));
            }
            let Some(exp_shape) = expected_shapes.get(&rec.name) else {
                return Err(invalid(format!(
                    "unexpected tensor `{}` in adapter manifest",
                    rec.name
                )));
            };
            if &rec.shape != exp_shape {
                return Err(invalid(format!(
                    "adapter tensor `{}` shape mismatch",
                    rec.name
                )));
            }
            let view = st.tensor(&rec.name).map_err(|_| {
                invalid(format!(
                    "missing tensor `{}` in adapter.safetensors",
                    rec.name
                ))
            })?;
            if view.dtype() != Dtype::F32 || view.shape() != exp_shape.as_slice() {
                return Err(invalid(format!(
                    "adapter tensor `{}` dtype/shape mismatch",
                    rec.name
                )));
            }
            let data = view.data();
            if data.len() != rec.bytes || !data.len().is_multiple_of(4) {
                return Err(invalid(format!(
                    "adapter tensor `{}` byte length mismatch",
                    rec.name
                )));
            }
            if sha256(data) != rec.sha256 {
                return Err(invalid(format!(
                    "adapter tensor `{}` SHA-256 mismatch",
                    rec.name
                )));
            }
            let mut vals = Vec::with_capacity(data.len() / 4);
            for &chunk in data.as_chunks::<4>().0 {
                let v = f32::from_le_bytes(chunk);
                if !v.is_finite() {
                    return Err(invalid(format!(
                        "non-finite value in adapter tensor `{}`",
                        rec.name
                    )));
                }
                vals.push(v);
            }
            if tensors
                .insert(rec.name.clone(), (rec.shape.clone(), vals))
                .is_some()
            {
                return Err(invalid(format!(
                    "duplicate tensor `{}` in adapter manifest",
                    rec.name
                )));
            }
        }
        if tensors.len() != expected_shapes.len() {
            return Err(invalid("incomplete LoRA adapter tensor set"));
        }

        Ok(Self {
            manifest,
            manifest_sha256,
            tensors,
        })
    }

    /// Effective LoRA scale `alpha / rank`.
    pub fn scale(&self) -> f64 {
        self.manifest.alpha / (self.manifest.rank as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_adapter(
        n_layer: usize,
        n_embd: usize,
        rank: usize,
    ) -> BTreeMap<String, (Vec<usize>, Vec<f32>)> {
        let mut map = BTreeMap::new();
        for (idx, (name, shape)) in expected_lora_shapes(n_layer, n_embd, rank)
            .into_iter()
            .enumerate()
        {
            let n: usize = shape.iter().product();
            let vals = (0..n)
                .map(|k| ((idx * 17 + k) % 29) as f32 * 0.01 - 0.14)
                .collect();
            map.insert(name, (shape, vals));
        }
        map
    }

    #[test]
    fn lora_adapter_roundtrip_and_fail_closed_checks() {
        let base_sha = sha256(b"base-manifest-v1");
        let wrong_sha = sha256(b"different-base-manifest");
        let spec = LoraAdapterWriteSpec {
            base_profile: "local-ncm-v1".into(),
            base_manifest_sha256: base_sha.clone(),
            n_layer: 2,
            n_embd: 16,
            rank: 4,
            alpha: 8.0,
        };
        let tensors = synthetic_adapter(2, 16, 4);

        let dir = std::env::temp_dir().join(format!("ncm-lora-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        write_lora_adapter(&dir, &spec, &tensors, false).unwrap();

        // Duplicate without overwrite must fail.
        assert!(write_lora_adapter(&dir, &spec, &tensors, false).is_err());

        // Valid load with matching base hash succeeds.
        let loaded = LoraAdapterArtifact::load(&dir, Some(&base_sha)).unwrap();
        assert_eq!(loaded.manifest.rank, 4);
        assert_eq!(loaded.manifest.alpha, 8.0);
        assert_eq!(loaded.scale(), 2.0);
        assert_eq!(loaded.tensors, tensors);

        // Mismatched base hash fails closed.
        let err = LoraAdapterArtifact::load(&dir, Some(&wrong_sha)).unwrap_err();
        assert!(err.to_string().contains("base_manifest_sha256 mismatch"));

        // 1-byte weight tamper fails closed.
        let weights_path = dir.join("adapter.safetensors");
        let mut tampered = fs::read(&weights_path).unwrap();
        let last = tampered.len() - 1;
        tampered[last] ^= 0x01;
        fs::write(&weights_path, &tampered).unwrap();
        assert!(LoraAdapterArtifact::load(&dir, Some(&base_sha)).is_err());

        let _ = fs::remove_dir_all(&dir);
    }
}
