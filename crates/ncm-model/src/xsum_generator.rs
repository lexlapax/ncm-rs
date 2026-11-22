//! Host-only validation for the pinned XSum generator and token allowance.
use crate::{artifact_config, generator::GeneratorConfig, has_nonfinite_f32_le};
use ncm_core::{Error, Result};
use ncm_io::Artifact;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde::Deserialize;

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ContextEncoder {
    enabled: bool,
    n_layer: usize,
    vocab_size: usize,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct XsumModelConfig {
    attn_pattern: String,
    bias: bool,
    context_length: usize,
    ctx_encoder: ContextEncoder,
    dropout: f64,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
    predict_level0: bool,
    prefix_len: usize,
    prefix_pad_token_id: usize,
    query_level0: bool,
    raw_token_prefix: bool,
    rope_base: usize,
    tie_heads_learn_scale: bool,
    tie_heads_to_codebook: bool,
    use_flex_attention: bool,
    use_level_encoding: bool,
    use_positional_encoding: bool,
    use_qk_norm: bool,
    use_rope: bool,
    vocab_size: Vec<usize>,
}

#[derive(Deserialize)]
struct Envelope {
    model: XsumModelConfig,
}

/// Fully host-checked generator tensor metadata and retained FP32 allowance.
/// This does not construct a generator or perform any device upload.
pub struct XsumGeneratorInspection {
    pub tensor_count: usize,
    pub token_allowance: Vec<f32>,
    pub rope_tables: Option<(Vec<f32>, Vec<f32>)>,
}

impl XsumGeneratorInspection {
    pub fn from_artifact(artifact: &Artifact) -> Result<Self> {
        let config_bytes = artifact
            .file("config.json")
            .map_err(|e| Error::Weights(e.to_string()))?;
        crate::validate_artifact_profile(artifact, config_bytes)?;
        let codec = artifact_config(config_bytes)?;
        let config: Envelope = serde_json::from_slice(config_bytes)?;
        let expected: XsumModelConfig = serde_json::from_str(include_str!(
            "../../../tests/fixtures/xsum-model-config-v1.json"
        ))?;
        let profile = artifact.manifest().profile.as_str();
        if config.model != expected || !matches!(profile, "xsum-inference-v1" | "xsum-inference-v2")
        {
            return Err(Error::Config(
                "unsupported XSum generator configuration".into(),
            ));
        }
        let mut shapes = GeneratorConfig {
            attn_pattern: "block_diagonal".into(),
            bias: true,
            context_length: 560,
            dropout: 0.3,
            n_embd: 768,
            n_head: 12,
            n_layer: 12,
            prefix_len: 512,
            query_level0: true,
            raw_token_prefix: true,
            rope_base: 100000,
            tie_heads_to_codebook: false,
            use_level_encoding: true,
            use_positional_encoding: true,
            use_qk_norm: true,
            use_rope: true,
            vocab_size: vec![4096; 9],
        }
        .shapes(&codec);
        for i in 0..9 {
            shapes.remove(&format!("lm_heads.{i}.weight"));
            shapes.remove(&format!("lm_heads.{i}.bias"));
            shapes.insert(format!("lm_heads.{i}.proj.weight"), vec![384, 768]);
            shapes.insert(format!("lm_heads.{i}.log_inv_T"), vec![]);
        }
        if artifact.is_streaming_bundle("generator.safetensors") {
            if !artifact.bundle_f32_weights_verified_finite("generator.safetensors") {
                return Err(Error::Weights(
                    "streaming XSum generator missing finite-value preflight".into(),
                ));
            }
            let entries = artifact
                .bundle_tensors("generator.safetensors")
                .map_err(|e| Error::Weights(e.to_string()))?;
            if entries.len() != shapes.len() + 2 {
                return Err(Error::Weights(
                    "XSum generator tensor inventory differs".into(),
                ));
            }
            let by_name: std::collections::BTreeMap<_, _> = entries
                .iter()
                .map(|entry| (entry.name.as_str(), entry))
                .collect();
            for (name, shape) in &shapes {
                let entry = by_name.get(name.as_str()).ok_or_else(|| {
                    Error::Weights(format!("invalid XSum generator tensor {name}"))
                })?;
                if !matches!(entry.dtype, ncm_io::TensorDtype::F32) || entry.shape != *shape {
                    return Err(Error::Weights(format!(
                        "invalid XSum generator tensor {name}"
                    )));
                }
            }
            for i in 0..9 {
                let raw_bytes = artifact
                    .read_bundle_tensor("generator.safetensors", &format!("lm_heads.{i}.log_inv_T"))
                    .map_err(|e| Error::Weights(e.to_string()))?;
                let raw: [u8; 4] = raw_bytes
                    .as_slice()
                    .try_into()
                    .map_err(|_| Error::Weights("invalid head scale".into()))?;
                let temperature = f32::from_le_bytes(raw).exp();
                if !temperature.is_finite() || temperature <= 0.0 {
                    return Err(Error::Weights("invalid XSum head temperature".into()));
                }
            }
            let level_entry = by_name
                .get("lvl_1L")
                .ok_or_else(|| Error::Weights("missing generator level buffer".into()))?;
            let level_bytes = artifact
                .read_bundle_tensor("generator.safetensors", "lvl_1L")
                .map_err(|e| Error::Weights(e.to_string()))?;
            validate_levels_raw(
                level_entry.dtype,
                &level_entry.shape,
                &level_bytes,
                codec.levels(),
            )?;
            let mask_entry = by_name
                .get("attn_mask_for_masking")
                .ok_or_else(|| Error::Weights("missing generator attention mask".into()))?;
            let mask_bytes = artifact
                .read_bundle_tensor("generator.safetensors", "attn_mask_for_masking")
                .map_err(|e| Error::Weights(e.to_string()))?;
            validate_mask_raw(
                mask_entry.dtype,
                &mask_entry.shape,
                &mask_bytes,
                codec.levels(),
                config.model.prefix_len,
            )?;
        } else {
            let bytes = artifact
                .file("generator.safetensors")
                .map_err(|e| Error::Weights(e.to_string()))?;
            let tensors =
                SafeTensors::deserialize(bytes).map_err(|e| Error::Weights(e.to_string()))?;
            if tensors.len() != shapes.len() + 2 {
                return Err(Error::Weights(
                    "XSum generator tensor inventory differs".into(),
                ));
            }
            for (name, shape) in &shapes {
                let value = tensors
                    .tensor(name)
                    .map_err(|e| Error::Weights(e.to_string()))?;
                if value.dtype() != Dtype::F32
                    || value.shape() != shape
                    || has_nonfinite_f32_le(value.data())
                {
                    return Err(Error::Weights(format!(
                        "invalid XSum generator tensor {name}"
                    )));
                }
            }
            for i in 0..9 {
                let value = tensors
                    .tensor(&format!("lm_heads.{i}.log_inv_T"))
                    .map_err(|e| Error::Weights(e.to_string()))?;
                let raw: [u8; 4] = value
                    .data()
                    .try_into()
                    .map_err(|_| Error::Weights("invalid head scale".into()))?;
                let temperature = f32::from_le_bytes(raw).exp();
                if !temperature.is_finite() || temperature <= 0.0 {
                    return Err(Error::Weights("invalid XSum head temperature".into()));
                }
            }
            let level = tensors
                .tensor("lvl_1L")
                .map_err(|e| Error::Weights(e.to_string()))?;
            validate_levels(&level, codec.levels())?;
            let mask = tensors
                .tensor("attn_mask_for_masking")
                .map_err(|e| Error::Weights(e.to_string()))?;
            validate_mask(&mask, codec.levels(), config.model.prefix_len)?;
        }
        let bytes = artifact
            .file("inference.safetensors")
            .map_err(|e| Error::Weights(e.to_string()))?;
        let inference =
            SafeTensors::deserialize(bytes).map_err(|e| Error::Weights(e.to_string()))?;
        if inference.len() != if profile == "xsum-inference-v2" { 3 } else { 1 } {
            return Err(Error::Weights("invalid XSum inference inventory".into()));
        }
        let rope_tables = if profile == "xsum-inference-v2" {
            let mut tables = Vec::with_capacity(2);
            for (name, expected) in [
                (
                    "rope_cos",
                    "f1376ecd82c6f574ade39abbdc95db403338b466306c02062a260079116c7ed6",
                ),
                (
                    "rope_sin",
                    "81716ee36cb0d78e78c093171a1751124c57467459608f5d810b3a163866e523",
                ),
            ] {
                let tensor = inference
                    .tensor(name)
                    .map_err(|e| Error::Weights(e.to_string()))?;
                tables.push(decode_rope_tensor(&tensor, expected)?);
            }
            let mut tables = tables.into_iter();
            Some((
                tables.next().expect("two tables"),
                tables.next().expect("two tables"),
            ))
        } else {
            None
        };
        let allowance = inference
            .tensor("token_unigram")
            .map_err(|e| Error::Weights(e.to_string()))?;
        if allowance.dtype() != Dtype::F32 || allowance.shape() != [50304] {
            return Err(Error::Weights("invalid XSum allowance shape".into()));
        }
        let token_allowance: Vec<f32> = allowance
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| f32::from_le_bytes(*x))
            .collect();
        let sum: f64 = token_allowance.iter().map(|&p| f64::from(p)).sum();
        if token_allowance.iter().any(|p| !p.is_finite() || *p < 0.0) || (sum - 1.0).abs() > 1e-6 {
            return Err(Error::Weights("invalid XSum token allowance".into()));
        }
        Ok(Self {
            tensor_count: shapes.len(),
            token_allowance,
            rope_tables,
        })
    }
}

fn decode_rope_tensor(tensor: &TensorView<'_>, expected: &str) -> Result<Vec<f32>> {
    if tensor.dtype() != Dtype::F32
        || tensor.shape() != [512, 32]
        || ncm_io::sha256(tensor.data()) != expected
    {
        return Err(Error::Weights("invalid XSum rotary reference".into()));
    }
    Ok(tensor
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect())
}

pub(crate) fn validate_levels(level: &TensorView<'_>, lengths: &[usize]) -> Result<()> {
    let dtype = match level.dtype() {
        Dtype::I64 => ncm_io::TensorDtype::I64,
        Dtype::F32 => ncm_io::TensorDtype::F32,
        _ => return Err(Error::Weights("incorrect generator level buffer".into())),
    };
    validate_levels_raw(dtype, level.shape(), level.data(), lengths)
}

pub(crate) fn validate_levels_raw(
    dtype: ncm_io::TensorDtype,
    shape: &[usize],
    data: &[u8],
    lengths: &[usize],
) -> Result<()> {
    let total: usize = lengths.iter().sum();
    let expected = lengths
        .iter()
        .enumerate()
        .flat_map(|(i, &len)| std::iter::repeat_n(i as i64, len));
    if !matches!(dtype, ncm_io::TensorDtype::I64)
        || shape != [1, total]
        || !data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|x| i64::from_le_bytes(*x))
            .eq(expected)
    {
        return Err(Error::Weights("incorrect generator level buffer".into()));
    }
    Ok(())
}

pub(crate) fn validate_mask(mask: &TensorView<'_>, lengths: &[usize], prefix: usize) -> Result<()> {
    let dtype = match mask.dtype() {
        Dtype::F32 => ncm_io::TensorDtype::F32,
        Dtype::I64 => ncm_io::TensorDtype::I64,
        _ => return Err(Error::Weights("invalid generator attention mask".into())),
    };
    validate_mask_raw(dtype, mask.shape(), mask.data(), lengths, prefix)
}

pub(crate) fn validate_mask_raw(
    dtype: ncm_io::TensorDtype,
    shape: &[usize],
    data: &[u8],
    lengths: &[usize],
    prefix: usize,
) -> Result<()> {
    let seq = prefix + lengths.iter().sum::<usize>();
    if !matches!(dtype, ncm_io::TensorDtype::F32) || shape != [1, 1, seq, seq] {
        return Err(Error::Weights("invalid generator attention mask".into()));
    }
    let mut levels = vec![usize::MAX; prefix];
    for (i, &len) in lengths.iter().enumerate() {
        levels.extend(std::iter::repeat_n(i, len));
    }
    for (i, raw) in data.as_chunks::<4>().0.iter().enumerate() {
        let (row, col) = (i / seq, i % seq);
        let allowed = col < prefix || (row >= prefix && levels[row] == levels[col]);
        let expected = if allowed {
            0f32.to_bits()
        } else {
            f32::NEG_INFINITY.to_bits()
        };
        if u32::from_le_bytes(*raw) != expected {
            return Err(Error::Weights("incorrect generator attention mask".into()));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{XsumModelConfig, decode_rope_tensor, validate_levels, validate_mask};
    use safetensors::{Dtype, tensor::TensorView};

    #[test]
    fn rotary_reference_rejects_shape_and_hash_changes() {
        let bytes = vec![0u8; 512 * 32 * 4];
        let correct_shape = TensorView::new(Dtype::F32, vec![512, 32], &bytes).unwrap();
        assert!(decode_rope_tensor(&correct_shape, "not-the-pinned-hash").is_err());
        let wrong_shape = TensorView::new(Dtype::F32, vec![256, 64], &bytes).unwrap();
        assert!(decode_rope_tensor(&wrong_shape, &ncm_io::sha256(&bytes)).is_err());
    }

    #[test]
    fn level_and_attention_buffers_reject_corruption() {
        let levels: Vec<u8> = [0_i64, 0, 1]
            .into_iter()
            .flat_map(i64::to_le_bytes)
            .collect();
        let level = TensorView::new(Dtype::I64, vec![1, 3], &levels).unwrap();
        validate_levels(&level, &[2, 1]).unwrap();
        assert!(validate_levels(&level, &[1, 2]).is_err());
        let sequence = [usize::MAX, usize::MAX, 0, 0, 1];
        let mut bytes = Vec::new();
        for row in 0..5 {
            for col in 0..5 {
                let allowed = col < 2 || (row >= 2 && sequence[row] == sequence[col]);
                bytes.extend(if allowed { 0f32 } else { f32::NEG_INFINITY }.to_le_bytes());
            }
        }
        let mask = TensorView::new(Dtype::F32, vec![1, 1, 5, 5], &bytes).unwrap();
        validate_mask(&mask, &[2, 1], 2).unwrap();
        bytes[4 * 2..4 * 3].copy_from_slice(&0f32.to_le_bytes());
        let bad = TensorView::new(Dtype::F32, vec![1, 1, 5, 5], &bytes).unwrap();
        assert!(validate_mask(&bad, &[2, 1], 2).is_err());
    }

    #[test]
    fn model_config_is_strict_and_detects_semantic_changes() {
        let raw = include_str!("../../../tests/fixtures/xsum-model-config-v1.json");
        let expected: XsumModelConfig = serde_json::from_str(raw).unwrap();
        let mut changed = raw.replace(
            "\"prefix_pad_token_id\": 50257",
            "\"prefix_pad_token_id\": 50256",
        );
        let other: XsumModelConfig = serde_json::from_str(&changed).unwrap();
        assert_ne!(expected, other);
        changed = raw.replacen("\"bias\": true", "\"bias\": true, \"bias\": true", 1);
        assert!(serde_json::from_str::<XsumModelConfig>(&changed).is_err());
        changed = raw.replacen("\"bias\": true", "\"bias\": true, \"unknown\": false", 1);
        assert!(serde_json::from_str::<XsumModelConfig>(&changed).is_err());
    }
}
