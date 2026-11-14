//! Shared codec state and preflight-checked FP32 runtime weight loading.
//! Forward execution is implemented separately from artifact validation.
mod blocks;
mod codec;
pub use blocks::{NoTrace, TraceSink, codebook_distances};
pub use codec::{Decoded, Encoded};

use ncm_core::{CodecConfig, Error, Result, WeightBackend};
use ncm_io::Artifact;
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, de::IgnoredAny};
use std::{collections::BTreeMap, sync::OnceLock};

// IEEE-754 infinities and NaNs are exactly the values with an all-one exponent.
// Check fixed-size groups before branching so finite weight buffers can be
// scanned efficiently without allocating or requiring aligned input.
fn has_nonfinite_f32_le(data: &[u8]) -> bool {
    debug_assert_eq!(data.len() % 4, 0);
    let (groups, tail) = data.as_chunks::<64>();
    for group in groups {
        let mut bad = false;
        for bytes in group.as_chunks::<4>().0 {
            bad |= u32::from_le_bytes(*bytes) & 0x7f80_0000 == 0x7f80_0000;
        }
        if bad {
            return true;
        }
    }
    tail.as_chunks::<4>()
        .0
        .iter()
        .any(|bytes| u32::from_le_bytes(*bytes) & 0x7f80_0000 == 0x7f80_0000)
}

/// Immutable parameters plus backend and lazily prepared parameter-only norms.
/// Activations and inference state belong to each request.
pub struct Codec<B: WeightBackend> {
    pub(crate) backend: B,
    pub(crate) config: CodecConfig,
    pub(crate) weights: BTreeMap<String, B::Tensor>,
    // Model-owned, bounded by validated levels; never shared across models/devices.
    pub(crate) codebook_norms: Vec<OnceLock<B::Tensor>>,
}
impl<B: WeightBackend> Codec<B> {
    /// Integrity checks belong to Artifact; this adds codec semantic checks, not
    /// manifest authentication. Generator/evaluation metadata is not executed.
    pub fn from_artifact(backend: B, artifact: &Artifact) -> Result<Self> {
        let bytes = artifact
            .file("config.json")
            .map_err(|e| Error::Weights(e.to_string()))?;
        let config = artifact_config(bytes)?;
        let bytes = artifact
            .file("codec.safetensors")
            .map_err(|e| Error::Weights(e.to_string()))?;
        Self::from_safetensors(backend, config, bytes)
    }

    /// Supports explicit synthetic fixtures as well as the checked artifact path.
    /// Complete host preflight precedes the FIRST backend allocation. Native
    /// allocation errors are propagated, never retried on a different device.
    pub fn from_safetensors(backend: B, config: CodecConfig, bytes: &[u8]) -> Result<Self> {
        if bytes.len() > 256 * 1024 * 1024 + 1024 * 1024 {
            return Err(Error::Weights(
                "safetensors file exceeds codec limit".into(),
            ));
        }
        let tensors = SafeTensors::deserialize(bytes).map_err(|e| Error::Weights(e.to_string()))?;
        let expected = config.weight_shapes();
        if tensors.len() != expected.len() {
            return Err(Error::Weights(
                "runtime tensor inventory size differs".into(),
            ));
        }
        for (name, shape) in &expected {
            let t = tensors
                .tensor(name)
                .map_err(|e| Error::Weights(e.to_string()))?;
            if t.dtype() != Dtype::F32 || t.shape() != shape {
                return Err(Error::Weights(format!("wrong dtype/shape for {name}")));
            }
            if t.data().len() != shape.iter().product::<usize>() * 4
                || has_nonfinite_f32_le(t.data())
            {
                return Err(Error::Weights(format!("invalid/nonfinite data for {name}")));
            }
        }
        let mut weights = BTreeMap::new();
        for (name, shape) in expected {
            let t = tensors
                .tensor(&name)
                .map_err(|e| Error::Weights(e.to_string()))?;
            // Decode little-endian bytes without alignment casts/unsafe. Only one
            // temporary tensor is retained, not a second full host weight copy.
            let values: Vec<f32> = t
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            weights.insert(name, backend.upload_f32(&values, &shape)?);
        }
        let codebook_norms = (0..config.levels().len())
            .map(|_| OnceLock::new())
            .collect();
        Ok(Self {
            backend,
            config,
            weights,
            codebook_norms,
        })
    }
    pub fn config(&self) -> &CodecConfig {
        &self.config
    }
    pub fn backend(&self) -> &B {
        &self.backend
    }
    pub fn weights(&self) -> &BTreeMap<String, B::Tensor> {
        &self.weights
    }
}

fn artifact_config(bytes: &[u8]) -> Result<CodecConfig> {
    if bytes.len() > 1024 * 1024 {
        return Err(Error::Config("config exceeds 1 MiB".into()));
    }
    let envelope: Envelope = serde_json::from_slice(bytes)?;
    if envelope.task != "sudoku-easy"
        || envelope.tokenizer.pre_tokenizer.name != "sudoku"
        || envelope.tokenizer.pre_tokenizer.tokenizer != "sudoku"
    {
        return Err(Error::Config(
            "unsupported codec artifact task/tokenizer".into(),
        ));
    }
    CodecConfig::validate(envelope.tokenizer.vqvae)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    task: String,
    tokenizer: Tokenizer,
    // Explicitly not codec runtime fields. Paths are never opened/evaluated here.
    #[serde(rename = "checkpoints")]
    _checkpoints: IgnoredAny,
    #[serde(rename = "data")]
    _data: IgnoredAny,
    #[serde(rename = "eval")]
    _eval: IgnoredAny,
    #[serde(rename = "generation")]
    _generation: IgnoredAny,
    #[serde(rename = "model")]
    _model: IgnoredAny,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Tokenizer {
    pre_tokenizer: PreTokenizer,
    vqvae: ncm_core::RawCodecConfig,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PreTokenizer {
    name: String,
    tokenizer: String,
}

#[cfg(test)]
mod tests {
    use super::{artifact_config, has_nonfinite_f32_le};
    use serde_json::{Value, json};

    #[test]
    fn finite_scan_covers_groups_tails_and_ieee_edge_cases() {
        let finite = [0, 0x8000_0000, 1, 0x007f_ffff, 0x7f7f_ffff, 0xff7f_ffff];
        let nonfinite = [
            0x7f80_0000,
            0xff80_0000,
            0x7f80_0001,
            0x7fc0_0000,
            0xffff_ffff,
        ];
        for count in 0..=35 {
            let words: Vec<u32> = (0..count).map(|i| finite[i % finite.len()]).collect();
            let mut data = vec![0]; // Deliberately unaligned buffer start.
            data.extend(words.iter().flat_map(|word| word.to_le_bytes()));
            assert!(!has_nonfinite_f32_le(&data[1..]));
            for index in 0..count {
                for bits in nonfinite {
                    let mut changed = data.clone();
                    changed[1 + 4 * index..1 + 4 * (index + 1)]
                        .copy_from_slice(&u32::to_le_bytes(bits));
                    assert!(has_nonfinite_f32_le(&changed[1..]));
                }
            }
        }
    }

    #[test]
    fn artifact_envelope_is_bounded_and_strict() {
        let fixture: Value =
            serde_json::from_str(include_str!("../../../tests/fixtures/codec-v1.json")).unwrap();
        let valid = json!({"task": "sudoku-easy", "tokenizer": {
            "pre_tokenizer": {"name":"sudoku", "tokenizer":"sudoku"}, "vqvae": fixture["config"]},
            "checkpoints": {}, "data": {}, "eval": {}, "generation": {}, "model": {}});
        assert!(artifact_config(&serde_json::to_vec(&valid).unwrap()).is_ok());
        for (pointer, replacement) in [
            ("/task", json!("text")),
            ("/tokenizer/pre_tokenizer/tokenizer", json!("gpt2")),
            ("/tokenizer/vqvae/compression_factor", json!(2)),
        ] {
            let mut bad = valid.clone();
            *bad.pointer_mut(pointer).unwrap() = replacement;
            assert!(artifact_config(&serde_json::to_vec(&bad).unwrap()).is_err());
        }
        let mut bad = valid;
        bad["tokenizer"]["unknown_option"] = json!(true);
        assert!(artifact_config(&serde_json::to_vec(&bad).unwrap()).is_err());
        assert!(artifact_config(&vec![b' '; 1024 * 1024 + 1]).is_err());
    }
}
