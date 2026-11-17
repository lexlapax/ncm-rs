//! Shared codec state and preflight-checked FP32 runtime weight loading.
//! Forward execution is implemented separately from artifact validation.
mod blocks;
mod codec;
mod generator;
mod xsum_decode;
mod xsum_generator;
pub use blocks::{NoTrace, TraceSink, codebook_distances};
pub use codec::{Decoded, Encoded};
pub use generator::{
    Generated, Generator, GeneratorConfig, GeneratorWeights, TinyStoriesGeneratorInspection,
    TinyStoriesTeacher, XsumGenerated, XsumTeacher, XsumTeacherOutput,
};
pub use xsum_decode::select_xsum_tokens_cpu;
pub use xsum_generator::XsumGeneratorInspection;

use ncm_core::{CodecConfig, Error, Result, WeightBackend};
use ncm_io::Artifact;
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, de::IgnoredAny};
use std::{borrow::Cow, collections::BTreeMap, sync::OnceLock};

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

/// Borrow aligned little-endian F32 payloads when possible; otherwise decode an
/// owned copy. Shape, dtype and finite-value checks remain the caller's preflight.
fn upload_values(data: &[u8]) -> Result<Cow<'_, [f32]>> {
    if !data.len().is_multiple_of(4) {
        return Err(Error::Weights("invalid F32 payload length".into()));
    }
    if cfg!(target_endian = "little")
        && let Ok(values) = bytemuck::try_cast_slice::<u8, f32>(data)
    {
        return Ok(Cow::Borrowed(values));
    }
    Ok(Cow::Owned(
        data.as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
    ))
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
        validate_artifact_profile(artifact, bytes)?;
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
            // The checked borrowed view avoids a temporary copy when aligned.
            let values = upload_values(t.data())?;
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
    match envelope.task.as_str() {
        "sudoku-easy" | "sudoku-medium" | "sudoku-hard" => {
            let pre: SudokuPreTokenizer = serde_json::from_value(envelope.tokenizer.pre_tokenizer)?;
            if pre.name != "sudoku" || pre.tokenizer != "sudoku" {
                return Err(Error::Config("unsupported Sudoku tokenizer".into()));
            }
            // Preserve the original Sudoku wire schema before using the shared
            // profile-capable deserializer. The typed envelope already rejects
            // duplicate fields in the original input.
            let document: serde_json::Value = serde_json::from_slice(bytes)?;
            let codec = document
                .pointer("/tokenizer/vqvae")
                .ok_or_else(|| Error::Config("missing Sudoku codec config".into()))?;
            CodecConfig::from_json(&serde_json::to_vec(codec)?)
        }
        "tinystories" => {
            let pre: TinyStoriesPreTokenizer =
                serde_json::from_value(envelope.tokenizer.pre_tokenizer)?;
            if pre.name != "bpe"
                || pre.tokenizer != "bpe"
                || pre.vocab_file_path != "bpe_vocab_8192.pickle"
                || pre.merges_file_path != "bpe_merges_8192.pickle"
                || pre.special_tokens.eos_token != "<|endoftext|>"
                || pre.special_tokens.pad_token != "<|pad|>"
            {
                return Err(Error::Config("unsupported TinyStories tokenizer".into()));
            }
            CodecConfig::validate_tinystories(envelope.tokenizer.vqvae)
        }
        "xsum" => {
            let pre: XsumPreTokenizer = serde_json::from_value(envelope.tokenizer.pre_tokenizer)?;
            if pre.name != "hf"
                || pre.tokenizer != "hf"
                || pre.model_id != "gpt2"
                || pre.special_tokens.eos_token != "<|endoftext|>"
                || pre.special_tokens.pad_token != "<|pad|>"
            {
                return Err(Error::Config("unsupported XSum tokenizer".into()));
            }
            CodecConfig::validate_xsum(envelope.tokenizer.vqvae)
        }
        _ => Err(Error::Config(
            "unsupported codec artifact task/tokenizer".into(),
        )),
    }
}

fn validate_artifact_profile(artifact: &Artifact, bytes: &[u8]) -> Result<()> {
    let envelope: Envelope = serde_json::from_slice(bytes)?;
    let expected = match envelope.task.as_str() {
        "tinystories" => "tinystories-inference-v1".to_owned(),
        "xsum" if artifact.manifest().profile == "xsum-inference-v2" => {
            "xsum-inference-v2".to_owned()
        }
        "xsum" => "xsum-inference-v1".to_owned(),
        _ => format!("{}-inspection-v1", envelope.task),
    };
    if artifact.manifest().profile != expected {
        return Err(Error::Config("artifact profile does not match task".into()));
    }
    Ok(())
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
    pre_tokenizer: serde_json::Value,
    vqvae: ncm_core::RawCodecConfig,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SudokuPreTokenizer {
    name: String,
    tokenizer: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TinyStoriesPreTokenizer {
    name: String,
    tokenizer: String,
    vocab_file_path: String,
    merges_file_path: String,
    special_tokens: TinyStoriesSpecialTokens,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct XsumPreTokenizer {
    name: String,
    tokenizer: String,
    model_id: String,
    special_tokens: TinyStoriesSpecialTokens,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TinyStoriesSpecialTokens {
    eos_token: String,
    pad_token: String,
}

#[cfg(test)]
mod tests {
    use super::{artifact_config, has_nonfinite_f32_le, upload_values};
    use serde_json::{Value, json};
    use std::borrow::Cow;

    #[test]
    fn upload_values_borrows_aligned_and_preserves_unaligned_bits() {
        let words = [0x8000_0000u32, 0x3f80_0000, 1];
        let mut aligned = [0u32; 3];
        bytemuck::cast_slice_mut::<u32, u8>(&mut aligned).copy_from_slice(
            &words
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let bytes: &[u8] = bytemuck::cast_slice(&aligned);
        let values = upload_values(bytes).unwrap();
        #[cfg(target_endian = "little")]
        assert!(matches!(values, Cow::Borrowed(_)));
        assert_eq!(
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            words
        );

        let mut storage = [0u32; 3];
        let raw: &mut [u8] = bytemuck::cast_slice_mut(&mut storage);
        let expected = [0x8000_0000u32, 0x3f80_0000];
        raw[1..9].copy_from_slice(
            &expected
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let unaligned = upload_values(&raw[1..9]).unwrap();
        assert!(matches!(unaligned, Cow::Owned(_)));
        assert_eq!(
            unaligned.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected
        );
        assert!(upload_values(&raw[..3]).is_err());
    }

    #[test]
    fn xsum_envelope_accepts_only_its_pinned_tokenizer_and_codec_profile() {
        let codec: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/xsum-codec-config-v1.json"
        ))
        .unwrap();
        let mut envelope = json!({"task":"xsum", "tokenizer": {
            "pre_tokenizer": {"name":"hf", "tokenizer":"hf", "model_id":"gpt2",
                "special_tokens":{"eos_token":"<|endoftext|>", "pad_token":"<|pad|>"}},
            "vqvae":codec}, "checkpoints":{}, "data":{}, "eval":{}, "generation":{}, "model":{}});
        let parsed = artifact_config(&serde_json::to_vec(&envelope).unwrap()).unwrap();
        assert_eq!(parsed.levels()[0], 4);
        assert_eq!(parsed.strides(), [4, 3]);
        envelope["tokenizer"]["pre_tokenizer"]["model_id"] = "other".into();
        assert!(artifact_config(&serde_json::to_vec(&envelope).unwrap()).is_err());
    }

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
        for task in ["sudoku-medium", "sudoku-hard"] {
            let mut profile = valid.clone();
            profile["task"] = json!(task);
            assert!(artifact_config(&serde_json::to_vec(&profile).unwrap()).is_ok());
        }
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
