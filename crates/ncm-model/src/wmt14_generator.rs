//! Host-only semantic inspection of the pinned WMT14 generator.
use crate::{generator::GeneratorConfig, has_nonfinite_f32_le, xsum_generator};
use ncm_core::{CodecConfig, Error, RawCodecConfig, Result};
use ncm_io::Artifact;
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, de::IgnoredAny};
use std::collections::BTreeMap;

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ContextEncoder {
    enabled: bool,
    n_layer: usize,
    vocab_size: usize,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Wmt14ModelConfig {
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
    raw_token_prefix: bool,
    rope_base: usize,
    untied_head_bias: bool,
    use_flex_attention: bool,
    use_level_encoding: bool,
    use_positional_encoding: bool,
    use_qk_norm: bool,
    use_rope: bool,
    vocab_size: Vec<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    task: String,
    model: Wmt14ModelConfig,
    tokenizer: Tokenizer,
    #[serde(rename = "checkpoints")]
    _checkpoints: IgnoredAny,
    #[serde(rename = "data")]
    _data: IgnoredAny,
    #[serde(rename = "generation")]
    _generation: IgnoredAny,
    #[serde(rename = "eval")]
    _eval: IgnoredAny,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Tokenizer {
    pre_tokenizer: PreTokenizer,
    vqvae: RawCodecConfig,
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct PreTokenizer {
    name: String,
    tokenizer: String,
    model_id: String,
    special_tokens: SpecialTokens,
}
#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct SpecialTokens {
    eos_token: String,
    pad_token: String,
}

fn validate_model_config(model: &Wmt14ModelConfig) -> Result<()> {
    let expected: Wmt14ModelConfig =
        serde_json::from_str(include_str!("../tests/fixtures/wmt14-model-config-v1.json"))?;
    if model != &expected {
        return Err(Error::Config(
            "unsupported WMT14 generator configuration".into(),
        ));
    }
    Ok(())
}

fn validate_pre_tokenizer(tokenizer: &PreTokenizer) -> Result<()> {
    let expected = PreTokenizer {
        name: "hf".into(),
        tokenizer: "hf".into(),
        model_id: "Qwen/Qwen3-0.6B".into(),
        special_tokens: SpecialTokens {
            eos_token: "<|im_end|>".into(),
            pad_token: "<|endoftext|>".into(),
        },
    };
    if tokenizer != &expected {
        return Err(Error::Config(
            "unsupported WMT14 tokenizer configuration".into(),
        ));
    }
    Ok(())
}

fn expected_shapes(codec: &CodecConfig) -> BTreeMap<String, Vec<usize>> {
    let mut shapes = GeneratorConfig {
        attn_pattern: "block_diagonal".into(),
        bias: true,
        context_length: 160,
        dropout: 0.1,
        n_embd: 768,
        n_head: 12,
        n_layer: 12,
        prefix_len: 96,
        query_level0: true,
        raw_token_prefix: true,
        rope_base: 100000,
        tie_heads_to_codebook: false,
        use_level_encoding: true,
        use_positional_encoding: true,
        use_qk_norm: true,
        use_rope: true,
        vocab_size: vec![4096; 10],
    }
    .shapes(codec);
    shapes.remove("l0_queries");
    shapes.insert("transformer.wle.weight".into(), vec![10, 768]);
    shapes.insert("lm_head_l0.weight".into(), vec![4096, 768]);
    for i in 0..10 {
        shapes.remove(&format!("lm_heads.{i}.bias"));
    }
    shapes
}

/// Checked WMT14 generator metadata and tensor values. This type does not
/// allocate a backend or enable WMT14 model execution.
pub struct Wmt14GeneratorInspection {
    pub tensor_count: usize,
}

impl Wmt14GeneratorInspection {
    pub fn from_artifact(artifact: &Artifact) -> Result<Self> {
        if artifact.manifest().profile != "wmt14-inference-v1" {
            return Err(Error::Config("unsupported WMT14 artifact profile".into()));
        }
        let config = artifact
            .file("config.json")
            .map_err(|error| Error::Weights(error.to_string()))?;
        let envelope: Envelope = serde_json::from_slice(config)?;
        if envelope.task != "wmt14" {
            return Err(Error::Config("unsupported WMT14 task".into()));
        }
        validate_model_config(&envelope.model)?;
        validate_pre_tokenizer(&envelope.tokenizer.pre_tokenizer)?;
        let codec = CodecConfig::validate_wmt14(envelope.tokenizer.vqvae)?;
        let shapes = expected_shapes(&codec);
        let data = artifact
            .file("generator.safetensors")
            .map_err(|error| Error::Weights(error.to_string()))?;
        let tensors =
            SafeTensors::deserialize(data).map_err(|error| Error::Weights(error.to_string()))?;
        if tensors.len() != shapes.len() + 2 {
            return Err(Error::Weights(
                "WMT14 generator tensor inventory differs".into(),
            ));
        }
        for (name, shape) in &shapes {
            let tensor = tensors
                .tensor(name)
                .map_err(|error| Error::Weights(error.to_string()))?;
            if tensor.dtype() != Dtype::F32
                || tensor.shape() != shape
                || has_nonfinite_f32_le(tensor.data())
            {
                return Err(Error::Weights(format!(
                    "invalid WMT14 generator tensor {name}"
                )));
            }
        }
        let lengths = &codec.levels()[1..];
        let levels = tensors
            .tensor("lvl_1L")
            .map_err(|error| Error::Weights(error.to_string()))?;
        xsum_generator::validate_levels(&levels, lengths)?;
        let mask = tensors
            .tensor("attn_mask_for_masking")
            .map_err(|error| Error::Weights(error.to_string()))?;
        xsum_generator::validate_mask(&mask, lengths, envelope.model.prefix_len)?;
        Ok(Self {
            tensor_count: tensors.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PreTokenizer, Wmt14ModelConfig, expected_shapes, validate_model_config,
        validate_pre_tokenizer,
    };
    use ncm_core::{CodecConfig, RawCodecConfig};

    #[test]
    fn wmt14_generator_has_explicit_untied_head_inventory() {
        let mut model: Wmt14ModelConfig =
            serde_json::from_str(include_str!("../tests/fixtures/wmt14-model-config-v1.json"))
                .unwrap();
        assert!(model.predict_level0);
        assert_eq!(model.prefix_len, 96);
        assert_eq!(model.prefix_pad_token_id, 151643);
        assert!(validate_model_config(&model).is_ok());
        model.predict_level0 = false;
        assert!(validate_model_config(&model).is_err());
        let raw: RawCodecConfig = serde_json::from_str(include_str!(
            "../../ncm-core/tests/fixtures/wmt14-codec-config-v1.json"
        ))
        .unwrap();
        let codec = CodecConfig::validate_wmt14(raw).unwrap();
        let shapes = expected_shapes(&codec);
        assert_eq!(shapes.len() + 2, 186);
        assert_eq!(shapes["lm_head_l0.weight"], [4096, 768]);
        assert_eq!(shapes["lm_heads.9.weight"], [4096, 768]);
        assert_eq!(shapes["transformer.wle.weight"], [10, 768]);
        assert!(!shapes.contains_key("lm_heads.0.bias"));
        assert!(!shapes.contains_key("l0_queries"));
    }

    #[test]
    fn wmt14_tokenizer_configuration_rejects_changes() {
        let mut tokenizer: PreTokenizer = serde_json::from_str(
            r#"{"name":"hf","tokenizer":"hf","model_id":"Qwen/Qwen3-0.6B","special_tokens":{"eos_token":"<|im_end|>","pad_token":"<|endoftext|>"}}"#,
        ).unwrap();
        assert!(validate_pre_tokenizer(&tokenizer).is_ok());
        tokenizer.special_tokens.eos_token = "<|endoftext|>".into();
        assert!(validate_pre_tokenizer(&tokenizer).is_err());
    }
}
