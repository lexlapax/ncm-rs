//! Host-only preflight of pinned LM1B-small generator preparation bytes.
use crate::{has_nonfinite_f32_le, xsum_generator};
use ncm_core::{CodecConfig, Error, RawCodecConfig, Result};
use ncm_io::Artifact;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use std::collections::BTreeMap;

const CONFIG_SHA: &str = "1bc5e0c10050c8221ec72c277d8510a9a541117db5705cb12173f12b733a1475";
const LEVELS: [usize; 12] = [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128];

pub(crate) fn codec_config(artifact: &Artifact) -> Result<CodecConfig> {
    if artifact.manifest().profile != "lm1b-small-inference-v1" {
        return Err(Error::Config("unsupported LM1B artifact profile".into()));
    }
    let config = artifact
        .file("config.json")
        .map_err(|e| Error::Weights(e.to_string()))?;
    if ncm_io::sha256(config) != CONFIG_SHA {
        return Err(Error::Config("unexpected pinned LM1B config".into()));
    }
    // This exact pinned configuration includes training-only fields outside the
    // shared inference wire schema. Remove only those known fields after hashing.
    let mut document: serde_json::Value = serde_json::from_slice(config)?;
    let vq = document
        .pointer_mut("/tokenizer/vqvae")
        .ok_or_else(|| Error::Config("missing codec config".into()))?;
    let quant = vq
        .get_mut("vector_quantizer_config")
        .and_then(|v| v.as_object_mut())
        .ok_or_else(|| Error::Config("missing quantizer".into()))?;
    quant.insert("learned_l1_cascade_norm".into(), false.into());
    quant.insert("learned_l1_cascade_norm_final".into(), false.into());
    let aux = quant
        .get_mut("aux")
        .and_then(|v| v.as_object_mut())
        .ok_or_else(|| Error::Config("missing aux".into()))?;
    aux.remove("code_corruption");
    let fine = aux
        .get_mut("fine_drop")
        .and_then(|v| v.as_object_mut())
        .ok_or_else(|| Error::Config("missing fine drop".into()))?;
    for name in ["dual", "w_full", "w_trunc"] {
        fine.remove(name);
    }
    let raw: RawCodecConfig = serde_json::from_value(vq.clone())?;
    CodecConfig::validate_lm1b_small(raw)
}

fn expected_shapes() -> BTreeMap<String, Vec<usize>> {
    let mut shapes = BTreeMap::new();
    for (name, shape) in [
        ("l0_queries", vec![1, 1, 384]),
        ("lm_heads.head.weight", vec![8192, 768]),
        ("transformer.wle.weight", vec![12, 768]),
        ("transformer.wte.weight", vec![768, 384]),
        ("transformer.wte.bias", vec![768]),
        ("transformer.ln_f.weight", vec![768]),
        ("transformer.ln_f.bias", vec![768]),
    ] {
        shapes.insert(name.into(), shape);
    }
    for level in 0..12 {
        shapes.insert(format!("lm_heads.biases.{level}"), vec![8192]);
        shapes.insert(
            format!("post_upsample_convs.{level}.weight"),
            vec![384, 384, 3],
        );
        shapes.insert(format!("post_upsample_convs.{level}.bias"), vec![384]);
    }
    for block in 0..12 {
        for (name, shape) in [
            ("ln_1.weight", vec![768]),
            ("ln_1.bias", vec![768]),
            ("ln_2.weight", vec![768]),
            ("ln_2.bias", vec![768]),
            ("attn.c_attn.weight", vec![2304, 768]),
            ("attn.c_attn.bias", vec![2304]),
            ("attn.c_proj.weight", vec![768, 768]),
            ("attn.c_proj.bias", vec![768]),
            ("attn.q_norm.weight", vec![64]),
            ("attn.k_norm.weight", vec![64]),
            ("mlp.c_fc.weight", vec![3072, 768]),
            ("mlp.c_fc.bias", vec![3072]),
            ("mlp.c_proj.weight", vec![768, 3072]),
            ("mlp.c_proj.bias", vec![768]),
        ] {
            shapes.insert(format!("transformer.blocks.{block}.{name}"), shape);
        }
    }
    shapes
}

fn expected_codec_shapes() -> BTreeMap<String, Vec<usize>> {
    let mut shapes = BTreeMap::new();
    for (name, shape) in [
        ("decoder.ln_f.weight", vec![384]),
        ("decoder.output_layer.weight", vec![30528, 384]),
        ("decoder.wpe.weight", vec![128, 384]),
        ("encoder.wte.weight", vec![30528, 384]),
        ("vq.learned_down_l1.weight", vec![384, 384, 128]),
        ("vq.learned_down_l1.bias", vec![384]),
        ("vq.learned_up_l1.weight", vec![384, 384, 128]),
        ("vq.learned_up_l1.bias", vec![384]),
    ] {
        shapes.insert(name.into(), shape);
    }
    for level in 0..12 {
        shapes.insert(format!("vq.embeddings.{level}.weight"), vec![8192, 384]);
        shapes.insert(format!("vq.quant_resi.{level}.weight"), vec![384, 384, 3]);
        shapes.insert(format!("vq.quant_resi.{level}.bias"), vec![384]);
    }
    let blocks = std::iter::once("decoder.middle_block".to_owned())
        .chain((0..3).map(|i| format!("decoder.extra_blocks.{i}")));
    for block in blocks {
        for (name, shape) in [
            ("ln_1.weight", vec![384]),
            ("ln_1.bias", vec![384]),
            ("ln_2.weight", vec![384]),
            ("ln_2.bias", vec![384]),
            ("attn.c_attn.weight", vec![1152, 384]),
            ("attn.c_attn.bias", vec![1152]),
            ("attn.c_proj.weight", vec![384, 384]),
            ("attn.c_proj.bias", vec![384]),
            ("attn.q_norm.weight", vec![64]),
            ("attn.k_norm.weight", vec![64]),
            ("mlp.c_fc.weight", vec![1536, 384]),
            ("mlp.c_fc.bias", vec![1536]),
            ("mlp.c_proj.weight", vec![384, 1536]),
            ("mlp.c_proj.bias", vec![384]),
        ] {
            shapes.insert(format!("{block}.{name}"), shape);
        }
    }
    shapes
}

/// Host-only preparation inspection; it does not construct a codec, authenticate
/// an artifact, allocate a backend, or enable LM1B inference.
pub struct Lm1bSmallCodecInspection {
    pub codec_tensors: usize,
}
impl Lm1bSmallCodecInspection {
    pub fn from_preparation_bytes(config: &[u8], codec: &[u8]) -> Result<Self> {
        if ncm_io::sha256(config) != CONFIG_SHA {
            return Err(Error::Config(
                "unexpected pinned LM1B-small preparation config".into(),
            ));
        }
        let tensors = SafeTensors::deserialize(codec).map_err(|e| Error::Weights(e.to_string()))?;
        let shapes = expected_codec_shapes();
        if tensors.len() != shapes.len() {
            return Err(Error::Weights("LM1B codec inventory differs".into()));
        }
        for (name, shape) in &shapes {
            let value = tensors
                .tensor(name)
                .map_err(|e| Error::Weights(e.to_string()))?;
            if value.dtype() != Dtype::F32
                || value.shape() != shape
                || has_nonfinite_f32_le(value.data())
            {
                return Err(Error::Weights(format!("invalid LM1B codec tensor {name}")));
            }
        }
        Ok(Self {
            codec_tensors: tensors.len(),
        })
    }
}

fn validate_distribution(tensor: &TensorView<'_>, width: usize) -> Result<()> {
    if tensor.dtype() != Dtype::F32 || tensor.shape() != [width] {
        return Err(Error::Weights(
            "invalid LM1B inference distribution shape".into(),
        ));
    }
    let mut sum = 0.0_f64;
    for bytes in tensor.data().as_chunks::<4>().0 {
        let value = f32::from_le_bytes(*bytes);
        if !value.is_finite() || value < 0.0 {
            return Err(Error::Weights(
                "invalid LM1B inference distribution value".into(),
            ));
        }
        sum += f64::from(value);
    }
    if (sum - 1.0).abs() > 1e-6 {
        return Err(Error::Weights(
            "invalid LM1B inference distribution sum".into(),
        ));
    }
    Ok(())
}

/// Checks known preparation bytes on the host; it does not construct a model,
/// authenticate an artifact, upload weights, or enable LM1B inference.
pub struct Lm1bSmallGeneratorInspection {
    pub generator_tensors: usize,
    pub inference_tensors: usize,
}
impl Lm1bSmallGeneratorInspection {
    pub fn from_preparation_bytes(
        config: &[u8],
        generator: &[u8],
        inference: &[u8],
    ) -> Result<Self> {
        if ncm_io::sha256(config) != CONFIG_SHA {
            return Err(Error::Config(
                "unexpected pinned LM1B-small preparation config".into(),
            ));
        }
        let tensors =
            SafeTensors::deserialize(generator).map_err(|e| Error::Weights(e.to_string()))?;
        let shapes = expected_shapes();
        if tensors.len() != shapes.len() + 2 {
            return Err(Error::Weights("LM1B generator inventory differs".into()));
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
                    "invalid LM1B generator tensor {name}"
                )));
            }
        }
        let levels = tensors
            .tensor("lvl_1L")
            .map_err(|e| Error::Weights(e.to_string()))?;
        xsum_generator::validate_levels(&levels, &LEVELS)?;
        let mask = tensors
            .tensor("attn_mask_for_masking")
            .map_err(|e| Error::Weights(e.to_string()))?;
        xsum_generator::validate_mask(&mask, &LEVELS, 1)?;
        let tensors_inference =
            SafeTensors::deserialize(inference).map_err(|e| Error::Weights(e.to_string()))?;
        if tensors_inference.len() != 2 {
            return Err(Error::Weights("LM1B inference inventory differs".into()));
        }
        for (name, width) in [("level0_unigram", 8192), ("token_unigram", 30528)] {
            let tensor = tensors_inference
                .tensor(name)
                .map_err(|e| Error::Weights(e.to_string()))?;
            validate_distribution(&tensor, width)?;
        }
        Ok(Self {
            generator_tensors: tensors.len(),
            inference_tensors: tensors_inference.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Lm1bSmallCodecInspection, Lm1bSmallGeneratorInspection, expected_codec_shapes,
        expected_shapes, validate_distribution,
    };
    use safetensors::{Dtype, tensor::TensorView};

    #[test]
    fn exact_generator_shape_inventory() {
        let shapes = expected_shapes();
        assert_eq!(shapes.len(), 211);
        assert_eq!(shapes["lm_heads.head.weight"], [8192, 768]);
        assert_eq!(shapes["post_upsample_convs.11.weight"], [384, 384, 3]);
        assert!(!shapes.contains_key("lm_heads.0.weight"));
        assert!(Lm1bSmallGeneratorInspection::from_preparation_bytes(b"{}", b"", b"").is_err());
    }

    #[test]
    fn exact_codec_shape_inventory() {
        let shapes = expected_codec_shapes();
        assert_eq!(shapes.len(), 100);
        assert_eq!(shapes["decoder.wpe.weight"], [128, 384]);
        assert_eq!(
            shapes["decoder.extra_blocks.2.attn.c_attn.weight"],
            [1152, 384]
        );
        assert_eq!(shapes["vq.learned_up_l1.weight"], [384, 384, 128]);
        assert!(!shapes.contains_key("vq.ema_w_0"));
        assert!(Lm1bSmallCodecInspection::from_preparation_bytes(b"{}", b"").is_err());
    }

    #[test]
    fn distributions_reject_wrong_shape_values_and_sum() {
        let good: Vec<u8> = [0.25_f32, 0.75]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let value = TensorView::new(Dtype::F32, vec![2], &good).unwrap();
        assert!(validate_distribution(&value, 2).is_ok());
        assert!(validate_distribution(&value, 1).is_err());
        let bad: Vec<u8> = [f32::NAN, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        assert!(
            validate_distribution(&TensorView::new(Dtype::F32, vec![2], &bad).unwrap(), 2).is_err()
        );
        let bad_sum: Vec<u8> = [0.2_f32, 0.2]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        assert!(
            validate_distribution(&TensorView::new(Dtype::F32, vec![2], &bad_sum).unwrap(), 2)
                .is_err()
        );
    }
}
