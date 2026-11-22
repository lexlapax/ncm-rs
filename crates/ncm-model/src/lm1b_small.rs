//! Preflight of pinned LM1B models (small, medium, large) and diagnostic rotary bytes.
use crate::{has_nonfinite_f32_le, xsum_generator};
use ncm_core::{CodecConfig, Error, RawCodecConfig, Result};
use ncm_io::Artifact;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use std::collections::BTreeMap;

const CONFIG_SHA: &str = "1bc5e0c10050c8221ec72c277d8510a9a541117db5705cb12173f12b733a1475";
const LEVELS: [usize; 12] = [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128];
const LM1B_ROPE_COS_SHA256: &str =
    "d657dbfb9ad59d73f5d108d944e5cfc6ca81feef4b01a6cf46c68a26315af18a";
const LM1B_ROPE_SIN_SHA256: &str =
    "bb349c94065a91579983a0e0c1e20621e943e4975d720a41e474170f3822ba29";

/// A pinned LM1B generator size. All sizes share the codec, tokenizer, levels and
/// 64-wide attention heads; only generator depth, head count and width differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lm1bSize {
    pub profile: &'static str,
    config_sha256: &'static str,
    pub layers: usize,
    pub heads: usize,
    pub width: usize,
}

/// The three released LM1B sizes, with their exported `config.json` SHA-256.
pub const LM1B_SIZES: [Lm1bSize; 3] = [
    Lm1bSize {
        profile: "lm1b-small-inference-v1",
        config_sha256: CONFIG_SHA,
        layers: 12,
        heads: 12,
        width: 768,
    },
    Lm1bSize {
        profile: "lm1b-medium-inference-v1",
        config_sha256: "017c3a44d475ee549a969ee08c484936471430de0560f0f91dcd881969a8f3fd",
        layers: 24,
        heads: 16,
        width: 1024,
    },
    Lm1bSize {
        profile: "lm1b-large-inference-v1",
        config_sha256: "cb824d57d3ab56717858176fd7ca59939b17d92cca33bccda3b05da2774a7d61",
        layers: 36,
        heads: 20,
        width: 1280,
    },
];

impl Lm1bSize {
    /// The size for an artifact profile (`-v1` or self-contained `-v2`); unknown profiles fail explicitly.
    pub fn from_profile(profile: &str) -> Result<Self> {
        let normalized = profile
            .strip_suffix("-v2")
            .map(|prefix| format!("{prefix}-v1"));
        let candidate = normalized.as_deref().unwrap_or(profile);
        LM1B_SIZES
            .into_iter()
            .find(|size| size.profile == candidate)
            .ok_or_else(|| Error::Config("unsupported LM1B artifact profile".into()))
    }

    /// The size whose pinned configuration has exactly these bytes.
    fn from_config(config: &[u8]) -> Result<Self> {
        let digest = ncm_io::sha256(config);
        LM1B_SIZES
            .into_iter()
            .find(|size| size.config_sha256 == digest)
            .ok_or_else(|| Error::Config("unexpected pinned LM1B config".into()))
    }

    /// Generator tensors in the artifact, including the two mask/level metadata tensors.
    pub fn generator_tensors(&self) -> usize {
        expected_shapes(self).len() + 2
    }
}

/// Admit the pinned LM1B-small reference rotary diagnostic.
pub fn lm1b_small_reference_rope(bytes: &[u8]) -> Result<(Vec<f32>, Vec<f32>)> {
    if ncm_io::sha256(bytes) != "9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc" {
        return Err(Error::Weights(
            "unexpected LM1B diagnostic rotary tables".into(),
        ));
    }
    let tables = SafeTensors::deserialize(bytes).map_err(|e| Error::Weights(e.to_string()))?;
    let values = |name| -> Result<Vec<f32>> {
        let tensor = tables
            .tensor(name)
            .map_err(|e| Error::Weights(e.to_string()))?;
        if tensor.dtype() != Dtype::F32 || tensor.shape() != [128, 32] {
            return Err(Error::Weights(
                "unexpected LM1B rotary shape or type".into(),
            ));
        }
        Ok(tensor
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect())
    };
    Ok((values("cosine")?, values("sine")?))
}

pub(crate) fn codec_config(artifact: &Artifact) -> Result<CodecConfig> {
    let size = Lm1bSize::from_profile(&artifact.manifest().profile)?;
    let config = artifact
        .file("config.json")
        .map_err(|e| Error::Weights(e.to_string()))?;
    if Lm1bSize::from_config(config)? != size {
        return Err(Error::Config("LM1B config differs from profile".into()));
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
    // Medium and large configs name the shared tokenizer checkpoint as "../tokenizer.pt".
    // The path is metadata only and the config bytes are already pinned per size.
    if let Some(path) = vq.get_mut("checkpoint_path")
        && path == "../tokenizer.pt"
    {
        *path = "tokenizer.pt".into();
    }
    let raw: RawCodecConfig = serde_json::from_value(vq.clone())?;
    CodecConfig::validate_lm1b_small(raw)
}

fn expected_shapes(size: &Lm1bSize) -> BTreeMap<String, Vec<usize>> {
    let w = size.width;
    let mut shapes = BTreeMap::new();
    for (name, shape) in [
        ("l0_queries", vec![1, 1, 384]),
        ("lm_heads.head.weight", vec![8192, w]),
        ("transformer.wle.weight", vec![12, w]),
        ("transformer.wte.weight", vec![w, 384]),
        ("transformer.wte.bias", vec![w]),
        ("transformer.ln_f.weight", vec![w]),
        ("transformer.ln_f.bias", vec![w]),
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
    for block in 0..size.layers {
        for (name, shape) in [
            ("ln_1.weight", vec![w]),
            ("ln_1.bias", vec![w]),
            ("ln_2.weight", vec![w]),
            ("ln_2.bias", vec![w]),
            ("attn.c_attn.weight", vec![3 * w, w]),
            ("attn.c_attn.bias", vec![3 * w]),
            ("attn.c_proj.weight", vec![w, w]),
            ("attn.c_proj.bias", vec![w]),
            ("attn.q_norm.weight", vec![64]),
            ("attn.k_norm.weight", vec![64]),
            ("mlp.c_fc.weight", vec![4 * w, w]),
            ("mlp.c_fc.bias", vec![4 * w]),
            ("mlp.c_proj.weight", vec![w, 4 * w]),
            ("mlp.c_proj.bias", vec![w]),
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
        Lm1bSize::from_config(config)?;
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

fn decode_lm1b_rope_tensor(tensor: &TensorView<'_>, expected_sha256: &str) -> Result<Vec<f32>> {
    if tensor.dtype() != Dtype::F32
        || tensor.shape() != [128, 32]
        || ncm_io::sha256(tensor.data()) != expected_sha256
    {
        return Err(Error::Weights("invalid LM1B embedded rotary table".into()));
    }
    Ok(tensor
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect())
}

fn validate_inference_bundle(
    inference: &[u8],
    expect_rope: bool,
) -> Result<(usize, Option<(Vec<f32>, Vec<f32>)>)> {
    let tensors_inference =
        SafeTensors::deserialize(inference).map_err(|e| Error::Weights(e.to_string()))?;
    let expected_count = if expect_rope { 4 } else { 2 };
    if tensors_inference.len() != expected_count {
        return Err(Error::Weights("LM1B inference inventory differs".into()));
    }
    for (name, width) in [("level0_unigram", 8192), ("token_unigram", 30528)] {
        let tensor = tensors_inference
            .tensor(name)
            .map_err(|e| Error::Weights(e.to_string()))?;
        validate_distribution(&tensor, width)?;
    }
    let rope_tables = if expect_rope {
        let cos = tensors_inference
            .tensor("rope_cos")
            .map_err(|e| Error::Weights(e.to_string()))?;
        let sin = tensors_inference
            .tensor("rope_sin")
            .map_err(|e| Error::Weights(e.to_string()))?;
        Some((
            decode_lm1b_rope_tensor(&cos, LM1B_ROPE_COS_SHA256)?,
            decode_lm1b_rope_tensor(&sin, LM1B_ROPE_SIN_SHA256)?,
        ))
    } else {
        None
    };
    Ok((tensors_inference.len(), rope_tables))
}

/// Checks known preparation bytes on the host; it does not construct a model,
/// authenticate an artifact, upload weights, or enable LM1B inference.
pub struct Lm1bSmallGeneratorInspection {
    pub size: Lm1bSize,
    pub generator_tensors: usize,
    pub inference_tensors: usize,
    pub rope_tables: Option<(Vec<f32>, Vec<f32>)>,
}
impl Lm1bSmallGeneratorInspection {
    pub fn from_artifact(artifact: &Artifact) -> Result<Self> {
        let expect_rope = artifact.manifest().profile.ends_with("-v2");
        let config = artifact
            .file("config.json")
            .map_err(|e| Error::Weights(e.to_string()))?;
        let inference = artifact
            .file("inference.safetensors")
            .map_err(|e| Error::Weights(e.to_string()))?;
        if !artifact.is_streaming_bundle("generator.safetensors") {
            let generator = artifact
                .file("generator.safetensors")
                .map_err(|e| Error::Weights(e.to_string()))?;
            return Self::from_preparation_bytes_with_rope(
                config,
                generator,
                inference,
                expect_rope,
            );
        }
        let size = Lm1bSize::from_config(config)?;
        if !artifact.bundle_f32_weights_verified_finite("generator.safetensors") {
            return Err(Error::Weights(
                "streaming LM1B generator missing finite-value preflight".into(),
            ));
        }
        let entries = artifact
            .bundle_tensors("generator.safetensors")
            .map_err(|e| Error::Weights(e.to_string()))?;
        let shapes = expected_shapes(&size);
        if entries.len() != shapes.len() + 2 {
            return Err(Error::Weights("LM1B generator inventory differs".into()));
        }
        let by_name: BTreeMap<_, _> = entries
            .iter()
            .map(|entry| (entry.name.as_str(), entry))
            .collect();
        for (name, shape) in &shapes {
            let entry = by_name
                .get(name.as_str())
                .ok_or_else(|| Error::Weights(format!("invalid LM1B generator tensor {name}")))?;
            if !matches!(entry.dtype, ncm_io::TensorDtype::F32) || entry.shape != *shape {
                return Err(Error::Weights(format!(
                    "invalid LM1B generator tensor {name}"
                )));
            }
        }
        let level_entry = by_name
            .get("lvl_1L")
            .ok_or_else(|| Error::Weights("missing generator level buffer".into()))?;
        let level_bytes = artifact
            .read_bundle_tensor("generator.safetensors", "lvl_1L")
            .map_err(|e| Error::Weights(e.to_string()))?;
        xsum_generator::validate_levels_raw(
            level_entry.dtype,
            &level_entry.shape,
            &level_bytes,
            &LEVELS,
        )?;
        let mask_entry = by_name
            .get("attn_mask_for_masking")
            .ok_or_else(|| Error::Weights("missing generator attention mask".into()))?;
        let mask_bytes = artifact
            .read_bundle_tensor("generator.safetensors", "attn_mask_for_masking")
            .map_err(|e| Error::Weights(e.to_string()))?;
        xsum_generator::validate_mask_raw(
            mask_entry.dtype,
            &mask_entry.shape,
            &mask_bytes,
            &LEVELS,
            1,
        )?;
        let (inference_tensors, rope_tables) = validate_inference_bundle(inference, expect_rope)?;
        Ok(Self {
            size,
            generator_tensors: entries.len(),
            inference_tensors,
            rope_tables,
        })
    }

    pub fn from_preparation_bytes(
        config: &[u8],
        generator: &[u8],
        inference: &[u8],
    ) -> Result<Self> {
        Self::from_preparation_bytes_with_rope(config, generator, inference, false)
    }

    fn from_preparation_bytes_with_rope(
        config: &[u8],
        generator: &[u8],
        inference: &[u8],
        expect_rope: bool,
    ) -> Result<Self> {
        let size = Lm1bSize::from_config(config)?;
        let tensors =
            SafeTensors::deserialize(generator).map_err(|e| Error::Weights(e.to_string()))?;
        let shapes = expected_shapes(&size);
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
        let (inference_tensors, rope_tables) = validate_inference_bundle(inference, expect_rope)?;
        Ok(Self {
            size,
            generator_tensors: tensors.len(),
            inference_tensors,
            rope_tables,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        LM1B_SIZES, Lm1bSize, Lm1bSmallCodecInspection, Lm1bSmallGeneratorInspection,
        expected_codec_shapes, expected_shapes, validate_distribution,
    };
    use safetensors::{Dtype, tensor::TensorView};

    #[test]
    fn exact_generator_shape_inventory() {
        let shapes = expected_shapes(&LM1B_SIZES[0]);
        assert_eq!(shapes.len(), 211);
        assert_eq!(shapes["lm_heads.head.weight"], [8192, 768]);
        assert_eq!(shapes["post_upsample_convs.11.weight"], [384, 384, 3]);
        assert!(!shapes.contains_key("lm_heads.0.weight"));
        assert!(Lm1bSmallGeneratorInspection::from_preparation_bytes(b"{}", b"", b"").is_err());
    }

    #[test]
    fn sizes_match_released_inventories() {
        let counts: Vec<_> = LM1B_SIZES.iter().map(Lm1bSize::generator_tensors).collect();
        assert_eq!(counts, [213, 381, 549]);
        for size in LM1B_SIZES {
            assert_eq!(size.width, size.heads * 64);
            assert_eq!(Lm1bSize::from_profile(size.profile).unwrap(), size);
            let v2 = size.profile.replace("-v1", "-v2");
            assert_eq!(Lm1bSize::from_profile(&v2).unwrap(), size);
        }
        let large = expected_shapes(&LM1B_SIZES[2]);
        assert_eq!(
            large["transformer.blocks.35.attn.c_attn.weight"],
            [3840, 1280]
        );
        assert_eq!(
            large["transformer.blocks.35.mlp.c_proj.weight"],
            [1280, 5120]
        );
        assert!(!large.contains_key("transformer.blocks.36.ln_1.weight"));
        assert!(Lm1bSize::from_profile("lm1b-huge-inference-v1").is_err());
        assert!(Lm1bSize::from_config(b"{}").is_err());
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
