use ncm_core::{CodecConfig, RawCodecConfig};

fn config() -> RawCodecConfig {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/xsum-codec-config-v1.json"
    ))
    .unwrap()
}

#[test]
fn xsum_level_zero_geometry_and_inventory_match_checkpoint() {
    let raw = config();
    assert!(CodecConfig::validate(raw.clone()).is_err());
    assert!(CodecConfig::validate_tinystories(raw.clone()).is_err());
    let validated = CodecConfig::validate_xsum(raw).unwrap();
    assert_eq!(validated.levels(), [4, 6, 10, 14, 20, 26, 32, 40, 48]);
    assert_eq!(validated.strides(), [4, 3]);
    let weights = validated.weight_shapes();
    assert_eq!(weights.len(), 53);
    assert_eq!(weights["vq.learned_down_l1.0.weight"], [384, 384, 4]);
    assert_eq!(weights["vq.learned_down_l1.2.weight"], [384, 384, 3]);
    assert_eq!(weights["vq.learned_up_l1.0.weight"], [384, 384, 3]);
    assert_eq!(weights["vq.learned_up_l1.2.weight"], [384, 384, 4]);
}

#[test]
fn xsum_profile_rejects_modified_first_level_and_normalization() {
    let mut raw = config();
    raw.vector_quantizer_config.levels.manual_levels[0] = 1;
    assert!(CodecConfig::validate_xsum(raw).is_err());
    let mut raw = config();
    raw.vector_quantizer_config.learned_l1_cascade_norm_final = false;
    assert!(CodecConfig::validate_xsum(raw).is_err());
}
