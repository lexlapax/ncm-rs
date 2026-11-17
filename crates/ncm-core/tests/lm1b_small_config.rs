use ncm_core::{CodecConfig, RawCodecConfig};

fn fixture() -> RawCodecConfig {
    serde_json::from_str(include_str!("fixtures/lm1b-small-codec-config-v1.json")).unwrap()
}

#[test]
fn pinned_architecture_has_exact_inventory_and_remains_profile_scoped() {
    let raw = fixture();
    assert!(CodecConfig::validate(raw.clone()).is_err());
    assert!(CodecConfig::validate_tinystories(raw.clone()).is_err());
    assert!(CodecConfig::validate_wmt14(raw.clone()).is_err());
    let codec = CodecConfig::validate_lm1b_small(raw).unwrap();
    assert_eq!(codec.strides(), &[128]);
    assert_eq!(codec.code_length(), 641);
    let shapes = codec.weight_shapes();
    assert_eq!(shapes.len(), 100);
    assert!(!shapes.contains_key("encoder.wpe.weight"));
    assert_eq!(shapes["decoder.wpe.weight"], [128, 384]);
    assert_eq!(
        shapes["decoder.extra_blocks.2.attn.c_attn.weight"],
        [1152, 384]
    );
    assert_eq!(shapes["vq.learned_down_l1.weight"], [384, 384, 128]);
}

#[test]
fn rejects_unsupported_lm1b_architectures_and_invalid_probabilities() {
    let mut raw = fixture();
    raw.dec_layers = 3;
    assert!(CodecConfig::validate_lm1b_small(raw).is_err());
    let mut raw = fixture();
    raw.enc_positional = true;
    assert!(CodecConfig::validate_lm1b_small(raw).is_err());
    let mut raw = fixture();
    raw.vector_quantizer_config.learned_l1_cascade = true;
    assert!(CodecConfig::validate_lm1b_small(raw).is_err());
    let mut raw = fixture();
    raw.dropout = 1.0;
    assert!(CodecConfig::validate_lm1b_small(raw).is_err());
}
