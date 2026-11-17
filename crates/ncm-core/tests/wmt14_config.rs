use ncm_core::{CodecConfig, RawCodecConfig};

fn fixture() -> RawCodecConfig {
    serde_json::from_str(include_str!("fixtures/wmt14-codec-config-v1.json")).unwrap()
}

#[test]
fn wmt14_config_is_explicit_and_keeps_existing_profiles_narrow() {
    let raw = fixture();
    assert!(CodecConfig::validate(raw.clone()).is_err());
    assert!(CodecConfig::validate_tinystories(raw.clone()).is_err());
    assert!(CodecConfig::validate_xsum(raw.clone()).is_err());
    let config = CodecConfig::validate_wmt14(raw).unwrap();
    assert_eq!(config.strides(), &[4, 4, 4]);
    assert_eq!(config.code_length(), 313);
    let shapes = config.weight_shapes();
    assert_eq!(shapes.len(), 63);
    let bytes: usize = shapes
        .values()
        .map(|shape| shape.iter().product::<usize>() * 4)
        .sum();
    assert_eq!(bytes, 576_011_264);
    assert_eq!(shapes["encoder.wte.weight"], [151680, 384]);
    assert_eq!(shapes["vq.learned_up_l1.4.weight"], [384, 384, 4]);
    let mut tokens = vec![151679; 64];
    assert!(config.check_tokens(&tokens, 1).is_ok());
    tokens[0] = 151680;
    assert!(config.check_tokens(&tokens, 1).is_err());
    let mut codes = vec![4095; 313];
    assert!(config.check_codes(&codes, 1).is_ok());
    codes[0] = 4096;
    assert!(config.check_codes(&codes, 1).is_err());
}

#[test]
fn wmt14_rejects_profile_and_architecture_mutations() {
    let mut bad = fixture();
    bad.in_vocab_size = 151679;
    assert!(CodecConfig::validate_wmt14(bad).is_err());
    let mut bad = fixture();
    bad.vector_quantizer_config.levels.manual_levels[1] = 5;
    assert!(CodecConfig::validate_wmt14(bad).is_err());
    let mut bad = fixture();
    bad.vector_quantizer_config.learned_l1_cascade_norm_final = false;
    assert!(CodecConfig::validate_wmt14(bad).is_err());
    let mut bad = fixture();
    bad.vector_quantizer_config.learned_l1_cascade_base = 2;
    assert!(CodecConfig::validate_wmt14(bad).is_err());
    let mut bad = fixture();
    bad.vector_quantizer_config.quant_resi.ratio = 0.25;
    assert!(CodecConfig::validate_wmt14(bad).is_err());
    let mut bad = fixture();
    bad.pre_quant_groupnorm = 4;
    assert!(CodecConfig::validate_wmt14(bad).is_err());
    let mut bad = fixture();
    bad.dec_layers = 2;
    assert!(CodecConfig::validate_wmt14(bad).is_err());
}
