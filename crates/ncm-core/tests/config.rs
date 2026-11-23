use ncm_core::CodecConfig;
use serde_json::{Value, json};

fn tiny() -> Value {
    serde_json::from_str::<Value>(include_str!("../../../tests/fixtures/codec-v1.json")).unwrap()["config"].clone()
}
fn parse(v: &Value) -> ncm_core::Result<CodecConfig> {
    CodecConfig::from_json(&serde_json::to_vec(v).unwrap())
}

#[test]
fn tinystories_codec_is_explicit_and_does_not_widen_sudoku() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/tinystories-config-phase4.json"
    ))
    .unwrap();
    let raw = serde_json::from_value(fixture["tokenizer"]["vqvae"].clone()).unwrap();
    assert!(CodecConfig::validate(raw).is_err());
    let raw = serde_json::from_value(fixture["tokenizer"]["vqvae"].clone()).unwrap();
    let config = CodecConfig::validate_tinystories(raw).unwrap();
    assert_eq!(config.raw().embed_dim, 384);
    assert_eq!(config.strides(), [4, 4, 4, 2]);
    assert_eq!(config.code_length(), 641);
    assert_eq!(config.weight_shapes().len(), 70);
    assert_eq!(
        config.weight_shapes()["vq.embeddings.11.weight"],
        [4096, 384]
    );
    let mut bad = config.raw().clone();
    bad.vector_quantizer_config.ema_scale_by_world_size = true;
    assert!(CodecConfig::validate_tinystories(bad).is_err());
    let mut bad = config.raw().clone();
    bad.embed_dim = 256;
    assert!(CodecConfig::validate_tinystories(bad).is_err());

    // Upstream e34c3fab (15k EMA) inference config omits training-only keys.
    let mut trimmed = fixture["tokenizer"]["vqvae"].clone();
    let obj = trimmed.as_object_mut().unwrap();
    obj.remove("pre_quant_dropout");
    obj.insert("dropout".into(), json!(0.0));
    let vq = obj
        .get_mut("vector_quantizer_config")
        .unwrap()
        .as_object_mut()
        .unwrap();
    for key in [
        "decay",
        "epsilon",
        "commitment_cost",
        "aux",
        "ema_scale_by_world_size",
    ] {
        vq.remove(key);
    }
    let raw_trimmed = serde_json::from_value(trimmed).unwrap();
    let trimmed_cfg = CodecConfig::validate_tinystories(raw_trimmed).unwrap();
    assert_eq!(trimmed_cfg.weight_shapes().len(), 70);
}

#[test]
fn validated_inventory_matches_both_reference_profiles() {
    let v = tiny();
    let c = parse(&v).unwrap();
    assert_eq!(c.strides(), [4, 4]);
    assert_eq!(c.code_length(), 27);
    let fixture: Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/codec-v1.json")).unwrap();
    let actual: std::collections::BTreeMap<String, Vec<usize>> = fixture["weights"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, key)| {
            (
                name.clone(),
                serde_json::from_value(fixture["arrays"][key.as_str().unwrap()]["shape"].clone())
                    .unwrap(),
            )
        })
        .collect();
    assert_eq!(c.weight_shapes(), actual);
    let mut v = v;
    v["context_length"] = json!(89);
    v["embed_dim"] = json!(256);
    v["vqvae_vocab_size"] = json!([16; 12].to_vec());
    v["vector_quantizer_config"]["levels"]["manual_levels"] =
        json!([1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89]);
    v["vector_quantizer_config"]["learned_l1_cascade_norm"] = json!(false);
    v["vector_quantizer_config"]["learned_l1_cascade_norm_final"] = json!(false);
    let c = parse(&v).unwrap();
    assert_eq!(c.strides(), [89]);
    assert_eq!(c.code_length(), 322);
    let manifest: Value = serde_json::from_str(include_str!(
        "../../../docs/results/2026-11-10-sudoku-artifacts/raw/artifact-manifest.json"
    ))
    .unwrap();
    let expected = manifest["tensors"]["codec.safetensors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| {
            (
                t["name"].as_str().unwrap().to_string(),
                serde_json::from_value(t["shape"].clone()).unwrap(),
            )
        })
        .collect();
    assert_eq!(c.weight_shapes(), expected);
}

#[test]
fn malformed_and_unsupported_configs_fail() {
    for (pointer, value) in [
        ("/n_layers", json!(0)),
        ("/n_layers", json!(3)),
        ("/embed_dim", json!(0)),
        ("/embed_dim", json!(65536)),
        ("/context_length", json!(usize::MAX)),
        ("/compression_factor", json!(2)),
        ("/dec_layers", json!(2)),
        ("/enc_layers", json!(1)),
        ("/dec_positional", json!(true)),
        ("/pre_quant_groupnorm", json!(0)),
        ("/pre_quant_norm_type", json!("layer")),
        ("/dropout", json!(1.0)),
        ("/checkpoint_path", json!("../other.pt")),
        ("/vector_quantizer_config/clss", json!("other")),
        ("/vector_quantizer_config/learned_all_sampling", json!(true)),
        ("/vector_quantizer_config/learned_l1_cascade_base", json!(1)),
        (
            "/vector_quantizer_config/levels/manual_levels",
            json!([1, 3, 3, 16]),
        ),
        (
            "/vector_quantizer_config/levels/manual_levels",
            json!([1, 3, 7, 15]),
        ),
        (
            "/vector_quantizer_config/quant_resi/learnable_ratio",
            json!(true),
        ),
        ("/vector_quantizer_config/quant_resi/ratio", json!(0.0)),
        ("/vector_quantizer_config/aux/fine_drop/min_keep", json!(0)),
        ("/vqvae_vocab_size", json!([4, 4])),
    ] {
        let mut v = tiny();
        let (parent, key) = pointer.rsplit_once('/').unwrap();
        v.pointer_mut(parent).unwrap()[key] = value;
        assert!(parse(&v).is_err(), "accepted {pointer}");
    }
    for field in ["pre_quant_norm_type", "dec_layers"] {
        let mut v = tiny();
        v.as_object_mut().unwrap().remove(field);
        assert!(parse(&v).is_err(), "accepted missing {field}");
    }
    let mut v = tiny();
    v["vector_quantizer_config"]["ema_scale_by_world_size"] = json!(false);
    assert!(parse(&v).is_err());
    let mut v = tiny();
    v["unknown_inference_option"] = json!(true);
    assert!(parse(&v).is_err());
    let mut v = tiny();
    v["vector_quantizer_config"]["unknown"] = json!(true);
    assert!(parse(&v).is_err());
    let raw = serde_json::to_string(&tiny()).unwrap().replacen(
        "\"n_layers\":4",
        "\"n_layers\":4,\"n_layers\":8",
        1,
    );
    assert!(CodecConfig::from_json(raw.as_bytes()).is_err());
    let mut raw = parse(&tiny()).unwrap().raw().clone();
    raw.dropout = f64::NAN;
    assert!(CodecConfig::validate(raw).is_err());
}

#[test]
fn ids_and_batch_shapes_are_checked_before_execution() {
    let c = parse(&tiny()).unwrap();
    assert!(c.check_tokens(&[0; 32], 2).is_ok());
    assert!(c.check_tokens(&[0; 16], 2).is_err());
    assert!(c.check_tokens(&[12; 16], 1).is_err());
    assert!(c.check_tokens(&[], 0).is_err());
    assert!(c.check_tokens(&[], usize::MAX).is_err());
    assert!(c.check_codes(&[3; 54], 2).is_ok());
    assert!(c.check_codes(&[3; 53], 2).is_err());
    assert!(c.check_codes(&[4; 27], 1).is_err());
    let mut raw = c.raw().clone();
    raw.vqvae_vocab_size = vec![2, 3, 4, 5];
    let c = CodecConfig::validate(raw).unwrap();
    let mut ids = vec![0; 27];
    ids[1] = 2;
    assert!(c.check_codes(&ids, 1).is_ok());
    ids[0] = 2;
    assert!(c.check_codes(&ids, 1).is_err());
}
