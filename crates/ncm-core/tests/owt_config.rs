use ncm_core::OwtConfigOnly;
use serde_json::Value;

const FIXTURE: &str = include_str!("../../../tests/fixtures/owt-config-v1.json");

fn edited(path: &str, value: Value) -> Vec<u8> {
    let mut doc: Value = serde_json::from_str(FIXTURE).unwrap();
    *doc.pointer_mut(path).unwrap() = value;
    serde_json::to_vec(&doc).unwrap()
}

#[test]
fn pinned_owt_config_is_validated_but_not_executable() {
    let owt = OwtConfigOnly::from_json(FIXTURE.as_bytes()).unwrap();
    assert_eq!(owt.levels.len(), 32);
    assert_eq!((owt.levels[0], owt.levels[31]), (1, 1024));
    assert_eq!(owt.levels.iter().sum::<usize>(), 11_440);
    assert_eq!(
        (
            owt.generator_layers,
            owt.generator_heads,
            owt.generator_width
        ),
        (16, 16, 1024)
    );
    assert_eq!(owt.generator_width / owt.generator_heads, 64);
    assert!(!owt.cut_at_eos);
    let error = owt.require_released_checkpoint().unwrap_err().to_string();
    assert!(error.contains("config-only"));
}

#[test]
fn changed_owt_semantics_are_rejected() {
    for (path, value) in [
        ("/generation/cut_at_eos", Value::Bool(true)),
        ("/model/n_layer", 12.into()),
        ("/model/tie_heads_bias", Value::Bool(false)),
        ("/model/query_level0", Value::Bool(true)),
        ("/generation/top_p", 0.9.into()),
        ("/tokenizer/vqvae/context_length", 512.into()),
        ("/task", "lm1b-small".into()),
    ] {
        assert!(
            OwtConfigOnly::from_json(&edited(path, value)).is_err(),
            "{path} change accepted"
        );
    }
    let mut doc: Value = serde_json::from_str(FIXTURE).unwrap();
    doc["tokenizer"]["vqvae"]["vector_quantizer_config"]["levels"]["manual_levels"][31] =
        1023.into();
    assert!(OwtConfigOnly::from_json(&serde_json::to_vec(&doc).unwrap()).is_err());
    assert!(OwtConfigOnly::from_json(b"{}").is_err());
}
