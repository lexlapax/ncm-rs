//! Compare explicit CPU selection with the pinned upstream CPU/CUDA oracle.
use ncm_model::select_xsum_tokens_cpu;
use serde_json::Value;

#[test]
fn xsum_selection_matches_pinned_upstream_on_varied_ties_and_repetition() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/xsum-penalty-oracle-v2.json"
    ))
    .unwrap();
    assert_eq!(fixture["schema_version"], 1);
    assert_eq!(
        fixture["upstream_commit"],
        "c70274273f489abc73832b2b98584a8483b6b203"
    );
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3);
    for case in cases {
        let values = |key: &str| -> Vec<f32> {
            case[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        };
        let expected: Vec<u32> = serde_json::from_value(case["expected"].clone()).unwrap();
        let batch = case["batch"].as_u64().unwrap() as usize;
        let time = case["time"].as_u64().unwrap() as usize;
        let allowance = values("allowance");
        assert_eq!(allowance.len(), case["vocab"].as_u64().unwrap() as usize);
        let actual = select_xsum_tokens_cpu(&values("logits"), &allowance, batch, time).unwrap();
        assert_eq!(actual, expected, "case {}", case["name"]);
    }
}
