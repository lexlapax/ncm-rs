//! Check the pinned text tokenizer when embedded in a validated model artifact.
use ncm_tokenizers::TextTokenizer;
use serde_json::Value;
use std::fs;

#[test]
#[ignore = "requires the local pinned TinyStories artifact"]
fn tinystories_embedded_assets_preserve_pinned_tokenizer_behavior() {
    let root = std::path::PathBuf::from(
        std::env::var_os("NCM_PHASE4_ARTIFACT").expect("NCM_PHASE4_ARTIFACT required"),
    );
    let manifest = fs::read(root.join("tokenizer-manifest.json")).unwrap();
    let payload = fs::read(root.join("tokenizer.json")).unwrap();
    let license = fs::read(root.join("tokenizer.LICENSE")).unwrap();
    let tokenizer = TextTokenizer::from_asset_bytes(&manifest, &payload, &license).unwrap();
    let fixture: Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/tokenizers-v1.json")).unwrap();
    let cases = fixture["profiles"]["tinystories"]["cases"]
        .as_array()
        .unwrap();
    assert!(!cases.is_empty());
    for case in cases {
        let ids: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
        assert_eq!(
            tokenizer.encode(case["text"].as_str().unwrap()).unwrap(),
            ids
        );
        for (skip, key) in [(false, "decoded"), (true, "decoded_skip_special")] {
            assert_eq!(
                tokenizer.decode(&ids, skip).unwrap(),
                case[key].as_str().unwrap()
            );
        }
    }
    for case in fixture["profiles"]["tinystories"]["decode_cases"]
        .as_array()
        .unwrap()
    {
        let ids: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
        for (skip, key) in [(false, "decoded"), (true, "decoded_skip_special")] {
            assert_eq!(
                tokenizer.decode(&ids, skip).unwrap(),
                case[key].as_str().unwrap()
            );
        }
    }
    let mut changed = payload.clone();
    changed[0] ^= 1;
    assert!(TextTokenizer::from_asset_bytes(&manifest, &changed, &license).is_err());
    let mut changed = manifest.clone();
    changed[0] ^= 1;
    assert!(TextTokenizer::from_asset_bytes(&changed, &payload, &license).is_err());
    let mut changed = license.clone();
    changed[0] ^= 1;
    assert!(TextTokenizer::from_asset_bytes(&manifest, &payload, &changed).is_err());
}

#[test]
#[ignore = "requires the local pinned XSum artifact"]
fn xsum_embedded_gpt2_matches_pinned_tokenizer_fixture() {
    let root = std::path::PathBuf::from(
        std::env::var_os("NCM_XSUM_ARTIFACT").expect("NCM_XSUM_ARTIFACT required"),
    );
    let manifest = fs::read(root.join("tokenizer-manifest.json")).unwrap();
    let payload = fs::read(root.join("tokenizer.json")).unwrap();
    let license = fs::read(root.join("tokenizer.LICENSE")).unwrap();
    let tokenizer = TextTokenizer::from_asset_bytes(&manifest, &payload, &license).unwrap();
    let fixture: Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/tokenizers-v1.json")).unwrap();
    let profile = &fixture["profiles"]["gpt2"];
    let cases = profile["cases"].as_array().unwrap();
    assert!(!cases.is_empty());
    for case in cases {
        let ids: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
        assert_eq!(
            tokenizer.encode(case["text"].as_str().unwrap()).unwrap(),
            ids
        );
        for (skip, key) in [(false, "decoded"), (true, "decoded_skip_special")] {
            assert_eq!(
                tokenizer.decode(&ids, skip).unwrap(),
                case[key].as_str().unwrap()
            );
        }
    }
    let decode_cases = profile["decode_cases"].as_array().unwrap();
    assert!(!decode_cases.is_empty());
    for case in decode_cases {
        let ids: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
        for (skip, key) in [(false, "decoded"), (true, "decoded_skip_special")] {
            assert_eq!(
                tokenizer.decode(&ids, skip).unwrap(),
                case[key].as_str().unwrap()
            );
        }
    }
    let mut changed = license;
    changed[0] ^= 1;
    assert!(TextTokenizer::from_asset_bytes(&manifest, &payload, &changed).is_err());
}
