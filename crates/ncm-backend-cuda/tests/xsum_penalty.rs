//! Real-CUDA comparisons for the explicitly selected FP64 XSum decoder.
#![cfg(feature = "cuda")]
use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, GeneratorBackend, WeightBackend};
use ncm_model::select_xsum_tokens_cpu;

#[test]
fn xsum_fp64_device_matches_pinned_oracle_and_boundaries() {
    let backend = CandleBackend::cuda(0).unwrap();
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/xsum-penalty-oracle-v2.json"
    ))
    .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let batch = case["batch"].as_u64().unwrap() as usize;
        let time = case["time"].as_u64().unwrap() as usize;
        let vocab = case["vocab"].as_u64().unwrap() as usize;
        let logits: Vec<f32> = serde_json::from_value(case["logits"].clone()).unwrap();
        let allowance: Vec<f32> = serde_json::from_value(case["allowance"].clone()).unwrap();
        let expected: Vec<u32> = serde_json::from_value(case["expected"].clone()).unwrap();
        let x = backend.upload_f32(&logits, &[batch, time, vocab]).unwrap();
        let a = backend.upload_f32(&allowance, &[vocab]).unwrap();
        let result = backend.xsum_penalty_fp64(&x, &a).unwrap();
        assert_eq!(backend.indices_shape(&result), [batch, time]);
        assert_eq!(
            backend.read_indices(&result).unwrap(),
            expected,
            "{}",
            case["name"]
        );
    }
    for vocab in [1, 2, 7, 255, 256, 257, 511, 512, 513, 4097] {
        let (batch, time) = (2, 9);
        let logits: Vec<f32> = (0..batch * time * vocab)
            .map(|i| ((i * 37 + 11) % 29) as f32 / 4.0 - 3.0)
            .collect();
        let allowance: Vec<f32> = (0..vocab).map(|i| ((i * 13) % 37) as f32 / 64.0).collect();
        let expected = select_xsum_tokens_cpu(&logits, &allowance, batch, time).unwrap();
        let x = backend.upload_f32(&logits, &[batch, time, vocab]).unwrap();
        let a = backend.upload_f32(&allowance, &[vocab]).unwrap();
        assert_eq!(
            backend
                .read_indices(&backend.xsum_penalty_fp64(&x, &a).unwrap())
                .unwrap(),
            expected,
            "vocab {vocab}"
        );
    }
    let x = backend.upload_f32(&[f32::NAN, 0.0], &[1, 1, 2]).unwrap();
    let a = backend.upload_f32(&[0.0, 0.0], &[2]).unwrap();
    assert_eq!(
        backend
            .read_indices(&backend.xsum_penalty_fp64(&x, &a).unwrap())
            .unwrap(),
        [u32::MAX]
    );
    let later = backend
        .upload_f32(&[1.0, 0.0, f32::NAN, 0.0], &[1, 2, 2])
        .unwrap();
    assert!(
        backend
            .read_indices(&backend.xsum_penalty_fp64(&later, &a).unwrap())
            .unwrap()
            .contains(&u32::MAX)
    );
    let negative = backend.upload_f32(&[0.0, -1.0], &[2]).unwrap();
    let finite = backend.upload_f32(&[1.0, 0.0], &[1, 1, 2]).unwrap();
    assert!(
        backend
            .read_indices(&backend.xsum_penalty_fp64(&finite, &negative).unwrap())
            .unwrap()
            .contains(&u32::MAX)
    );
    let wrong = backend.upload_f32(&[0.0], &[1]).unwrap();
    assert!(backend.xsum_penalty_fp64(&x, &wrong).is_err());
}
