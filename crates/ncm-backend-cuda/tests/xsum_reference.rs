#![cfg(feature = "cuda-cudnn")]

use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, GeneratorBackend, WeightBackend};
use serde_json::Value;

fn input(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % 97) as f32 - 48.0) / 32.0).collect()
}
fn weight(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 1.0 + ((i % 11) as f32 - 5.0) / 128.0)
        .collect()
}
fn bias(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % 7) as f32 - 3.0) / 64.0).collect()
}

#[test]
#[ignore = "requires CUDA, cuDNN, and the explicitly selected real GPU"]
fn reference_operators_match_pinned_pytorch_cuda() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/xsum-reference-ops-v1.json"
    ))
    .unwrap();
    assert_eq!(fixture["schema_version"], 1);
    let backend = CandleBackend::cuda_cudnn(0)
        .unwrap()
        .xsum_reference_backend()
        .unwrap();
    let x = backend.upload_f32(&input(768), &[1, 1, 768]).unwrap();
    let w = backend.upload_f32(&weight(768), &[768]).unwrap();
    let b = backend.upload_f32(&bias(768), &[768]).unwrap();
    let q = backend.upload_f32(&input(768), &[1, 12, 1, 64]).unwrap();
    let rw = backend.upload_f32(&weight(64), &[64]).unwrap();
    let c = backend.upload_f32(&[0.8; 32], &[1, 32]).unwrap();
    let s = backend.upload_f32(&[0.6; 32], &[1, 32]).unwrap();
    let g = backend.upload_f32(&input(12 * 384), &[1, 12, 384]).unwrap();
    for (name, value) in [
        (
            "layernorm",
            backend.layer_norm(&x, &w, Some(&b), 1e-6).unwrap(),
        ),
        ("rmsnorm", backend.rms_norm(&q, &rw, 1e-6).unwrap()),
        ("groupnorm", backend.group_norm(&g, 8, 1e-5).unwrap()),
        ("gelu", backend.gelu(&x).unwrap()),
        ("rope", backend.rope(&q, &c, &s).unwrap()),
    ] {
        let actual = backend.read_f32(&value).unwrap();
        let expected = fixture["bits"][name].as_array().unwrap();
        assert_eq!(actual.len(), expected.len(), "{name}");
        for (i, (value, bits)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                u64::from(value.to_bits()),
                bits.as_u64().unwrap(),
                "{name}[{i}]"
            );
        }
    }
}
