//! Controlled, same-draw validation of the experimental CUDA categorical kernel.
#![cfg(feature = "cuda")]
use ncm_backend_cuda::{CandleBackend, categorical::categorical_from_uniforms};
use ncm_core::{CodecBackend, GeneratorBackend, WeightBackend};

#[test]
fn categorical_uniforms_and_invalid_rows() {
    let backend = CandleBackend::cuda(0).unwrap();
    let draws = [
        0, 0x10000000, 0x7fffffff, 0xffffffff, 0x40000000, 0x80000000, 0x20000000,
    ];
    let mut scores = vec![0.0f32; draws.len() * 4096];
    scores[4 * 4096..5 * 4096].fill(f32::NEG_INFINITY);
    scores[4 * 4096 + 237] = 0.0;
    scores[5 * 4096..6 * 4096].fill(f32::NEG_INFINITY);
    scores[6 * 4096 + 1] = f32::NAN;
    let x = backend
        .upload_f32(&scores, &[1, draws.len(), 4096])
        .unwrap();
    let u = backend.upload_indices(&draws, &[1, draws.len()]).unwrap();
    let result = backend
        .read_indices(&categorical_from_uniforms(&backend, &x, &u).unwrap())
        .unwrap();
    assert_eq!(result, [0, 256, 2047, 4095, 237, u32::MAX, u32::MAX]);
    let wrong = backend.upload_indices(&[0], &[1, 1]).unwrap();
    assert!(categorical_from_uniforms(&backend, &x, &wrong).is_err());
}

#[test]
fn categorical_matches_reference_cdf_for_controlled_nonuniform_rows() {
    let backend = CandleBackend::cuda(0).unwrap();
    let mut scores = Vec::new();
    let mut uniforms = Vec::new();
    let mut expected = Vec::new();
    for row in 0..64u32 {
        let row_scores: Vec<f32> = (0..4096u32)
            .map(|i| -(((i * 43 + row * 31) % 127) as f32) * 0.03125)
            .collect();
        let weights: Vec<f64> = row_scores.iter().map(|&v| f64::from(v).exp()).collect();
        let total: f64 = weights.iter().sum();
        let random = row.wrapping_mul(2_654_435_761).wrapping_add(1_234_567);
        let threshold = f64::from(random) / 4_294_967_296.0 * total;
        let mut sum = 0.0;
        let selected = weights
            .iter()
            .position(|weight| {
                sum += weight;
                threshold < sum
            })
            .unwrap();
        scores.extend(row_scores);
        uniforms.push(random);
        expected.push(selected as u32);
    }
    let x = backend.upload_f32(&scores, &[1, 64, 4096]).unwrap();
    let u = backend.upload_indices(&uniforms, &[1, 64]).unwrap();
    let actual = backend
        .read_indices(&categorical_from_uniforms(&backend, &x, &u).unwrap())
        .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn sampled_embedding_bounds_invalid_sentinel_without_erasing_it() {
    let backend = CandleBackend::cuda(0).unwrap();
    let weights: Vec<f32> = (0..4096).map(|i| i as f32).collect();
    let weight = backend.upload_f32(&weights, &[4096, 1]).unwrap();
    let original = [0, u32::MAX, 4095, 4096];
    let indices = backend.upload_indices(&original, &[1, 4]).unwrap();
    let values = backend.sampled_embedding_4096(&weight, &indices).unwrap();
    assert_eq!(
        backend.read_f32(&values).unwrap(),
        [0.0, 4095.0, 4095.0, 4095.0]
    );
    assert_eq!(backend.read_indices(&indices).unwrap(), original);
}

#[test]
fn sampled_embedding_8192_bounds_invalid_sentinel_without_erasing_it() {
    let backend = CandleBackend::cuda(0).unwrap();
    let weights: Vec<f32> = (0..8192).map(|i| i as f32).collect();
    let weight = backend.upload_f32(&weights, &[8192, 1]).unwrap();
    let original = [0, u32::MAX, 8191, 8192];
    let indices = backend.upload_indices(&original, &[1, 4]).unwrap();
    let values = backend.sampled_embedding_8192(&weight, &indices).unwrap();
    assert_eq!(
        backend.read_f32(&values).unwrap(),
        [0.0, 8191.0, 8191.0, 8191.0]
    );
    assert_eq!(backend.read_indices(&indices).unwrap(), original);
}

#[test]
fn unique_selection_preserves_invalid_rows_through_safe_embedding_and_concatenation() {
    let backend = CandleBackend::cuda(0).unwrap();
    let mut scores = vec![0.0f32; 3 * 4096];
    scores[7] = 2.0;
    scores[4096 + 2] = 3.0;
    scores[4096 + 4] = 3.0;
    scores[2 * 4096 + 1] = f32::NAN;
    let logits = backend.upload_f32(&scores, &[1, 3, 4096]).unwrap();
    let selected = backend.unique_max_4096(&logits).unwrap();
    let weight = backend
        .upload_f32(&(0..4096).map(|i| i as f32).collect::<Vec<_>>(), &[4096, 1])
        .unwrap();
    let embedded = backend.sampled_embedding_4096(&weight, &selected).unwrap();
    let collected = backend
        .concat_indices(&[selected.clone(), selected], 1)
        .unwrap();
    assert_eq!(backend.read_f32(&embedded).unwrap(), [7.0, 4095.0, 4095.0]);
    assert_eq!(
        backend.read_indices(&collected).unwrap(),
        [7, u32::MAX, u32::MAX, 7, u32::MAX, u32::MAX]
    );
}
