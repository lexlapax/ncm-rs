//! Controlled draws and invalid rows on real Metal.
#![cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use ncm_backend_mlx::MlxBackend;
use ncm_core::{CodecBackend, GeneratorBackend, WeightBackend};

#[test]
fn categorical_respects_boundaries_and_rejects_invalid_rows() {
    let backend = MlxBackend::gpu().unwrap();
    let mut logits = vec![0.0_f32; 9 * 4096];
    logits[2 * 4096..3 * 4096].fill(f32::NEG_INFINITY);
    logits[3 * 4096..4 * 4096].fill(f32::NEG_INFINITY);
    logits[3 * 4096 + 4095] = 0.0;
    logits[4 * 4096] = f32::NAN;
    logits[5 * 4096] = f32::INFINITY;
    logits[6 * 4096..9 * 4096].fill(f32::NEG_INFINITY);
    logits[6 * 4096] = 0.0;
    logits[7 * 4096] = 0.0;
    logits[7 * 4096 + 1] = 0.0;
    logits[8 * 4096] = 0.0;
    logits[8 * 4096 + 1] = 0.0;
    let logits = backend.upload_f32(&logits, &[9, 1, 4096]).unwrap();
    let draws = backend
        .upload_indices(
            &[0, u32::MAX, 0, 0, 1, 1, u32::MAX, 0x7fff_ffff, 0x8000_0000],
            &[9, 1],
        )
        .unwrap();
    let output = backend.categorical_4096(&logits, &draws).unwrap();
    assert_eq!(
        backend.read_indices(&output).unwrap(),
        [0, 4095, u32::MAX, 4095, u32::MAX, u32::MAX, 0, 0, 1]
    );
    let bad = backend.upload_indices(&[0; 9], &[1, 9]).unwrap();
    assert!(backend.categorical_4096(&logits, &bad).is_err());
}

#[test]
fn categorical_matches_independent_f64_for_controlled_draws() {
    let backend = MlxBackend::gpu().unwrap();
    let rows = 256;
    let mut state = 0x721c_14a9_u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let mut logits = Vec::with_capacity(rows * 4096);
    let mut uniforms = Vec::with_capacity(rows);
    let mut expected = Vec::with_capacity(rows);
    for _ in 0..rows {
        let row: Vec<f32> = (0..4096)
            .map(|_| (next() % 16001) as f32 / 1000.0 - 8.0)
            .collect();
        let bits = next();
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f64> = row
            .iter()
            .map(|&x| (f64::from(x) - f64::from(max)).exp())
            .collect();
        let threshold = f64::from(bits) * (1.0 / 4294967296.0) * weights.iter().sum::<f64>();
        let mut prefix = 0.0;
        let index = weights
            .iter()
            .position(|w| {
                prefix += w;
                threshold < prefix
            })
            .unwrap();
        logits.extend(row);
        uniforms.push(bits);
        expected.push(index as u32);
    }
    let logits = backend.upload_f32(&logits, &[4, 64, 4096]).unwrap();
    let uniforms = backend.upload_indices(&uniforms, &[4, 64]).unwrap();
    let actual = backend.categorical_4096(&logits, &uniforms).unwrap();
    assert_eq!(backend.read_indices(&actual).unwrap(), expected);
}

#[test]
fn sampled_embedding_bounds_invalid_sentinel_without_erasing_it() {
    let backend = MlxBackend::gpu().unwrap();
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
    let backend = MlxBackend::gpu().unwrap();
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
