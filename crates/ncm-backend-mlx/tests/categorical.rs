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
