//! Finite maxima, tie policy and sentinel behavior on real Metal.
#![cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use ncm_backend_mlx::MlxBackend;
use ncm_core::{CodecBackend, GeneratorBackend, WeightBackend};

#[test]
fn xsum_top1_honors_ties_and_rejects_nonfinite_values() {
    let backend = MlxBackend::gpu().unwrap();
    let mut data = vec![-4.0_f32; 6 * 4096];
    data[17] = 3.0;
    data[4096 + 3] = 4.0;
    data[4096 + 2049] = 4.0;
    data[2 * 4096] = f32::NAN;
    data[3 * 4096 + 1] = f32::INFINITY;
    data[4 * 4096 + 2] = f32::NEG_INFINITY;
    data[5 * 4096..].fill(-1.0);
    data[5 * 4096 + 5] = -0.0;
    data[5 * 4096 + 9] = 0.0;
    let input = backend.upload_f32(&data, &[6, 1, 4096]).unwrap();
    let unique = backend.unique_max_4096(&input).unwrap();
    let reference = backend.xsum_reference_top1_4096(&input).unwrap();
    assert_eq!(
        backend.read_indices(&unique).unwrap(),
        [17, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX]
    );
    assert_eq!(
        backend.read_indices(&reference).unwrap(),
        [17, 3, u32::MAX, u32::MAX, u32::MAX, 5]
    );
    let wrong = backend.upload_f32(&[1.0; 32], &[1, 1, 32]).unwrap();
    assert!(backend.unique_max_4096(&wrong).is_err());
}
