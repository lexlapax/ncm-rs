//! Real-device checks for the restricted XSum unique top-1 operation.
#![cfg(feature = "cuda")]
use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, GeneratorBackend, WeightBackend};

#[test]
fn unique_top1_rejects_ties_and_nonfinite_rows() {
    let backend = CandleBackend::cuda(0).unwrap();
    let mut scores = vec![-3.0f32; 8 * 4096];
    scores[301] = 2.0;
    scores[4096 + 4095] = 3.0;
    scores[2 * 4096 + 17] = 5.0;
    scores[2 * 4096 + 18] = 5.0;
    scores[3 * 4096 + 17] = 5.0;
    scores[3 * 4096 + 2000] = 5.0;
    scores[4 * 4096 + 2] = f32::NAN;
    scores[5 * 4096 + 2] = f32::INFINITY;
    scores[6 * 4096 + 2] = f32::NEG_INFINITY;
    scores[7 * 4096 + 16] = 1.0;
    scores[7 * 4096 + 17] = 1.0;
    scores[7 * 4096 + 18] = 2.0;
    let logits = backend.upload_f32(&scores, &[2, 4, 4096]).unwrap();
    let selected = backend.unique_max_4096(&logits).unwrap();
    assert_eq!(backend.indices_shape(&selected), [2, 4]);
    assert_eq!(
        backend.read_indices(&selected).unwrap(),
        [
            301,
            4095,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            18
        ]
    );
    let wrong = backend.upload_f32(&[1.0; 10], &[1, 1, 10]).unwrap();
    assert!(backend.unique_max_4096(&wrong).is_err());
}

#[cfg(feature = "cuda-cudnn")]
#[test]
fn reference_top1_selects_first_tie_and_rejects_nonfinite_rows() {
    let backend = CandleBackend::cuda_cudnn(0).unwrap();
    let reference = backend.xsum_reference_backend().unwrap();
    let mut scores = vec![-3.0f32; 8 * 4096];
    scores[301] = 2.0;
    scores[4096 + 4095] = 3.0;
    scores[2 * 4096 + 17] = 5.0;
    scores[2 * 4096 + 18] = 5.0;
    scores[3 * 4096 + 17] = 5.0;
    scores[3 * 4096 + 2000] = 5.0;
    scores[4 * 4096 + 2] = f32::NAN;
    scores[5 * 4096 + 2] = f32::INFINITY;
    scores[6 * 4096 + 2] = f32::NEG_INFINITY;
    scores[7 * 4096 + 16] = -0.0;
    scores[7 * 4096 + 2048] = 0.0;
    let logits = backend.upload_f32(&scores, &[2, 4, 4096]).unwrap();
    assert!(backend.xsum_reference_top1_4096(&logits).is_err());
    let selected = reference.xsum_reference_top1_4096(&logits).unwrap();
    assert_eq!(reference.indices_shape(&selected), [2, 4]);
    assert_eq!(
        reference.read_indices(&selected).unwrap(),
        [301, 4095, 17, 17, u32::MAX, u32::MAX, u32::MAX, 16]
    );
}
