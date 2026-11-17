//! Real-CUDA parity of the explicit WMT14 first-maximum diagnostic.
#![cfg(feature = "cuda")]
use ncm_backend_cuda::{CandleBackend, wmt14_argmax};
use ncm_core::{CodecBackend, WeightBackend};

#[test]
fn first_max_matches_generic_with_ties_and_nonfinite_values() {
    const WIDTH: usize = 151_680;
    let backend = CandleBackend::cuda(0).unwrap();
    let mut values = vec![-7.0f32; 64 * WIDTH];
    values[WIDTH - 1] = 8.0;
    values[WIDTH + 3] = 5.0;
    values[WIDTH + WIDTH - 1] = 5.0;
    values[2 * WIDTH + 12] = f32::NAN;
    values[2 * WIDTH + 2000] = 11.0;
    values[3 * WIDTH + 7] = f32::INFINITY;
    values[3 * WIDTH + 300] = f32::INFINITY;
    values[4 * WIDTH..5 * WIDTH].fill(f32::NEG_INFINITY);
    values[5 * WIDTH..6 * WIDTH].fill(f32::NAN);
    values[6 * WIDTH + 7] = -0.0;
    values[6 * WIDTH + 2048] = 0.0;
    values[7 * WIDTH..8 * WIDTH].fill(f32::NAN);
    values[7 * WIDTH + 257] = f32::NEG_INFINITY;
    for row in 8..64 {
        values[row * WIDTH + (row * 2171 % WIDTH)] = (row as f32) / 7.0;
    }
    let logits = backend.upload_f32(&values, &[1, 64, WIDTH]).unwrap();
    let generic = backend.select_last(&logits, true).unwrap();
    let candidate = wmt14_argmax::select(&backend, &logits).unwrap();
    let expected = backend.read_indices(&generic).unwrap();
    assert_eq!(backend.read_indices(&candidate).unwrap(), expected);
    assert_eq!(
        &expected[..8],
        &[WIDTH as u32 - 1, 3, 2000, 7, 0, u32::MAX, 7, 257]
    );
    #[cfg(feature = "cuda-cudnn")]
    {
        let candidate = CandleBackend::cuda_wmt14_fast_argmax_experimental(0).unwrap();
        let candidate_logits = candidate.upload_f32(&values, &[1, 64, WIDTH]).unwrap();
        let cloned = candidate.clone();
        assert_eq!(
            cloned
                .read_indices(&cloned.select_last(&candidate_logits, true).unwrap())
                .unwrap(),
            expected
        );
    }
    let wrong = backend.upload_f32(&[0.0; 10], &[1, 1, 10]).unwrap();
    assert!(wmt14_argmax::select(&backend, &wrong).is_err());
}
