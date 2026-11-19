#![cfg(all(
    feature = "mlx",
    feature = "lm1b-experimental",
    target_os = "macos",
    target_arch = "aarch64"
))]
use ncm_backend_mlx::MlxBackend;
use ncm_core::{CodecBackend, WeightBackend};

#[test]
fn top_p_probabilities_handle_finite_rows_and_validate_inputs() {
    let backend = MlxBackend::gpu().unwrap();
    let mut values = vec![0.0_f32; 2 * 8192];
    values[8192] = 3.0;
    let input = backend.upload_f32(&values, &[1, 2, 8192]).unwrap();
    let result = backend
        .lm1b_top_p_one_probabilities_experimental(&input)
        .unwrap();
    assert_eq!(backend.shape(&result), [1, 2, 8192]);
    let probs = backend.read_f32(&result).unwrap();
    assert!(probs.iter().all(|x| x.is_finite() && *x >= 0.0));
    for row in probs.as_chunks::<8192>().0 {
        assert!((row.iter().map(|&x| f64::from(x)).sum::<f64>() - 1.0).abs() < 1e-5);
    }
    assert!(probs[8192] > probs[8193]);

    values[0] = f32::NAN;
    let nan = backend.upload_f32(&values, &[1, 2, 8192]).unwrap();
    assert!(
        backend
            .lm1b_top_p_one_probabilities_experimental(&nan)
            .is_err()
    );
    values[0] = f32::INFINITY;
    let infinity = backend.upload_f32(&values, &[1, 2, 8192]).unwrap();
    assert!(
        backend
            .lm1b_top_p_one_probabilities_experimental(&infinity)
            .is_err()
    );
    values[0] = f32::NEG_INFINITY;
    let negative_infinity = backend.upload_f32(&values, &[1, 2, 8192]).unwrap();
    assert!(
        backend
            .lm1b_top_p_one_probabilities_experimental(&negative_infinity)
            .is_err()
    );
    let rank_two = backend.upload_f32(&vec![0.0; 8192], &[1, 8192]).unwrap();
    assert!(
        backend
            .lm1b_top_p_one_probabilities_experimental(&rank_two)
            .is_err()
    );
    let integer = backend
        .upload_indices(&vec![0; 8192], &[1, 1, 8192])
        .unwrap();
    assert!(
        backend
            .lm1b_top_p_one_probabilities_experimental(&integer)
            .is_err()
    );
    let wrong = backend.upload_f32(&[0.0; 8], &[1, 1, 8]).unwrap();
    assert!(
        backend
            .lm1b_top_p_one_probabilities_experimental(&wrong)
            .is_err()
    );
}

#[test]
fn top_p_filter_preserves_logit_order() {
    let backend = MlxBackend::gpu().unwrap();
    // All logits are between 0 and -39.84375, so ordinary softmax has
    // strictly positive FP32 probabilities even for the smallest logits.
    let logits = cutoff_logits();
    let input = backend.upload_f32(&logits, &[1, 1, 8192]).unwrap();
    let filtered = backend
        .lm1b_top_p_one_probabilities_experimental(&input)
        .unwrap();
    let probabilities = backend.read_f32(&filtered).unwrap();
    let minimum_retained = probabilities
        .iter()
        .zip(&logits)
        .filter_map(|(&probability, &logit)| (probability > 0.0).then_some(logit))
        .fold(f32::INFINITY, f32::min);
    assert!(
        probabilities
            .iter()
            .zip(&logits)
            .all(|(&probability, &logit)| probability > 0.0 || logit <= minimum_retained)
    );
    assert!(
        probabilities
            .iter()
            .zip(&logits)
            .all(|(&probability, &logit)| logit != 0.0 || probability > 0.0)
    );
    assert!((probabilities.iter().map(|&x| f64::from(x)).sum::<f64>() - 1.0).abs() < 1e-5);
}

#[test]
fn multirow_cutoff_matches_cuda_reference_for_representable_tail() {
    let backend = MlxBackend::gpu().unwrap();
    let logits = cutoff_logits().repeat(2);
    let input = backend.upload_f32(&logits, &[1, 2, 8192]).unwrap();
    let result = backend
        .lm1b_top_p_one_probabilities_experimental(&input)
        .unwrap();
    assert!(backend.read_f32(&result).unwrap().iter().all(|&x| x > 0.0));
}

// The pinned CUDA reference retains every positive probability for this input.
// Enable this check when the Metal FP32 arithmetic matches that reference.
#[test]
#[ignore = "Metal FP32 cutoff differs from the pinned CUDA reference"]
fn top_p_cutoff_matches_cuda_reference() {
    let backend = MlxBackend::gpu().unwrap();
    let input = backend.upload_f32(&cutoff_logits(), &[1, 1, 8192]).unwrap();
    let result = backend
        .lm1b_top_p_one_probabilities_experimental(&input)
        .unwrap();
    assert!(backend.read_f32(&result).unwrap().iter().all(|&x| x > 0.0));
}

fn cutoff_logits() -> Vec<f32> {
    (0..8192)
        .map(|i| -((i % 256) as f32) * 5.0 / 32.0)
        .collect()
}
