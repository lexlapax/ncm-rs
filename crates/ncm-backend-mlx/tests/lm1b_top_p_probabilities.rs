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
