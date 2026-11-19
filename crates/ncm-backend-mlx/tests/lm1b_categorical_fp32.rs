#![cfg(all(
    feature = "mlx",
    feature = "lm1b-experimental",
    target_os = "macos",
    target_arch = "aarch64"
))]
use ncm_backend_mlx::MlxBackend;
use ncm_core::{CodecBackend, GeneratorBackend, WeightBackend};

fn draw(backend: &MlxBackend, logits: &[f32], uniforms: &[f32]) -> (Vec<u32>, Vec<u32>) {
    let rows = uniforms.len();
    let logits = backend.upload_f32(logits, &[1, rows, 8192]).unwrap();
    let uniforms = backend.upload_f32(uniforms, &[1, rows]).unwrap();
    let (choices, invalid) = backend
        .lm1b_categorical_8192_fp32(&logits, &uniforms)
        .unwrap();
    assert_eq!(backend.indices_shape(&choices), [1, rows]);
    (
        backend.read_indices(&choices).unwrap(),
        backend.read_indices(&invalid).unwrap(),
    )
}

#[test]
fn uniform_rows_select_the_exact_inverse_cdf_index() {
    let backend = MlxBackend::gpu().unwrap();
    // Equal logits give probabilities of 1/8192, whose FP32 prefix sums are exact.
    let wanted = [0_u32, 100, 4095, 8191];
    let uniforms: Vec<f32> = wanted.iter().map(|&k| (k as f32 + 0.5) / 8192.0).collect();
    let logits = vec![0.0_f32; wanted.len() * 8192];
    let (choices, invalid) = draw(&backend, &logits, &uniforms);
    assert_eq!(choices, wanted);
    assert_eq!(invalid, [0]);
}

#[test]
fn dominant_and_zero_probability_tokens() {
    let backend = MlxBackend::gpu().unwrap();
    let mut logits = vec![-1000.0_f32; 3 * 8192];
    logits[7] = 50.0;
    for row in 1..3 {
        logits[row * 8192 + 10] = 0.0;
        logits[row * 8192 + 20] = 0.0;
    }
    let (choices, invalid) = draw(&backend, &logits, &[0.999, 0.25, 0.75]);
    assert_eq!(choices, [7, 10, 20]);
    assert_eq!(invalid, [0]);
}

#[test]
fn nonfinite_rows_are_counted_and_choices_stay_valid() {
    let backend = MlxBackend::gpu().unwrap();
    let mut logits = vec![0.0_f32; 3 * 8192];
    logits[8192 + 5] = f32::NAN;
    logits[2 * 8192 + 9] = f32::INFINITY;
    let (choices, invalid) = draw(&backend, &logits, &[0.5, 0.5, 0.5]);
    assert_eq!(invalid, [2]);
    assert!(choices.iter().all(|&choice| choice < 8192));
    assert_eq!(choices[0], 4096);
}

#[test]
fn mismatched_shapes_are_rejected() {
    let backend = MlxBackend::gpu().unwrap();
    let logits = backend
        .upload_f32(&vec![0.0; 2 * 8192], &[1, 2, 8192])
        .unwrap();
    let uniforms = backend.upload_f32(&[0.5], &[1, 1]).unwrap();
    assert!(
        backend
            .lm1b_categorical_8192_fp32(&logits, &uniforms)
            .is_err()
    );
    let narrow = backend.upload_f32(&vec![0.0; 4096], &[1, 1, 4096]).unwrap();
    assert!(
        backend
            .lm1b_categorical_8192_fp32(&narrow, &uniforms)
            .is_err()
    );
}
