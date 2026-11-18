//! Real-device ownership and failure checks for the opt-in cuDNN backend.
#![cfg(feature = "cuda-cudnn")]
use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, WeightBackend};

#[test]
fn strict_handle_can_move_and_drop_on_another_host_thread() {
    let backend = CandleBackend::cuda_cudnn(0).unwrap();
    assert!(backend.uses_strict_cudnn());
    assert!(backend.clone().uses_strict_cudnn());
    let x = backend.upload_f32(&[1.0; 8], &[1, 4, 2]).unwrap();
    let weight = backend.upload_f32(&[0.0; 12], &[2, 2, 3]).unwrap();
    let bias = backend.upload_f32(&[1.0, -1.0], &[2]).unwrap();
    assert!(backend.conv1d(&x, &weight, &bias, 0, 1, false).is_err());
    let y = backend.conv1d(&x, &weight, &bias, 1, 1, false).unwrap();
    assert_eq!(
        backend.read_f32(&y).unwrap(),
        [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    );
    drop((x, weight, bias, y));
    std::thread::spawn(move || drop(backend)).join().unwrap();
}

#[test]
fn strict_convolution_handles_forward_transpose_and_invalid_geometry() {
    let backend = CandleBackend::cuda_cudnn(0).unwrap();
    let bias = backend.upload_f32(&[0.0], &[1]).unwrap();
    let input = backend.upload_f32(&[1.0, 2.0, 3.0], &[1, 3, 1]).unwrap();
    let weights = backend.upload_f32(&[1.0, 2.0], &[1, 1, 2]).unwrap();
    let output = backend
        .conv1d(&input, &weights, &bias, 1, 0, false)
        .unwrap();
    assert_eq!(backend.read_f32(&output).unwrap(), [5.0, 8.0]);

    let input = backend.upload_f32(&[1.0, 2.0], &[1, 2, 1]).unwrap();
    let weights = backend.upload_f32(&[1.0, 2.0, 3.0], &[1, 1, 3]).unwrap();
    let output = backend.conv1d(&input, &weights, &bias, 2, 0, true).unwrap();
    assert_eq!(
        backend.read_f32(&output).unwrap(),
        [1.0, 2.0, 5.0, 4.0, 6.0]
    );
    assert!(backend.conv1d(&input, &weights, &bias, 1, 3, true).is_err());
    assert!(
        backend
            .conv1d(&input, &weights, &bias, 0, 0, false)
            .is_err()
    );
}

#[cfg(feature = "lm1b-experimental")]
#[test]
fn lm1b_default_math_mode_is_explicit_and_survives_clone() {
    let backend = CandleBackend::cuda_lm1b_cudnn_default_math_experimental(0).unwrap();
    assert!(!backend.uses_strict_cudnn());
    assert!(!backend.clone().uses_strict_cudnn());
    assert!(
        CandleBackend::cuda_lm1b_qkv_reference_experimental(0)
            .unwrap()
            .uses_strict_cudnn()
    );
}

#[cfg(feature = "lm1b-experimental")]
#[test]
fn lm1b_graph_reuses_workspace_across_shapes_and_drops_on_another_thread() {
    let backend = CandleBackend::cuda_lm1b_cudnn_graph_experimental(0).unwrap();
    assert!(!backend.uses_strict_cudnn());
    let mut kernel = vec![0.0; 384 * 384 * 3];
    for channel in 0..384 {
        kernel[(channel * 384 + channel) * 3 + 1] = 1.0;
    }
    let weight = backend.upload_f32(&kernel, &[384, 384, 3]).unwrap();
    let bias = backend.upload_f32(&[0.0; 384], &[384]).unwrap();

    // More than one cache's worth of shapes, followed by hits, across the
    // same stream. Defer readback so several launches share the workspace.
    let mut outputs = Vec::new();
    for width in (8..=24).chain([9, 24, 8]) {
        let values: Vec<_> = (0..width * 384).map(|i| (i % 13) as f32 - 6.0).collect();
        let input = backend.upload_f32(&values, &[1, width, 384]).unwrap();
        let output = backend.conv1d(&input, &weight, &bias, 1, 1, false).unwrap();
        outputs.push((values, output));
    }
    for (values, output) in outputs {
        assert_eq!(backend.read_f32(&output).unwrap(), values);
    }

    let clone = backend.clone();
    assert!(!clone.uses_strict_cudnn());
    let values = [1.0; 8 * 384];
    let input = clone.upload_f32(&values, &[1, 8, 384]).unwrap();
    let output = clone.conv1d(&input, &weight, &bias, 1, 1, false).unwrap();
    assert_eq!(clone.read_f32(&output).unwrap(), values);
    drop((input, weight, bias, output, clone));
    std::thread::spawn(move || drop(backend)).join().unwrap();
}
