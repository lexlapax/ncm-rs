#![cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
#[path = "../../../tests/support/codec.rs"]
mod support;
use ncm_backend_mlx::MlxBackend;
use ncm_core::{CodecBackend, WeightBackend};

#[test]
fn metal_codec_components() {
    // The backend must override and restore an explicitly CPU-scoped caller.
    mlx_rs::with_device(mlx_rs::Device::cpu(), || {
        let b = MlxBackend::gpu().expect("real Metal required; no fallback");
        let report = support::components(b.clone());
        assert!(report.tensors > 100 && report.elements > 10000);
        println!("Metal components {report:?}");
        let full = support::full_codec(b);
        println!("Metal complete codec {full:?}");
        assert!(matches!(
            mlx_rs::Stream::thread_local_or_default()
                .device()
                .unwrap()
                .get_type()
                .unwrap(),
            mlx_rs::DeviceType::Cpu
        ));
    });
}
#[test]
fn metal_selection_shapes_and_snapshots() {
    let b = MlxBackend::gpu().unwrap();
    let mut values = vec![0f32; 2050];
    values[3] = -7.;
    values[1024] = -7.;
    values[27] = 9.;
    values[1001] = 9.;
    values[1026] = -0.;
    let x = b.upload_f32(&values, &[2, 1025]).unwrap();
    assert_eq!(
        b.read_indices(&b.select_last(&x, false).unwrap()).unwrap(),
        [3, 0]
    );
    assert_eq!(
        b.read_indices(&b.select_last(&x, true).unwrap()).unwrap(),
        [27, 0]
    );
    let original = b.upload_f32(&[1., 2.], &[1, 2]).unwrap();
    let first = original.clone();
    let second = b.add(&first, &original).unwrap();
    let third = b.add(&second, &original).unwrap();
    b.evaluate(&[first.clone(), second.clone(), third.clone()])
        .unwrap();
    assert_eq!(b.read_f32(&first).unwrap(), [1., 2.]);
    assert_eq!(b.read_f32(&second).unwrap(), [2., 4.]);
    assert_eq!(b.read_f32(&third).unwrap(), [3., 6.]);
    let wrong = b.upload_indices(&[1, 2], &[1, 2]).unwrap();
    assert!(b.add(&wrong, &first).is_err());
    assert!(b.read_indices(&first).is_err());
    assert!(b.upload_f32(&[1.], &[2]).is_err());
    assert!(b.upload_indices(&[1], &[usize::MAX, 2]).is_err());
    assert!(b.narrow(&first, 1, 1, 2).is_err());
    assert!(b.narrow(&first, 3, 0, 1).is_err());
}
#[test]
fn metal_full_span_projection() {
    let b = MlxBackend::gpu().unwrap();
    let values: Vec<f32> = (0..128).map(|x| x as f32).collect();
    let x = b.upload_f32(&values, &[2, 4, 16]).unwrap();
    let weights: Vec<f32> = (0..128).map(|i| if i < 64 { 1. } else { -1. }).collect();
    let w = b.upload_f32(&weights, &[2, 16, 4]).unwrap();
    let bias = b.upload_f32(&[1., -1.], &[2]).unwrap();
    for stride in [1, 4, 7] {
        let y = b.conv1d(&x, &w, &bias, stride, 0, false).unwrap();
        assert_eq!(b.shape(&y), [2, 1, 2]);
        assert_eq!(b.read_f32(&y).unwrap(), [2017., -2017., 6113., -6113.]);
    }
}
#[test]
fn metal_batched_transpose() {
    let b = MlxBackend::gpu().unwrap();
    for length in [1, 4] {
        let values: Vec<f32> = (0..2 * length * 16).map(|x| x as f32).collect();
        let x = b.upload_f32(&values, &[2, length, 16]).unwrap();
        let mut weights = vec![0.; 16 * 16 * 4];
        for c in 0..16 {
            for k in 0..4 {
                weights[(c * 16 + c) * 4 + k] = 1.;
            }
        }
        let w = b.upload_f32(&weights, &[16, 16, 4]).unwrap();
        let bias = b.upload_f32(&[0.; 16], &[16]).unwrap();
        assert!(b.conv1d(&x, &w, &bias, 0, 0, true).is_err());
        assert!(b.conv1d(&x, &w, &bias, 1, usize::MAX, true).is_err());
        let y = b.conv1d(&x, &w, &bias, 4, 0, true).unwrap();
        let expected: Vec<f32> = values
            .as_chunks::<16>()
            .0
            .iter()
            .flat_map(|row| (0..4).flat_map(move |_| row.iter().copied()))
            .collect();
        assert_eq!(b.read_f32(&y).unwrap(), expected, "length={length}");
    }
}

#[test]
fn weighted_accelerated_norm_matches_existing_composition() {
    let b = MlxBackend::gpu().unwrap();
    let values: Vec<f32> = (0..4 * 7 * 33)
        .map(|i| ((i * 73 % 509) as f32 - 254.) / 37.)
        .collect();
    let x = b.upload_f32(&values, &[4, 7, 33]).unwrap();
    let weights: Vec<f32> = (0..33).map(|i| 0.5 + i as f32 / 33.).collect();
    let biases: Vec<f32> = (0..33).map(|i| (i as f32 - 16.) / 20.).collect();
    let w = b.upload_f32(&weights, &[33]).unwrap();
    let bias = b.upload_f32(&biases, &[33]).unwrap();
    for affine_bias in [None, Some(&bias)] {
        let fused = b.layer_norm_accelerated(&x, &w, affine_bias, 1e-6).unwrap();
        let reference = b.layer_norm(&x, &w, affine_bias, 1e-6).unwrap();
        let actual = b.read_f32(&fused).unwrap();
        let expected = b.read_f32(&reference).unwrap();
        for (&a, &e) in actual.iter().zip(&expected) {
            assert!((a - e).abs() <= 1e-5 + 1e-4 * e.abs());
        }
    }
}

#[test]
fn fused_projection_bias_matches_composition() {
    let b = MlxBackend::gpu().unwrap();
    let x: Vec<f32> = (0..4 * 7 * 17)
        .map(|i| ((i * 37 % 113) as f32 - 56.) / 19.)
        .collect();
    let weights: Vec<f32> = (0..13 * 17)
        .map(|i| ((i * 13 % 61) as f32 - 30.) / 23.)
        .collect();
    let bias: Vec<f32> = (0..13).map(|i| (i as f32 - 6.) / 7.).collect();
    let x = b.upload_f32(&x, &[4, 7, 17]).unwrap();
    let w = b.upload_f32(&weights, &[13, 17]).unwrap();
    let bias = b.upload_f32(&bias, &[13]).unwrap();
    let fused = b.project_bias(&x, &w, &bias).unwrap();
    let separate = b.add(&b.project(&x, &w).unwrap(), &bias).unwrap();
    for (a, e) in b
        .read_f32(&fused)
        .unwrap()
        .iter()
        .zip(b.read_f32(&separate).unwrap())
    {
        assert!((a - e).abs() <= 1e-5 + 1e-4 * e.abs());
    }

    // The fallback accepts the same broadcast shapes as the original add.
    let broadcast_bias = b.upload_f32(&[1.; 13], &[1, 1, 13]).unwrap();
    let fallback = b.project_bias(&x, &w, &broadcast_bias).unwrap();
    let expected = b.add(&b.project(&x, &w).unwrap(), &broadcast_bias).unwrap();
    assert_eq!(b.shape(&fallback), b.shape(&expected));
    assert_eq!(
        b.read_f32(&fallback).unwrap(),
        b.read_f32(&expected).unwrap()
    );

    let vector = b.upload_f32(&[0.25; 17], &[17]).unwrap();
    let vector_fused = b.project_bias(&vector, &w, &bias).unwrap();
    let vector_expected = b.add(&b.project(&vector, &w).unwrap(), &bias).unwrap();
    assert_eq!(b.shape(&vector_fused), [13]);
    for (a, e) in b
        .read_f32(&vector_fused)
        .unwrap()
        .iter()
        .zip(b.read_f32(&vector_expected).unwrap())
    {
        assert!((a - e).abs() <= 1e-5 + 1e-4 * e.abs());
    }

    let wrong_width = b.upload_f32(&[0.; 16], &[16]).unwrap();
    let wrong_type = b.upload_indices(&[0; 13], &[13]).unwrap();
    assert!(b.project_bias(&wrong_width, &w, &bias).is_err());
    assert!(b.project_bias(&x, &w, &wrong_type).is_err());
}

#[test]
fn weighted_rms_norm_matches_broadcast_fallback() {
    let b = MlxBackend::gpu().unwrap();
    let values: Vec<f32> = (0..2 * 3 * 17)
        .map(|i| ((i * 31 % 127) as f32 - 63.) / 19.)
        .collect();
    let weights: Vec<f32> = (0..17).map(|i| 0.5 + i as f32 / 17.).collect();
    let x = b.upload_f32(&values, &[2, 3, 17]).unwrap();
    let vector = b.upload_f32(&weights, &[17]).unwrap();
    let broadcast = b.upload_f32(&weights, &[1, 1, 17]).unwrap();
    let fused = b.rms_norm(&x, &vector, 1e-6).unwrap();
    let reference = b.rms_norm(&x, &broadcast, 1e-6).unwrap();
    for (a, e) in b
        .read_f32(&fused)
        .unwrap()
        .iter()
        .zip(b.read_f32(&reference).unwrap())
    {
        assert!((a - e).abs() <= 1e-5 + 1e-4 * e.abs());
    }
    assert!(b.rms_norm(&x, &vector, f64::NAN).is_err());
    let wrong_type = b.upload_indices(&[0; 17], &[17]).unwrap();
    assert!(b.rms_norm(&x, &wrong_type, 1e-6).is_err());
}
