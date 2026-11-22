//! Regression tests for explicit `ComputePrecision` (`fp32`, `bf16`, `fp16`) on Metal.
#![cfg(feature = "mlx")]

use ncm_backend_mlx::MlxBackend;
use ncm_core::{CodecBackend, ComputePrecision, WeightBackend};

fn synthetic(len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 37 + 13) % 101) as f32 - 50.0) * (scale / 50.0))
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            assert!(
                x.is_finite() && y.is_finite(),
                "nonfinite value: {x} vs {y}"
            );
            (x - y).abs()
        })
        .fold(0.0f32, f32::max)
}

#[test]
fn mlx_fp16_rejects_out_of_range_weights() {
    let fp16 = MlxBackend::gpu()
        .expect("Metal GPU")
        .with_precision(ComputePrecision::Fp16)
        .expect("Metal FP16");
    let err = fp16.upload_f32(&[1.0, 70_000.0], &[1, 2]).unwrap_err();
    assert!(
        err.to_string().contains("FP16 finite range"),
        "unexpected error: {err}"
    );
    fp16.upload_f32(&[0.0, f32::NEG_INFINITY], &[1, 2])
        .expect("attention mask -inf allowed in FP16");
}

#[test]
fn mlx_bf16_and_fp16_operators_track_fp32() {
    let fp32 = MlxBackend::gpu().expect("Metal FP32");
    let x_data = synthetic(2 * 6 * 16, 1.5);
    let w_data = synthetic(32 * 16, 0.5);
    let b_data = synthetic(32, 0.25);
    let gamma_data: Vec<f32> = (0..16).map(|i| 1.0 + 0.02 * i as f32).collect();
    let beta_data = synthetic(16, 0.1);
    let qkv_data = synthetic(2 * 2 * 6 * 8, 1.0);
    let mask_data = vec![0.0f32; 6 * 6];

    let x32 = fp32.upload_f32(&x_data, &[2, 6, 16]).unwrap();
    let w32 = fp32.upload_f32(&w_data, &[32, 16]).unwrap();
    let b32 = fp32.upload_f32(&b_data, &[32]).unwrap();
    let g32 = fp32.upload_f32(&gamma_data, &[16]).unwrap();
    let beta32 = fp32.upload_f32(&beta_data, &[16]).unwrap();
    let q32 = fp32.upload_f32(&qkv_data, &[2, 2, 6, 8]).unwrap();
    let m32 = fp32.upload_f32(&mask_data, &[1, 1, 6, 6]).unwrap();

    let proj32 = fp32
        .read_f32(&fp32.project_bias(&x32, &w32, &b32).unwrap())
        .unwrap();
    let ln32 = fp32
        .read_f32(&fp32.layer_norm(&x32, &g32, Some(&beta32), 1e-5).unwrap())
        .unwrap();
    let rms32 = fp32
        .read_f32(&fp32.rms_norm(&x32, &g32, 1e-6).unwrap())
        .unwrap();
    let attn32 = fp32
        .read_f32(&fp32.attention(&q32, &q32, &q32, Some(&m32)).unwrap())
        .unwrap();

    for (precision, tol) in [
        (ComputePrecision::Bf16, 5e-2f32),
        (ComputePrecision::Fp16, 1e-2f32),
    ] {
        let be = MlxBackend::gpu()
            .unwrap()
            .with_precision(precision)
            .unwrap();
        assert_eq!(be.precision(), precision);

        let x = be.upload_f32(&x_data, &[2, 6, 16]).unwrap();
        let w = be.upload_f32(&w_data, &[32, 16]).unwrap();
        let b = be.upload_f32(&b_data, &[32]).unwrap();
        let g = be.upload_f32(&gamma_data, &[16]).unwrap();
        let beta = be.upload_f32(&beta_data, &[16]).unwrap();
        let q = be.upload_f32(&qkv_data, &[2, 2, 6, 8]).unwrap();
        let m = be.upload_f32(&mask_data, &[1, 1, 6, 6]).unwrap();

        let proj = be.project_bias(&x, &w, &b).unwrap();
        assert!(max_abs_diff(&proj32, &be.read_f32(&proj).unwrap()) <= tol);

        let ln = be.layer_norm(&x, &g, Some(&beta), 1e-5).unwrap();
        assert!(max_abs_diff(&ln32, &be.read_f32(&ln).unwrap()) <= tol);

        let rms = be.rms_norm(&x, &g, 1e-6).unwrap();
        assert!(max_abs_diff(&rms32, &be.read_f32(&rms).unwrap()) <= tol);

        let attn = be.attention(&q, &q, &q, Some(&m)).unwrap();
        assert!(max_abs_diff(&attn32, &be.read_f32(&attn).unwrap()) <= tol);
    }
}
