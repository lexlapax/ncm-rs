//! Regression tests for explicit `ComputePrecision` (`fp32`, `bf16`, `fp16`) handling.
#![cfg(feature = "cpu")]

use ncm_backend_cuda::CandleBackend;
use ncm_core::{ComputePrecision, WeightBackend};

#[test]
fn cpu_rejects_half_precision_explicitly() {
    let fp32 = CandleBackend::cpu()
        .with_precision(ComputePrecision::Fp32)
        .expect("CPU FP32 must succeed");
    assert_eq!(fp32.precision(), ComputePrecision::Fp32);

    assert!(
        CandleBackend::cpu()
            .with_precision(ComputePrecision::Bf16)
            .is_err(),
        "CPU BF16 must fail explicitly"
    );
    assert!(
        CandleBackend::cpu()
            .with_precision(ComputePrecision::Fp16)
            .is_err(),
        "CPU FP16 must fail explicitly"
    );
    assert!(
        CandleBackend::cpu()
            .with_precision(ComputePrecision::Q8)
            .is_err(),
        "CPU Q8 must fail explicitly"
    );
    assert!(
        CandleBackend::cpu()
            .with_precision(ComputePrecision::Q4)
            .is_err(),
        "CPU Q4 must fail explicitly"
    );
}

#[cfg(feature = "cuda")]
mod cuda_precision {
    use candle_core::DType;
    use ncm_backend_cuda::CandleBackend;
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
    fn cuda_fp16_rejects_out_of_range_weights() {
        let fp16 = CandleBackend::cuda(0)
            .expect("CUDA device 0")
            .with_precision(ComputePrecision::Fp16)
            .expect("CUDA FP16");
        let err = fp16.upload_f32(&[1.0, 70_000.0], &[1, 2]).unwrap_err();
        assert!(
            err.to_string().contains("FP16 finite range"),
            "unexpected error: {err}"
        );
        // Intentional -inf mask entries remain allowed.
        fp16.upload_f32(&[0.0, f32::NEG_INFINITY], &[1, 2])
            .expect("attention mask -inf allowed in FP16");
    }

    #[test]
    fn cuda_bf16_and_fp16_operators_track_fp32() {
        let fp32 = CandleBackend::cuda(0).expect("CUDA FP32");
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

        for (precision, expected_dtype, tol) in [
            (ComputePrecision::Bf16, DType::BF16, 5e-2f32),
            (ComputePrecision::Fp16, DType::F16, 1e-2f32),
        ] {
            let be = CandleBackend::cuda(0)
                .unwrap()
                .with_precision(precision)
                .unwrap();
            assert_eq!(be.precision(), precision);

            let x = be.upload_f32(&x_data, &[2, 6, 16]).unwrap();
            assert_eq!(x.dtype(), expected_dtype);
            let w = be.upload_f32(&w_data, &[32, 16]).unwrap();
            let b = be.upload_f32(&b_data, &[32]).unwrap();
            let g = be.upload_f32(&gamma_data, &[16]).unwrap();
            let beta = be.upload_f32(&beta_data, &[16]).unwrap();
            let q = be.upload_f32(&qkv_data, &[2, 2, 6, 8]).unwrap();
            let m = be.upload_f32(&mask_data, &[1, 1, 6, 6]).unwrap();

            let proj = be.project_bias(&x, &w, &b).unwrap();
            assert_eq!(proj.dtype(), expected_dtype);
            assert!(max_abs_diff(&proj32, &be.read_f32(&proj).unwrap()) <= tol);

            let ln = be.layer_norm(&x, &g, Some(&beta), 1e-5).unwrap();
            assert_eq!(ln.dtype(), expected_dtype);
            assert!(max_abs_diff(&ln32, &be.read_f32(&ln).unwrap()) <= tol);

            let rms = be.rms_norm(&x, &g, 1e-6).unwrap();
            assert_eq!(rms.dtype(), expected_dtype);
            assert!(max_abs_diff(&rms32, &be.read_f32(&rms).unwrap()) <= tol);

            let attn = be.attention(&q, &q, &q, Some(&m)).unwrap();
            assert_eq!(attn.dtype(), expected_dtype);
            assert!(max_abs_diff(&attn32, &be.read_f32(&attn).unwrap()) <= tol);
        }
    }

    #[test]
    fn cuda_q8_and_q4_linear_projections_track_fp32() {
        let fp32 = CandleBackend::cuda(0).expect("CUDA FP32");
        let x_data = synthetic(2 * 4 * 64, 1.0);
        let w_data = synthetic(128 * 64, 0.25);
        let b_data = synthetic(128, 0.1);

        let x32 = fp32.upload_f32(&x_data, &[2, 4, 64]).unwrap();
        let w32 = fp32.upload_linear_weight_f32(&w_data, &[128, 64]).unwrap();
        let b32 = fp32.upload_f32(&b_data, &[128]).unwrap();
        let y32 = fp32
            .read_f32(&fp32.project_bias(&x32, &w32, &b32).unwrap())
            .unwrap();

        for (precision, tol) in [
            (ComputePrecision::Q8, 0.15f32),
            (ComputePrecision::Q4, 0.65f32),
        ] {
            let be = CandleBackend::cuda(0)
                .unwrap()
                .with_precision(precision)
                .unwrap();
            assert_eq!(be.precision(), precision);
            let x = be.upload_f32(&x_data, &[2, 4, 64]).unwrap();
            let w = be.upload_linear_weight_f32(&w_data, &[128, 64]).unwrap();
            assert_eq!(be.shape(&w), vec![128, 64]);
            let b = be.upload_f32(&b_data, &[128]).unwrap();
            let y = be.project_bias(&x, &w, &b).unwrap();
            assert_eq!(be.shape(&y), vec![2, 4, 128]);
            let diff = max_abs_diff(&y32, &be.read_f32(&y).unwrap());
            assert!(diff <= tol, "{precision} max_abs_diff={diff} > {tol}");
        }
    }
}
