//! Regression tests for explicit `ComputePrecision` (`fp32`, `bf16`, `fp16`) on Metal.
#![cfg(feature = "mlx")]

use ncm_backend_mlx::MlxBackend;
use ncm_core::{CodecBackend, ComputePrecision, GeneratorBackend, WeightBackend};

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
    let err = match fp16.upload_f32(&[1.0, 70_000.0], &[1, 2]) {
        Err(err) => err,
        Ok(_) => panic!("expected FP16 finite-range rejection"),
    };
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
    let ln_nb32 = fp32
        .read_f32(&fp32.layer_norm(&x32, &g32, None, 1e-5).unwrap())
        .unwrap();
    let rms32 = fp32
        .read_f32(&fp32.rms_norm(&x32, &g32, 1e-6).unwrap())
        .unwrap();
    let gn32 = fp32
        .read_f32(&fp32.group_norm(&x32, 4, 1e-5).unwrap())
        .unwrap();
    let gelu32 = fp32.read_f32(&fp32.gelu(&x32).unwrap()).unwrap();
    let attn32 = fp32
        .read_f32(&fp32.attention(&q32, &q32, &q32, Some(&m32)).unwrap())
        .unwrap();
    let cos_data: Vec<f32> = (0..6 * 4).map(|i| (i as f32 * 0.1).cos()).collect();
    let sin_data: Vec<f32> = (0..6 * 4).map(|i| (i as f32 * 0.1).sin()).collect();
    let cos32 = fp32.upload_f32(&cos_data, &[6, 4]).unwrap();
    let sin32 = fp32.upload_f32(&sin_data, &[6, 4]).unwrap();
    let rope32 = fp32
        .read_f32(&fp32.rope(&q32, &cos32, &sin32).unwrap())
        .unwrap();
    // Fused width-64 QK RMSNorm + RoPE path (used by all released NCM generators).
    let q64_data = synthetic(2 * 2 * 6 * 64, 1.0);
    let g64_data: Vec<f32> = (0..64).map(|i| 1.0 + 0.01 * i as f32).collect();
    let cos64_data: Vec<f32> = (0..6 * 32).map(|i| (i as f32 * 0.07).cos()).collect();
    let sin64_data: Vec<f32> = (0..6 * 32).map(|i| (i as f32 * 0.07).sin()).collect();
    let q64_32 = fp32.upload_f32(&q64_data, &[2, 2, 6, 64]).unwrap();
    let g64_32 = fp32.upload_f32(&g64_data, &[64]).unwrap();
    let cos64_32 = fp32.upload_f32(&cos64_data, &[6, 32]).unwrap();
    let sin64_32 = fp32.upload_f32(&sin64_data, &[6, 32]).unwrap();
    let rms_rope32 = fp32
        .read_f32(
            &fp32
                .rms_rope(&q64_32, &g64_32, &cos64_32, &sin64_32, 1e-6)
                .unwrap(),
        )
        .unwrap();
    let lin_up32 = fp32.read_f32(&fp32.linear1d(&x32, 12).unwrap()).unwrap();
    let area_dn32 = fp32.read_f32(&fp32.area1d(&x32, 4).unwrap()).unwrap();
    let x32_b1 = fp32.upload_f32(&x_data[..6 * 16], &[1, 6, 16]).unwrap();
    let lin_b1_32 = fp32.read_f32(&fp32.linear1d(&x32_b1, 12).unwrap()).unwrap();
    let area_b1_32 = fp32.read_f32(&fp32.area1d(&x32_b1, 4).unwrap()).unwrap();

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
        let cos = be.upload_f32(&cos_data, &[6, 4]).unwrap();
        let sin = be.upload_f32(&sin_data, &[6, 4]).unwrap();
        let q64 = be.upload_f32(&q64_data, &[2, 2, 6, 64]).unwrap();
        let g64 = be.upload_f32(&g64_data, &[64]).unwrap();
        let cos64 = be.upload_f32(&cos64_data, &[6, 32]).unwrap();
        let sin64 = be.upload_f32(&sin64_data, &[6, 32]).unwrap();
        let x_b1 = be.upload_f32(&x_data[..6 * 16], &[1, 6, 16]).unwrap();

        let proj = be.project_bias(&x, &w, &b).unwrap();
        assert!(max_abs_diff(&proj32, &be.read_f32(&proj).unwrap()) <= tol);

        let ln = be.layer_norm(&x, &g, Some(&beta), 1e-5).unwrap();
        assert!(max_abs_diff(&ln32, &be.read_f32(&ln).unwrap()) <= tol);

        let ln_nb = be.layer_norm(&x, &g, None, 1e-5).unwrap();
        assert!(max_abs_diff(&ln_nb32, &be.read_f32(&ln_nb).unwrap()) <= tol);

        let rms = be.rms_norm(&x, &g, 1e-6).unwrap();
        assert!(max_abs_diff(&rms32, &be.read_f32(&rms).unwrap()) <= tol);

        let gn = be.group_norm(&x, 4, 1e-5).unwrap();
        assert!(max_abs_diff(&gn32, &be.read_f32(&gn).unwrap()) <= tol);

        let gelu = be.gelu(&x).unwrap();
        assert!(max_abs_diff(&gelu32, &be.read_f32(&gelu).unwrap()) <= tol);

        let attn = be.attention(&q, &q, &q, Some(&m)).unwrap();
        assert!(max_abs_diff(&attn32, &be.read_f32(&attn).unwrap()) <= tol);

        let rope = be.rope(&q, &cos, &sin).unwrap();
        assert!(max_abs_diff(&rope32, &be.read_f32(&rope).unwrap()) <= tol);

        let rms_rope = be.rms_rope(&q64, &g64, &cos64, &sin64, 1e-6).unwrap();
        assert!(max_abs_diff(&rms_rope32, &be.read_f32(&rms_rope).unwrap()) <= tol);

        let lin_up = be.linear1d(&x, 12).unwrap();
        assert!(max_abs_diff(&lin_up32, &be.read_f32(&lin_up).unwrap()) <= tol);

        let area_dn = be.area1d(&x, 4).unwrap();
        assert!(max_abs_diff(&area_dn32, &be.read_f32(&area_dn).unwrap()) <= tol);

        let lin_b1 = be.linear1d(&x_b1, 12).unwrap();
        assert!(max_abs_diff(&lin_b1_32, &be.read_f32(&lin_b1).unwrap()) <= tol);

        let area_b1 = be.area1d(&x_b1, 4).unwrap();
        assert!(max_abs_diff(&area_b1_32, &be.read_f32(&area_b1).unwrap()) <= tol);
    }
}

#[test]
fn mlx_q8_and_q4_linear_projections_track_fp32() {
    let fp32 = MlxBackend::gpu().expect("Metal FP32");
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
        let be = MlxBackend::gpu()
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

#[test]
fn mlx_prequant_upload_matches_on_the_fly_bit_exactly() {
    use ncm_core::{PrequantFormat, PrequantPayload};
    let x_data = synthetic(2 * 4 * 64, 1.0);
    let w_data = synthetic(128 * 64, 0.25);
    let b_data = synthetic(128, 0.1);
    let b_le: Vec<u8> = b_data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let b_bf16 = ncm_io::f32_bytes_to_bf16_bytes(&b_le).unwrap();

    for precision in [ComputePrecision::Q8, ComputePrecision::Q4] {
        let be = MlxBackend::gpu()
            .unwrap()
            .with_precision(precision)
            .unwrap();
        assert_eq!(be.prequant_format(), Some(PrequantFormat::Mlx));

        let x = be.upload_f32(&x_data, &[2, 4, 64]).unwrap();
        let w_otf = be.upload_linear_weight_f32(&w_data, &[128, 64]).unwrap();
        let b_otf = be.upload_f32(&b_data, &[128]).unwrap();
        let y_otf = be
            .read_f32(&be.project_bias(&x, &w_otf, &b_otf).unwrap())
            .unwrap();

        let raw = be
            .quantize_linear_raw(&w_data, &[128, 64])
            .unwrap()
            .expect("shape [128, 64] must quantize");
        let w_pq = be
            .upload_prequant(
                PrequantPayload::MlxAffine {
                    precision,
                    group_size: 64,
                    weights_u32: &raw.weights_u32,
                    weights_shape: &raw.weights_shape,
                    scales_bf16: &raw.scales_bf16,
                    scales_shape: &raw.scales_shape,
                    biases_bf16: &raw.biases_bf16,
                    biases_shape: &raw.biases_shape,
                },
                &[128, 64],
            )
            .unwrap();
        let b_pq = be
            .upload_prequant(PrequantPayload::Bf16(&b_bf16), &[128])
            .unwrap();
        let y_pq = be
            .read_f32(&be.project_bias(&x, &w_pq, &b_pq).unwrap())
            .unwrap();

        assert_eq!(
            y_otf, y_pq,
            "{precision} prequant output must be bit-identical"
        );
    }
}
