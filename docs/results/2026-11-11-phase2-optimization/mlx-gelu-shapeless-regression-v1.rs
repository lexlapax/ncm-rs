#![cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use ncm_backend_mlx::MlxBackend;
use ncm_core::{CodecBackend, WeightBackend};

#[test]
fn shapeless_gelu_matches_eager_across_shapes_views_and_clones() {
    let b = MlxBackend::gpu().unwrap();
    let clone = b.clone();
    let shapes: &[&[usize]] = &[
        &[1, 2, 3],
        &[3, 7, 33],
        &[1, 89, 1024],
        &[2, 89, 1024],
        &[4, 89, 1024],
        &[5, 3, 17],
        &[7, 13],
        &[19],
    ];
    for (iteration, shape) in shapes.iter().enumerate() {
        let count: usize = shape.iter().product();
        let mut values: Vec<f32> = (0..count)
            .map(|i| ((i * 73 % 1009) as f32 - 504.) / 80.)
            .collect();
        values[..5].copy_from_slice(&[0., -0., f32::INFINITY, f32::NEG_INFINITY, f32::NAN]);
        let x = b.upload_f32(&values, shape).unwrap();
        let x = if shape.len() == 3 && iteration % 2 == 1 {
            b.permute(&x, &[0, 2, 1]).unwrap()
        } else {
            x
        };
        b.set_compiled_gelu_for_diagnostic(false);
        let eager = b.read_f32(&b.gelu(&x).unwrap()).unwrap();
        clone.set_compiled_gelu_for_diagnostic(true);
        let compiled = clone.read_f32(&clone.gelu(&x).unwrap()).unwrap();
        assert_eq!(eager.len(), compiled.len());
        for (&a, &v) in eager.iter().zip(&compiled) {
            assert!(
                (a.is_nan() && v.is_nan())
                    || a.to_bits() == v.to_bits()
                    || (a.is_finite() && v.is_finite() && (a - v).abs() <= 1e-5 + 1e-4 * a.abs()),
                "shape={shape:?}: {a:?} vs {v:?}"
            );
        }
    }
}
