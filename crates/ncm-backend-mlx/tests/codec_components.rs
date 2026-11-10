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
