#![cfg(feature = "cpu")]
#[path = "../../../tests/support/codec.rs"]
mod support;
use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, WeightBackend};

fn selection_and_snapshots(b: &CandleBackend) {
    let mut rows = vec![0f32; 2050];
    rows[3] = -7.0;
    rows[1024] = -7.0;
    rows[27] = 9.0;
    rows[1001] = 9.0;
    rows[1026] = -0.0;
    let x = b.upload_f32(&rows, &[2, 1025]).unwrap();
    assert_eq!(
        b.read_indices(&b.select_last(&x, false).unwrap()).unwrap(),
        [3, 0]
    );
    assert_eq!(
        b.read_indices(&b.select_last(&x, true).unwrap()).unwrap(),
        [27, 0]
    );
    let book = b.upload_f32(&[0., 0., 3., 0., 0., 4.], &[3, 2]).unwrap();
    let points = b
        .upload_f32(&[0.1, 0.2, 2.9, 0.1, 0., 3.9], &[3, 2])
        .unwrap();
    let scores = ncm_model::codebook_distances(b, &points, &book).unwrap();
    let ids = b.select_last(&scores, false).unwrap();
    assert_eq!(b.read_indices(&ids).unwrap(), [0, 1, 2]);
    assert_eq!(
        b.read_f32(&b.embedding(&book, &ids).unwrap()).unwrap(),
        [0., 0., 3., 0., 0., 4.]
    );
    let initial = b.upload_f32(&[1., 2.], &[1, 2]).unwrap();
    let first = initial.clone();
    let second = b.add(&first, &initial).unwrap();
    let third = b.add(&second, &initial).unwrap();
    b.evaluate(&[first.clone(), second.clone(), third.clone()])
        .unwrap();
    assert_eq!(b.read_f32(&first).unwrap(), [1., 2.]);
    assert_eq!(b.read_f32(&second).unwrap(), [2., 4.]);
    assert_eq!(b.read_f32(&third).unwrap(), [3., 6.]);
    let wrong_dtype = b.upload_indices(&[1, 2], &[1, 2]).unwrap();
    assert!(b.add(&wrong_dtype, &initial).is_err());
    assert!(b.read_indices(&initial).is_err());
}
fn batched_transpose(b: &CandleBackend) {
    let x = b.upload_f32(&[1., 2., 3., 4.], &[2, 1, 2]).unwrap();
    let w = b
        .upload_f32(&[1., 1., 0., 0., 0., 0., 1., 1.], &[2, 2, 2])
        .unwrap();
    let bias = b.upload_f32(&[0., 0.], &[2]).unwrap();
    assert!(b.conv1d(&x, &w, &bias, 0, 0, true).is_err());
    assert!(b.conv1d(&x, &w, &bias, 1, 0, false).is_err());
    assert!(b.conv1d(&x, &w, &bias, 1, usize::MAX, true).is_err());
    let y = b.conv1d(&x, &w, &bias, 2, 0, true).unwrap();
    assert_eq!(b.read_f32(&y).unwrap(), [1., 2., 1., 2., 3., 4., 3., 4.]);
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
fn batched_transpose_regression() {
    batched_transpose(&CandleBackend::cpu());
}
#[test]
fn cpu_codec_components() {
    let b = CandleBackend::cpu();
    selection_and_snapshots(&b);
    let report = support::components(b);
    assert!(report.tensors > 100 && report.elements > 10000);
    println!("CPU {report:?}");
}
#[cfg(feature = "cuda")]
#[test]
fn cuda_codec_components() {
    let b = CandleBackend::cuda(0).expect("real CUDA required, no fallback");
    selection_and_snapshots(&b);
    batched_transpose(&b);
    let foreign = CandleBackend::cpu().upload_f32(&[1.], &[1]).unwrap();
    assert!(b.square(&foreign).is_err());
    let report = support::components(b);
    assert!(report.tensors > 100 && report.elements > 10000);
    println!("CUDA {report:?}");
}
