#![cfg(feature = "cpu")]
use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, WeightBackend};

#[test]
fn row_visits_cover_every_value_in_order() {
    let backend = CandleBackend::cpu();
    let mut values: Vec<f32> = (0..5 * 3 * 2).map(|x| x as f32 - 7.5).collect();
    values[3] = -0.0;
    values[11] = f32::NAN;
    let tensor = backend.upload_f32(&values, &[5, 3, 2]).unwrap();
    let blocks = backend
        .visit_f32_rows(&tensor, 2, |block| block.to_vec())
        .unwrap();
    assert_eq!(blocks.iter().map(Vec::len).collect::<Vec<_>>(), [12, 12, 6]);
    let joined: Vec<u32> = blocks.concat().iter().map(|x| x.to_bits()).collect();
    let direct: Vec<u32> = values.iter().map(|x| x.to_bits()).collect();
    assert_eq!(joined, direct);
    assert!(backend.visit_f32_rows(&tensor, 0, |_| ()).is_err());
}
