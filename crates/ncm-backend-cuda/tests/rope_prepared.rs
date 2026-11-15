#![cfg(feature = "cpu")]

use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, GeneratorBackend, WeightBackend};

#[test]
fn default_prepared_rotary_tables_preserve_the_existing_operation() {
    let backend = CandleBackend::cpu();
    let x = backend
        .upload_f32(
            &(0..48).map(|i| i as f32 / 17.0).collect::<Vec<_>>(),
            &[1, 2, 3, 8],
        )
        .unwrap();
    let cos = backend.upload_f32(&[0.8; 12], &[3, 4]).unwrap();
    let sin = backend.upload_f32(&[0.6; 12], &[3, 4]).unwrap();
    let [prepared_cos, prepared_sin] = backend.prepare_rope_tables(&cos, &sin).unwrap();
    assert_eq!(backend.shape(&prepared_cos), [3, 4]);
    let old = backend.rope(&x, &cos, &sin).unwrap();
    let new = backend
        .rope_prepared(&x, &prepared_cos, &prepared_sin)
        .unwrap();
    assert_eq!(
        backend.read_f32(&old).unwrap(),
        backend.read_f32(&new).unwrap()
    );
}
