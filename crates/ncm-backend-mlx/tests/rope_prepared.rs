#![cfg(feature = "mlx")]

use ncm_backend_mlx::MlxBackend;
use ncm_core::{CodecBackend, GeneratorBackend, WeightBackend};

#[test]
fn prepared_rotary_tables_match_existing_operation_on_metal() {
    let backend = MlxBackend::gpu().expect("real Metal required");
    for (batch, sequence, heads, width) in [(1, 5, 12, 64), (4, 17, 8, 64)] {
        let values: Vec<f32> = (0..batch * heads * sequence * width)
            .map(|i| ((i * 17 % 257) as f32 - 128.0) / 31.0)
            .collect();
        let cos: Vec<f32> = (0..sequence * width / 2)
            .map(|i| ((i * 7 % 127) as f32 / 127.0).cos())
            .collect();
        let sin: Vec<f32> = (0..sequence * width / 2)
            .map(|i| ((i * 7 % 127) as f32 / 127.0).sin())
            .collect();
        let x = backend
            .upload_f32(&values, &[batch, heads, sequence, width])
            .unwrap();
        let cos = backend.upload_f32(&cos, &[sequence, width / 2]).unwrap();
        let sin = backend.upload_f32(&sin, &[sequence, width / 2]).unwrap();
        let [prepared_cos, prepared_sin] = backend.prepare_rope_tables(&cos, &sin).unwrap();
        assert_eq!(backend.shape(&prepared_cos), [sequence, width]);
        assert_eq!(backend.shape(&prepared_sin), [sequence, width]);
        let reference = backend
            .read_f32(&backend.rope(&x, &cos, &sin).unwrap())
            .unwrap();
        for _ in 0..2 {
            let actual = backend
                .rope_prepared(&x, &prepared_cos, &prepared_sin)
                .unwrap();
            assert_eq!(backend.read_f32(&actual).unwrap(), reference);
        }
        assert!(backend.prepare_rope_tables(&cos, &x).is_err());
    }
}
