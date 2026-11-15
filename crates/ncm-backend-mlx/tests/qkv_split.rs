//! Verify generator Q/K/V split order and shape on a real device.
#![cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use ncm_backend_mlx::MlxBackend;
use ncm_core::{CodecBackend, GeneratorBackend, WeightBackend};

#[test]
fn split_qkv_preserves_each_head_and_position() {
    let backend = MlxBackend::gpu().unwrap();
    for batch in [2, 32] {
        check(&backend, batch);
    }
}

fn check(backend: &MlxBackend, batch: usize) {
    let (seq, heads, head) = (3, 2, 2);
    let width = heads * head;
    let values: Vec<f32> = (0..batch * seq * 3 * width).map(|x| x as f32).collect();
    let input = backend
        .upload_f32(&values, &[batch, seq, 3 * width])
        .unwrap();
    let parts = backend.split_qkv(&input, batch, seq, heads, head).unwrap();
    for (part, output) in parts.iter().enumerate() {
        let mut expected = Vec::new();
        for b in 0..batch {
            for h in 0..heads {
                for s in 0..seq {
                    for d in 0..head {
                        expected
                            .push(values[(b * seq + s) * 3 * width + part * width + h * head + d]);
                    }
                }
            }
        }
        assert_eq!(backend.shape(output), [batch, heads, seq, head]);
        assert_eq!(backend.read_f32(output).unwrap(), expected);
    }
    assert!(backend.split_qkv(&input, batch, seq, 3, head).is_err());
}
