//! Exact-value regression for specialized and generic generator layouts.
#![cfg(feature = "cpu")]
use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, GeneratorBackend, WeightBackend};

fn check(backend: &CandleBackend, batch: usize, seq: usize) {
    const H: usize = 8;
    const D: usize = 64;
    let size = batch * seq * 3 * H * D;
    let data: Vec<f32> = (0..size)
        .map(|i| match i % 1021 {
            0 => f32::from_bits(0x8000_0000),
            1 => f32::from_bits(0x7fc0_1234),
            _ => i as f32 * 0.125,
        })
        .collect();
    let input = backend.upload_f32(&data, &[batch, seq, 3 * H * D]).unwrap();
    let packed = backend.pack_qkv(&input, batch, seq, H, D).unwrap();
    assert_eq!(backend.shape(&packed), [3, batch, H, seq, D]);
    let actual = backend.read_f32(&packed).unwrap();
    for q in 0..3 {
        for b in 0..batch {
            for h in 0..H {
                for t in 0..seq {
                    for d in 0..D {
                        let source = ((b * seq + t) * 3 * H + q * H + h) * D + d;
                        let output = (((q * batch + b) * H + h) * seq + t) * D + d;
                        assert_eq!(actual[output].to_bits(), data[source].to_bits());
                    }
                }
            }
        }
    }
    let q = backend
        .reshape(
            &backend.narrow(&packed, 0, 0, 1).unwrap(),
            &[batch, H, seq, D],
        )
        .unwrap();
    let merged = backend.merge_heads(&q, batch, seq, H, D).unwrap();
    let actual = backend.read_f32(&merged).unwrap();
    for b in 0..batch {
        for t in 0..seq {
            for h in 0..H {
                for d in 0..D {
                    let source = ((b * seq + t) * 3 * H + h) * D + d;
                    let output = ((b * seq + t) * H + h) * D + d;
                    assert_eq!(actual[output].to_bits(), data[source].to_bits());
                }
            }
        }
    }
    assert!(backend.pack_qkv(&input, batch, seq, 7, D).is_err());
    assert!(backend.merge_heads(&q, batch, seq, H, D - 1).is_err());
}
#[test]
fn cpu_generic_layout() {
    check(&CandleBackend::cpu(), 2, 7);
}
#[cfg(feature = "cuda")]
#[test]
fn cuda_specialized_layout_exact() {
    let backend = CandleBackend::cuda(0).unwrap();
    for (batch, seq) in [(1, 1), (2, 7), (3, 92), (4, 180), (128, 1)] {
        check(&backend, batch, seq);
    }
}

fn check_generic_heads(backend: &CandleBackend) {
    let (batch, seq, heads, head) = (2, 7, 3, 4);
    let values: Vec<f32> = (0..batch * seq * 3 * heads * head)
        .map(|v| v as f32)
        .collect();
    let input = backend
        .upload_f32(&values, &[batch, seq, 3 * heads * head])
        .unwrap();
    let packed = backend.pack_qkv(&input, batch, seq, heads, head).unwrap();
    let actual = backend.read_f32(&packed).unwrap();
    for q in 0..3 {
        for b in 0..batch {
            for h in 0..heads {
                for t in 0..seq {
                    for d in 0..head {
                        let source = ((b * seq + t) * 3 * heads + q * heads + h) * head + d;
                        let output = (((q * batch + b) * heads + h) * seq + t) * head + d;
                        assert_eq!(actual[output], values[source]);
                    }
                }
            }
        }
    }
    let q = backend
        .reshape(
            &backend.narrow(&packed, 0, 0, 1).unwrap(),
            &[batch, heads, seq, head],
        )
        .unwrap();
    let merged = backend.merge_heads(&q, batch, seq, heads, head).unwrap();
    let actual = backend.read_f32(&merged).unwrap();
    for b in 0..batch {
        for t in 0..seq {
            for h in 0..heads {
                for d in 0..head {
                    let source = ((b * seq + t) * 3 * heads + h) * head + d;
                    let output = ((b * seq + t) * heads + h) * head + d;
                    assert_eq!(actual[output], values[source]);
                }
            }
        }
    }
}

#[test]
fn cpu_other_head_layout() {
    check_generic_heads(&CandleBackend::cpu());
}
#[cfg(feature = "cuda")]
#[test]
fn cuda_other_head_layout_falls_back_on_device() {
    check_generic_heads(&CandleBackend::cuda(0).unwrap());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_specialized_layout_accepts_strided_input() {
    let backend = CandleBackend::cuda(0).unwrap();
    let (batch, seq, width) = (2, 7, 1536);
    let source: Vec<f32> = (0..batch * (seq + 1) * width).map(|x| x as f32).collect();
    let whole = backend
        .upload_f32(&source, &[batch, seq + 1, width])
        .unwrap();
    let view = backend.narrow(&whole, 1, 0, seq).unwrap();
    let packed = backend.pack_qkv(&view, batch, seq, 8, 64).unwrap();
    let actual = backend.read_f32(&packed).unwrap();
    for q in 0..3 {
        for b in 0..batch {
            for h in 0..8 {
                for t in 0..seq {
                    for d in 0..64 {
                        let source_index = ((b * (seq + 1) + t) * 24 + q * 8 + h) * 64 + d;
                        let output = (((q * batch + b) * 8 + h) * seq + t) * 64 + d;
                        assert_eq!(actual[output], source[source_index]);
                    }
                }
            }
        }
    }
}
