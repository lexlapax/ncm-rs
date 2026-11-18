//! Compare fused long-attention softmax with the existing CUDA operation sequence.
#![cfg(feature = "cuda")]
use ncm_backend_cuda::{CandleBackend, masked_softmax};
use ncm_core::{CodecBackend, WeightBackend};

#[test]
fn fused_masked_softmax_matches_separate_cuda_operations() {
    let backend = CandleBackend::cuda(0).unwrap();
    for (batch, heads, seq, mask_batch) in [
        (1, 2, 512, 1),
        (2, 2, 516, 2),
        (2, 3, 544, 2),
        (2, 1, 545, 2),
        (2, 2, 560, 1),
        (1, 1, 561, 1),
    ] {
        let count = batch * heads * seq * seq;
        let mut state = 0x87e1_369bu32;
        let scores: Vec<f32> = (0..count)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state as i32 as f32) / 134_217_728.0
            })
            .collect();
        let mut masks = vec![0.0; mask_batch * seq * seq];
        for row in 0..mask_batch * seq {
            for col in 0..seq {
                if (col + row * 7) % 11 == 0 {
                    masks[row * seq + col] = f32::NEG_INFINITY;
                }
            }
        }
        masks[..seq].fill(f32::NEG_INFINITY);
        let input = backend
            .upload_f32(&scores, &[batch, heads, seq, seq])
            .unwrap();
        let mask = backend
            .upload_f32(&masks, &[mask_batch, 1, seq, seq])
            .unwrap();
        let scaled = (&input / 8.0).unwrap();
        let expected =
            candle_nn::ops::softmax_last_dim(&scaled.broadcast_add(&mask).unwrap()).unwrap();
        let actual = masked_softmax::apply(&backend, &input, &mask).unwrap();
        let expected = backend.read_f32(&expected).unwrap();
        let actual = backend.read_f32(&actual).unwrap();
        let differences = actual
            .iter()
            .zip(&expected)
            .filter(|(a, b)| a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan()))
            .count();
        assert_eq!(
            differences, 0,
            "batch {batch} heads {heads} seq {seq} mask batch {mask_batch}"
        );
    }
}

#[test]
fn fused_long_attention_matches_separate_cuda_operations() {
    let backend = CandleBackend::cuda(0).unwrap();
    let (batch, heads, seq, width) = (2, 2, 516, 64);
    let count = batch * heads * seq * width;
    let values: Vec<f32> = (0..count)
        .map(|i| ((i * 43 % 997) as f32 - 498.0) / 997.0)
        .collect();
    let q = backend
        .upload_f32(&values, &[batch, heads, seq, width])
        .unwrap();
    let k = backend
        .upload_f32(
            &values.iter().rev().copied().collect::<Vec<_>>(),
            &[batch, heads, seq, width],
        )
        .unwrap();
    let v = backend
        .upload_f32(&values, &[batch, heads, seq, width])
        .unwrap();
    let mut masks = vec![0.0; batch * seq * seq];
    for row in 0..batch * seq {
        for col in 0..seq {
            if (row + col * 3) % 17 == 0 {
                masks[row * seq + col] = f32::NEG_INFINITY;
            }
        }
    }
    let mask = backend.upload_f32(&masks, &[batch, 1, seq, seq]).unwrap();
    let scores = (&q.matmul(&k.transpose(2, 3).unwrap()).unwrap() / 8.0).unwrap();
    let expected = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(&mask).unwrap())
        .unwrap()
        .matmul(&v)
        .unwrap();
    let actual = backend.attention(&q, &k, &v, Some(&mask)).unwrap();
    let expected = backend.read_f32(&expected).unwrap();
    let actual = backend.read_f32(&actual).unwrap();
    assert!(
        actual
            .iter()
            .zip(&expected)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
}
