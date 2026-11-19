#![cfg(feature = "cpu")]
//! Synthetic operator coverage at the pinned OWT configuration's extreme shapes.
//!
//! OWT has no released checkpoint. These tests exercise the shared backend operators at the
//! shapes OWT would need: a 32-level squares ladder ending at 1,024 positions, one prefix token,
//! 16 heads of 64, and rotary base 100,000. Each operator is compared with a naive host reference.
use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, GeneratorBackend, WeightBackend};

const LATENT: usize = 1024;
const HEADS: usize = 16;
const HEAD: usize = 64;
const ROPE_BASE: f32 = 100_000.0;

/// Deterministic values in [-1, 1).
fn values(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn area_reference(x: &[f32], length: usize, channels: usize, out: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(out * channels);
    for i in 0..out {
        let start = i * length / out;
        let end = ((i + 1) * length).div_ceil(out);
        for c in 0..channels {
            let sum: f64 = (start..end).map(|j| f64::from(x[j * channels + c])).sum();
            result.push((sum / (end - start) as f64) as f32);
        }
    }
    result
}

fn linear_reference(x: &[f32], length: usize, channels: usize, out: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(out * channels);
    for i in 0..out {
        let source = ((i as f64 + 0.5) * length as f64 / out as f64 - 0.5).max(0.0);
        let low = (source.floor() as usize).min(length - 1);
        let high = (low + 1).min(length - 1);
        let weight = source - low as f64;
        for c in 0..channels {
            let a = f64::from(x[low * channels + c]);
            let b = f64::from(x[high * channels + c]);
            result.push((a + (b - a) * weight) as f32);
        }
    }
    result
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what} length");
    let worst = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0_f32, f32::max);
    assert!(worst <= tolerance, "{what}: max error {worst}");
}

#[test]
fn owt_ladder_resampling_matches_naive_area_and_linear() {
    let backend = CandleBackend::cpu();
    let levels: Vec<usize> = (1..=32).map(|n| n * n).collect();
    assert_eq!((levels[0], levels[31]), (1, LATENT));
    let channels = 3;
    let latent = values(LATENT * channels, 1);
    let x = backend.upload_f32(&latent, &[1, LATENT, channels]).unwrap();
    for &level in &[1, 961] {
        let down = backend
            .read_f32(&backend.area1d(&x, level).unwrap())
            .unwrap();
        assert_close(
            &down,
            &area_reference(&latent, LATENT, channels, level),
            1e-6,
            "area1d",
        );
        let small = backend.upload_f32(&down, &[1, level, channels]).unwrap();
        let up = backend
            .read_f32(&backend.linear1d(&small, LATENT).unwrap())
            .unwrap();
        assert_close(
            &up,
            &linear_reference(&down, level, channels, LATENT),
            1e-6,
            "linear1d",
        );
    }
}

/// Positions are the prefix token (0) followed by 0..length, as the generator supplies them.
fn rope_tables(length: usize) -> (Vec<f32>, Vec<f32>) {
    let (mut cosine, mut sine) = (Vec::new(), Vec::new());
    for pos in std::iter::once(0).chain(0..length) {
        for i in 0..HEAD / 2 {
            let angle = pos as f32 / ROPE_BASE.powf(i as f32 / (HEAD / 2) as f32);
            cosine.push(angle.cos());
            sine.push(angle.sin());
        }
    }
    (cosine, sine)
}

#[test]
fn owt_top_level_rotary_and_masked_attention_match_naive() {
    let backend = CandleBackend::cpu();
    let seq = 1 + LATENT;
    let count = HEADS * seq * HEAD;
    let (cosine, sine) = rope_tables(LATENT);
    let cos = backend.upload_f32(&cosine, &[seq, HEAD / 2]).unwrap();
    let sin = backend.upload_f32(&sine, &[seq, HEAD / 2]).unwrap();

    let raw = values(count, 2);
    let x = backend.upload_f32(&raw, &[1, HEADS, seq, HEAD]).unwrap();
    let rotated = backend
        .read_f32(&backend.rope(&x, &cos, &sin).unwrap())
        .unwrap();
    let mut expected = raw.clone();
    for row in 0..HEADS * seq {
        let pos = row % seq;
        for d in 0..HEAD / 2 {
            let (a, b) = (raw[row * HEAD + d], raw[row * HEAD + d + HEAD / 2]);
            let (c, s) = (cosine[pos * HEAD / 2 + d], sine[pos * HEAD / 2 + d]);
            expected[row * HEAD + d] = a * c - b * s;
            expected[row * HEAD + d + HEAD / 2] = b * c + a * s;
        }
    }
    assert_close(&rotated, &expected, 1e-6, "rope");

    // The prefix row may not attend to targets; targets see everything (block-diagonal per level).
    let mut mask = vec![0.0_f32; seq * seq];
    mask[1..seq].fill(f32::NEG_INFINITY);
    let mask_tensor = backend.upload_f32(&mask, &[1, 1, seq, seq]).unwrap();
    let (q, k, v) = (values(count, 3), values(count, 4), values(count, 5));
    let upload = |data: &[f32]| backend.upload_f32(data, &[1, HEADS, seq, HEAD]).unwrap();
    let output = backend
        .read_f32(
            &backend
                .attention(&upload(&q), &upload(&k), &upload(&v), Some(&mask_tensor))
                .unwrap(),
        )
        .unwrap();
    assert_eq!(output.len(), count);
    assert!(output.iter().all(|x| x.is_finite()));
    let scale = 1.0 / (HEAD as f64).sqrt();
    for head in [0, HEADS - 1] {
        let base = head * seq * HEAD;
        for row in (0..seq).step_by(97).chain([seq - 1]) {
            let scores: Vec<f64> = (0..seq)
                .map(|col| {
                    let dot: f64 = (0..HEAD)
                        .map(|d| {
                            f64::from(q[base + row * HEAD + d])
                                * f64::from(k[base + col * HEAD + d])
                        })
                        .sum();
                    dot * scale + f64::from(mask[row * seq + col])
                })
                .collect();
            let peak = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - peak).exp()).collect();
            let total: f64 = weights.iter().sum();
            for d in 0..HEAD {
                let value: f64 = (0..seq)
                    .map(|col| weights[col] * f64::from(v[base + col * HEAD + d]))
                    .sum::<f64>()
                    / total;
                let actual = f64::from(output[base + row * HEAD + d]);
                assert!(
                    (actual - value).abs() < 1e-5,
                    "attention head {head} row {row} dim {d}: {actual} vs {value}"
                );
            }
        }
    }
}
