//! Explicit host-side implementation of the pinned XSum output-token penalties.

use ncm_core::{Error, Result};

/// Apply the pinned XSum evaluation penalties to FP32 `[batch, time, vocab]`
/// decoder logits, producing raw token IDs before EOS cutting or text decoding.
///
/// The upstream evaluator converts logits and allowance to FP64 and applies the
/// adjustments sequentially. This explicit CPU operation preserves that order;
/// it is not an implicit backend fallback and is not yet wired into inference.
/// The caller owns the logits/readback, so peak transfer cost is not included.
pub fn select_xsum_tokens_cpu(
    logits: &[f32],
    allowance: &[f32],
    batch: usize,
    time: usize,
) -> Result<Vec<u32>> {
    let vocab = allowance.len();
    if !(1..=128).contains(&batch)
        || !(1..=512).contains(&time)
        || !(1..=1_000_000).contains(&vocab)
    {
        return Err(Error::Input("invalid XSum decoder dimensions".into()));
    }
    let row_len = time
        .checked_mul(vocab)
        .ok_or_else(|| Error::Input("XSum decoder dimensions overflow".into()))?;
    if batch.checked_mul(row_len) != Some(logits.len()) {
        return Err(Error::Input("XSum logits shape differs".into()));
    }
    if allowance.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(Error::Input("invalid XSum token allowance".into()));
    }
    let allowed: Vec<f64> = allowance
        .iter()
        .map(|&v| f64::from(v) * time as f64)
        .collect();
    let mut output = Vec::with_capacity(batch * time);
    let mut counts = vec![0_u32; vocab];
    const ADJACENCY: [f64; 4] = [-5.0, -2.5, -1.25, -0.625];
    for row in logits.chunks_exact(row_len) {
        counts.fill(0);
        let row_start = output.len();
        for (t, values) in row.chunks_exact(vocab).enumerate() {
            let remaining = (time - t) as f64 / time as f64;
            let mut best_id = 0usize;
            let mut best = f64::NEG_INFINITY;
            for (id, &value) in values.iter().enumerate() {
                if !value.is_finite() {
                    return Err(Error::Input("nonfinite XSum decoder logit".into()));
                }
                let count = f64::from(counts[id]);
                let mut score = f64::from(value);
                score -= 2.0 * (count - allowed[id]).max(0.0);
                score += remaining * (allowed[id] - count).max(0.0);
                for (j, &penalty) in ADJACENCY.iter().enumerate().take(t.min(4)) {
                    if output[row_start + t - j - 1] as usize == id {
                        score += penalty;
                    }
                }
                if score > best {
                    best = score;
                    best_id = id;
                }
            }
            output.push(best_id as u32);
            counts[best_id] += 1;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::select_xsum_tokens_cpu;

    #[test]
    fn ties_and_history_are_resolved_independently_per_row() {
        let logits = vec![0.0; 2 * 3 * 3];
        let ids = select_xsum_tokens_cpu(&logits, &[1.0 / 3.0; 3], 2, 3).unwrap();
        assert_eq!(ids, [0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn rejects_invalid_inputs_before_returning_ids() {
        assert!(select_xsum_tokens_cpu(&[], &[], 1, 1).is_err());
        assert!(select_xsum_tokens_cpu(&[0.0], &[1.0], 0, 1).is_err());
        assert!(select_xsum_tokens_cpu(&[0.0], &[1.0], 1, 2).is_err());
        assert!(select_xsum_tokens_cpu(&[0.0], &[f32::NAN], 1, 1).is_err());
        assert!(select_xsum_tokens_cpu(&[0.0], &[-1.0], 1, 1).is_err());
        assert!(select_xsum_tokens_cpu(&[f32::INFINITY], &[1.0], 1, 1).is_err());
    }
}
