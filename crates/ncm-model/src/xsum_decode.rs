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
    let mut excess = vec![0.0_f64; vocab];
    let mut deficit = vec![0.0_f64; vocab];
    const ADJACENCY: [f64; 4] = [-5.0, -2.5, -1.25, -0.625];
    for row in logits.chunks_exact(row_len) {
        counts.fill(0);
        excess.fill(0.0);
        deficit.copy_from_slice(&allowed);
        let row_start = output.len();
        for (t, values) in row.chunks_exact(vocab).enumerate() {
            let remaining = (time - t) as f64 / time as f64;
            // Only the previous four token IDs can receive adjacency penalties,
            // and those penalties never increase a score. Among the five best
            // unpenalized values at least one is unaffected, so no value below
            // them can win. Keep earlier IDs first for exact-score ties.
            let mut candidates = [(f64::NEG_INFINITY, usize::MAX); 5];
            let mut threshold = f64::NEG_INFINITY;
            for (id, ((&value, &ex), &def)) in values
                .iter()
                .zip(excess.iter())
                .zip(deficit.iter())
                .enumerate()
            {
                if !value.is_finite() {
                    return Err(Error::Input("nonfinite XSum decoder logit".into()));
                }
                let score = (f64::from(value) - ex) + remaining * def;
                if score > threshold {
                    let mut pos = 4;
                    while pos > 0 && score > candidates[pos - 1].0 {
                        candidates[pos] = candidates[pos - 1];
                        pos -= 1;
                    }
                    candidates[pos] = (score, id);
                    threshold = candidates[4].0;
                }
            }
            let mut best_id = usize::MAX;
            let mut best = f64::NEG_INFINITY;
            for (mut score, id) in candidates {
                if id == usize::MAX {
                    continue;
                }
                for (j, &penalty) in ADJACENCY.iter().enumerate().take(t.min(4)) {
                    if output[row_start + t - j - 1] as usize == id {
                        score += penalty;
                    }
                }
                if score > best || (score == best && id < best_id) {
                    best = score;
                    best_id = id;
                }
            }
            output.push(best_id as u32);
            counts[best_id] += 1;
            let count = f64::from(counts[best_id]);
            excess[best_id] = 2.0 * (count - allowed[best_id]).max(0.0);
            deficit[best_id] = (allowed[best_id] - count).max(0.0);
        }
    }
    Ok(output)
}

/// Process at most four independent rows concurrently while retaining row order.
pub(crate) fn select_xsum_chunk_parallel(
    logits: &[f32],
    allowance: &[f32],
    batch: usize,
    time: usize,
) -> Result<Vec<u32>> {
    if !(1..=4).contains(&batch) {
        return Err(Error::Input("invalid XSum chunk batch".into()));
    }
    let row_len = time
        .checked_mul(allowance.len())
        .ok_or_else(|| Error::Input("XSum chunk dimensions overflow".into()))?;
    if row_len == 0 || batch.checked_mul(row_len) != Some(logits.len()) {
        return Err(Error::Input("XSum chunk shape differs".into()));
    }
    if batch == 1 {
        return select_xsum_tokens_cpu(logits, allowance, 1, time);
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = logits
            .chunks_exact(row_len)
            .map(|row| scope.spawn(move || select_xsum_tokens_cpu(row, allowance, 1, time)))
            .collect();
        let mut ids = Vec::with_capacity(batch * time);
        for handle in handles {
            ids.extend(
                handle
                    .join()
                    .map_err(|_| Error::Backend("XSum selector thread failed".into()))??,
            );
        }
        Ok(ids)
    })
}

#[cfg(test)]
mod tests {
    use super::{select_xsum_chunk_parallel, select_xsum_tokens_cpu};

    fn straightforward(logits: &[f32], allowance: &[f32], batch: usize, time: usize) -> Vec<u32> {
        let vocab = allowance.len();
        let allowed: Vec<f64> = allowance
            .iter()
            .map(|&a| f64::from(a) * time as f64)
            .collect();
        let mut result = Vec::new();
        for row in logits.chunks_exact(time * vocab).take(batch) {
            let mut counts = vec![0_u32; vocab];
            let start = result.len();
            for (t, values) in row.chunks_exact(vocab).enumerate() {
                let mut best = f64::NEG_INFINITY;
                let mut selected = 0;
                for (id, &value) in values.iter().enumerate() {
                    let count = f64::from(counts[id]);
                    let mut score = f64::from(value);
                    score -= 2.0 * (count - allowed[id]).max(0.0);
                    score += (time - t) as f64 / time as f64 * (allowed[id] - count).max(0.0);
                    for (j, penalty) in [-5.0, -2.5, -1.25, -0.625]
                        .iter()
                        .enumerate()
                        .take(t.min(4))
                    {
                        if result[start + t - j - 1] as usize == id {
                            score += penalty;
                        }
                    }
                    if score > best {
                        best = score;
                        selected = id;
                    }
                }
                result.push(selected as u32);
                counts[selected] += 1;
            }
        }
        result
    }

    #[test]
    fn top_five_selection_matches_full_scan_with_ties_and_repetition() {
        let mut state = 0x7a5b_924du32;
        for case in 0..96 {
            let vocab = 1 + (case * 29 % 73);
            let time = 1 + (case * 13 % 24);
            let batch = 1 + case % 3;
            let mut next = || {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                state
            };
            let allowance: Vec<f32> = (0..vocab).map(|_| (next() % 13) as f32 / 20.0).collect();
            // Small value domain deliberately creates ties and repeated IDs.
            let logits: Vec<f32> = (0..batch * time * vocab)
                .map(|_| (next() % 17) as f32 / 4.0 - 2.0)
                .collect();
            assert_eq!(
                select_xsum_tokens_cpu(&logits, &allowance, batch, time).unwrap(),
                straightforward(&logits, &allowance, batch, time),
                "case {case}"
            );
        }
    }

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

    #[test]
    fn independent_batch_chunks_match_whole_batch() {
        let values: Vec<f32> = (0..5 * 3 * 7)
            .map(|i| ((i * 17 % 31) as f32 - 15.0) / 7.0)
            .collect();
        let allowance = [1.0 / 7.0; 7];
        let whole = select_xsum_tokens_cpu(&values, &allowance, 5, 3).unwrap();
        let mut chunked = select_xsum_tokens_cpu(&values[..4 * 3 * 7], &allowance, 4, 3).unwrap();
        chunked.extend(select_xsum_tokens_cpu(&values[4 * 3 * 7..], &allowance, 1, 3).unwrap());
        assert_eq!(whole, chunked);
        assert_eq!(
            select_xsum_chunk_parallel(&values[..4 * 3 * 7], &allowance, 4, 3).unwrap(),
            whole[..4 * 3]
        );
        assert!(select_xsum_chunk_parallel(&values, &allowance, 5, 3).is_err());
    }
}
