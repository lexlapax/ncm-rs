"""Explicit NumPy FP64 implementation of the pinned XSum output penalties."""
import numpy as np


def select(logits, unigram):
    if logits.ndim != 3 or logits.shape[0] < 1 or logits.shape[1] < 1 or logits.shape[2] != len(unigram):
        raise ValueError('invalid XSum selection shape')
    if not np.isfinite(logits).all() or not np.isfinite(unigram).all() or np.any(unigram < 0):
        raise ValueError('invalid XSum selection input')
    batch, length, vocab = logits.shape
    allowance = np.asarray(unigram, dtype=np.float64) * length
    counts = np.zeros((batch, vocab), dtype=np.float64)
    output = np.empty((batch, length), dtype=np.int64)
    index = np.arange(batch)
    for t in range(length):
        score = logits[:, t, :].astype(np.float64)
        score -= 2.0 * np.maximum(counts - allowance, 0.0)
        score += ((length - t) / length) * np.maximum(allowance - counts, 0.0)
        for j, penalty in enumerate((5.0, 2.5, 1.25, 0.625)):
            if j < t:
                score[index, output[:, t - j - 1]] -= penalty
        output[:, t] = score.argmax(axis=1)
        counts[index, output[:, t]] += 1.0
    return output
