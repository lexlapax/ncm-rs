# LM1B: user decisions for closing Phase 4

Recorded 2026-11-18. The user directed that Phase 4 close without going broader or deeper
than needed. The goal is to replicate the pinned upstream functionality with better performance.
These decisions are explicit and scoped. They do not convert any earlier failed comparison into
a pass, and they do not change the pinned upstream code, artifacts, oracle or numerical tolerances.

## 1. LM1B quality gate: the fresh 80-seed protocol is approved

The frozen five-seed comparison in the [held-out protocol](heldout-protocol-v1.md) failed its
entropy interval on CUDA. The [original failure](small-heldout-cuda-v1.md) and every later
retest stay on record. The same comparison applied to Python against itself also fails:
the [self-control](small-heldout-cuda-v1.md#descriptive-python-self-control) fails, as do
169 of 252 [partitions of Python's own ten seeds](small-heldout-cuda-v1.md#observed-reference-partition-control).
The user approved the [proposed 80-seed protocol](small-heldout-cuda-v1.md#manifest-bound-cuda-rehearsal)
as the LM1B quality gate:

- 80 fresh seeds per runtime, with 256 samples per seed in batch-eight requests, using the
  configured sampling, initial unigram, prefix and decoding penalties.
- The same pinned upstream FLM scorer, the same margins (perplexity at most **5.46** points worse;
  absolute mean entropy difference at most **0.0046** nats, including both 95% interval endpoints),
  and the same independent seed bootstrap (10,000 draws, RNG seed 42).
- The per-runtime seed lists are disjoint, lie outside the observed seeds 0–16 and 42, and are
  fixed in [cohort-seeds-v1.json](cohort-seeds-v1.json) before any generation. Rust Metal uses its
  own list and is compared with the same pinned Python CUDA captures, so quality remains anchored
  to the original implementation.
- The same protocol applies to LM1B-medium and LM1B-large.
- Missing seeds, wrong counts, invalid IDs, changed identities or nonfinite metrics fail
  explicitly. Margins and seed lists are not adjusted after results are seen.

## 2. Metal stochastic draw differences: scoped exception

Rust Metal does not reproduce pinned CUDA's sampled choices draw for draw. In the batch-16
fixed-history comparison, 30 of 10,240 same-history draws differed and 19 of 46 floating tensors
missed the float policy, starting at level-zero reconstruction. The FP32 top-p cutoff also
differs at retained boundary draws. Generator and decoder argmaxes and all 2,048
penalty-adjusted IDs matched. See the [large-batch evidence](small-controlled-cuda-v1.md#large-batch-fixed-history-comparison)
and the [top-p investigation](small-controlled-cuda-v1.md#metal-fp32-top-p-same-logit-investigation).

The contract does not require identical random samples across runtimes. The user accepted these
draw-level and float-policy differences for advancement and stopped the cross-device draw
investigation. LM1B Metal is judged on fixed-history parity of decisions and IDs, the 80-seed
quality protocol and performance. The `--top-p-fp32` option remains explicitly experimental.
This exception does not extend the numerical tolerance to other tensors, profiles or future
changes. A new decision or ID difference on a fixed history is still a failure.

## 3. Minimum bar for LM1B-medium and LM1B-large

For each size, Linux/CUDA first and then Metal:

- artifact export with identity checks, plus teacher-forced parity of decisions and IDs at one or
  two shapes;
- CLI generation;
- quality under the 80-seed protocol;
- one batch-sweep latency and memory comparison against the corresponding Python baseline,
  with explicit memory budgets.

New numerical investigations happen only when one of these checks fails.
