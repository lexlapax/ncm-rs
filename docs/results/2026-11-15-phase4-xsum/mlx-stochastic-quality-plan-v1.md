# XSum Metal stochastic quality comparison plan

Before inspecting any full Metal stochastic quality outputs, use the existing
[follow-up protocol](stochastic-followup-protocol-v1.json), SHA-256
`8ccd05295382f6ad33b17183a3f24e5d726da74a73fc1a7027eea81af10d5332`, without
changing its original margins: dataset rows 512–2559, seeds 4–8, batches of 32,
and batch seed `1000 * seed + batch_index`. Compare the fixed-point Metal
candidate against the already captured pinned Python CUDA outputs and use the
same scorer and 10,000-draw bootstrap, including its 97.5th percentile rule.
Sampling streams remain independent. Capture raw IDs, codes, input and binary
hashes, and retain every failure. Do not infer quality from identical seeds or
require exact sampled sequences.

This is a new backend comparison on rows and Python outputs already examined
in the CUDA follow-up. It is not a previously unseen study relative to those
prior results, and a pass would not erase the original CUDA held-out failure,
the recorded controlled-draw differences, or the additional-look limitations.
The earlier 32-passage Metal seeded diagnostics are also retained. The Metal
capture and scorer must be separately source-bound and validated; no Metal
quality result is claimed by this plan. If any bound fails, investigate rather
than selecting a looser margin or silently adding attempts.

The prepared capture script is `tools/bench/capture_xsum_mlx_stochastic.py`
(SHA-256 `0b3215af1443f46851f6db1a707350b1da38563be16f595a51a59cb47c5fec11`);
its malformed-output regression passes. It uses the previously screened Metal
CLI binary and hashes the frozen protocol and input. A separate Metal-aware
scorer reuses the pinned evaluator routines while validating the Metal capture
format. Because that wrapper has a new source identity, it must exactly replay
the retained Python seed-4 per-row scores and aggregate metrics before its Metal
scores can support this comparison. A failed replay remains visible and requires
investigation; no new numerical tolerance is assumed. The new bootstrap wrapper
uses the same frozen directions, seed, replicate count and percentile and passed
a synthetic injected-regression test. Neither wrapper is a quality result.
