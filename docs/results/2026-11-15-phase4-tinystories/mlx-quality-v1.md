# TinyStories Metal stochastic-quality retest

The native Metal implementation generated 256 samples for each official seed at batch 128. The same pinned Python CUDA sample files and the criteria fixed before the original comparison were used: Rust mean generative perplexity must be no more than Python's mean plus 1.15, and absolute mean entropy difference must be at most 0.019 nats. Because the earlier held-out results had already been observed, this is a retest, not a new blinded holdout. Different runtime random streams do not imply same-seed output identity.

| Seed | Pinned Python genPPL | Rust Metal genPPL | Pinned Python entropy | Rust Metal entropy |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 16.572501 | 16.164390 | 3.966081 | 3.970229 |
| 1 | 15.659647 | 16.023752 | 3.964875 | 3.970406 |
| 2 | 16.402500 | 15.932677 | 3.976839 | 3.959073 |
| 3 | 16.351816 | 16.425206 | 3.973933 | 3.968879 |
| 42 | 16.176005 | 16.090500 | 3.959324 | 3.969391 |
| Mean | 16.232494 | 16.127305 | 3.968210 | 3.967596 |

Both frozen stochastic-quality criteria pass. All 1,280 raw ID, decoded-text and re-encoded-token sequences passed checks against the pinned upstream BPE tokenizer before scoring. The scoring used the same pinned GPT-2-large model and qualified memory-bounded batch-32 evaluator as the [CUDA held-out report](heldout-quality-v1.md); that report documents the original scorer's GPU OOM and the evaluator qualification. Quality scores do not establish exact sample identity, settle the cross-runtime floating-point exceptions, or override the [Metal memory-gate failures](mlx-performance-v1.md).

The five sample files, source/executable/library fingerprints, process records and CPU-activity samples are retained under ignored `reports/phase4-tinystories-mlx-quality-v1/`. Capture used `tools/bench/run_phase4_tinystories_mlx_quality.py` after direct fixed-code Rust/Python MLX qualification. The independent scoring output `quality-v1.json` has SHA-256 `fe177a49d06ab69c4fab6ff3a8664ee369a5ba285ddf37d002cb97135b69954a`. Scoring occurred on the Linux CUDA device; generation occurred on the Mac. GPU activity was not directly measured during capture.
