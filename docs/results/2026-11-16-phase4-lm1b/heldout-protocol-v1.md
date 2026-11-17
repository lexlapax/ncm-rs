# LM1B-small held-out quality protocol

This protocol is fixed before scoring official LM1B-small seeds. It applies
only to the small checkpoint and the pinned FLM task; it does not qualify the
medium or large profiles. The five separate Python development seeds 11–15,
256 samples each in batches of eight, were scored with the pinned upstream
GPT-2-large FLM calculation. The [development evidence](inventory-preparation-v1.md)
and its hash-checked summary record the inputs, outputs, evaluator and limits.
No Rust or official-seed quality result is claimed here.

The development seed means were 92.975852 perplexity with sample standard
deviation 2.725898, and 4.311537 nats raw-ID entropy with sample standard
deviation 0.002280. The diagnostic seed-bootstrap 95% intervals for the Python
baseline were [91.103785, 95.426944] and [4.309858, 4.313362]. The intervals
resample five seed means 10,000 times with replacement using Python seed 42;
they are descriptive and uncertain with only five seeds.

For the later held-out quality comparison:

- Generate 256 samples for each configured seed `[42, 0, 1, 2, 3]` and each
  runtime, with the configured sampling, batch-eight inputs, initial unigram,
  prefix and decoding penalties. Preserve raw codes, IDs and enough identity
  data to reproduce scoring. Random streams need not match across runtimes.
- Score each seed using the pinned upstream `genppl_flm` behavior: decode raw
  IDs with BERT special tokens retained; retokenize with GPT-2-large, padding
  and truncating to 128; use its exact first-EOS validity expression and
  batch-16 corpus perplexity; calculate mean per-sample entropy on raw IDs.
- Compare the five-seed Rust and Python means. Rust minus Python perplexity
  must be at most **5.46 absolute points**, and the absolute difference in
  mean entropy must be at most **0.0046 nats**. These margins round up twice
  the observed Python development seed standard deviations, as a pragmatic
  noise-informed bound; they are not an upstream numerical tolerance or a
  guarantee of semantic correctness.
- Also report 95% percentile confidence intervals for Rust-minus-Python
  differences, independently resampling the five seed means from each runtime
  10,000 times with replacement using a dedicated deterministic RNG seed 42.
  The upper perplexity interval endpoint must be at most 5.46; both entropy
  interval endpoints must lie within `[-0.0046, 0.0046]`. Resampling is
  unpaired because matching seed numbers do not establish matching random
  streams. Five seeds limit the precision and interpretation of these bounds.
- Fail explicitly for missing seeds, wrong sample counts, invalid IDs, changed
  model/tokenizer identities, unavailable scorer, or nonfinite metrics. Keep
  all failed runs and do not adjust the margins after seeing held-out results.

Tokenizer and teacher-forced numerical parity, controlled stochastic draws,
resource measurements, CUDA and Metal gates, and other LM1B sizes are separate
requirements. The observed scorer batch fit is a development run with desktop
GPU activity, not a guarantee that all official scoring runs fit.

The separate held-out capture tools are `tools/reference/lm1b_cuda_heldout_capture.py`
and `lm1b_small_heldout_capture`; each accepts only the five frozen seeds and
256 samples in batch-eight requests. The Rust path explicitly requires the
reference-table/QKV diagnostic candidate and uses CUDA inverse-CDF and FP64
penalty with CPU level-zero sampling. The reference path checks the pinned
upstream checkout, checkpoint and generation configuration. The separate
`tools/reference/lm1b_heldout_score.py` verifies capture hashes, runtime-specific
model identities, shapes and ranges, then compares its scorer exactly with the
pinned upstream functions. Its five-seed metric comparison remains the existing
`lm1b_quality_compare.py`. The capture records the Rust executable hash; the
scorer verifies that hash, the Python capture source, and its own source identity.
`lm1b_heldout_summary.py` binds all ten captures and scores before applying
the existing comparison. The Rust seed guard, feature-specific lint, existing
metric-comparator tests and missing-input rejection pass; full CUDA validation
passed, including 18 CUDA device cases
(`reports/phase4-lm1b-heldout-final-tools-cuda-check-v1.log`). The initial
scorer help invocation lacked the tokenizer environment; the corrected
invocation passed. No held-out samples have been generated or scored at this
preparation point.
