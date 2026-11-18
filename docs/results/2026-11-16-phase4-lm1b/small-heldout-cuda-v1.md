# LM1B-small CUDA held-out quality comparison

The explicitly selected reference-operator/QKV and reference-table candidate
was evaluated against pinned Python on the five seeds frozen in the
[protocol](heldout-protocol-v1.md). Each runtime generated 256 samples per
seed in batch-eight requests, using its own random streams. The frozen pinned
FLM scorer matched upstream on every capture. Each report is bound to its
capture, runtime identity and scorer source; the Rust capture binary SHA-256 is
`3ffed0e6f491e9c55e413a806cdc642de3e3a29eb791b099a22a8e087f2aa8c4`.

| Seed | Python perplexity | Candidate perplexity | Python entropy | Candidate entropy |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 95.537766 | 90.984998 | 4.31250409 | 4.31370621 |
| 0 | 97.137632 | 94.539362 | 4.31592167 | 4.31615328 |
| 1 | 98.656143 | 92.580762 | 4.31673716 | 4.30361092 |
| 2 | 92.686634 | 95.779476 | 4.30996303 | 4.31526264 |
| 3 | 95.340399 | 93.675977 | 4.31839253 | 4.31437212 |

The identity-checked comparison **fails** the unchanged entropy interval
check. Candidate-minus-Python mean perplexity is -2.359600, with independent
seed-bootstrap 95% interval [-4.539778, -0.067769]; the point and upper-end
checks pass the 5.46 limit. Mean entropy difference is -0.002083 nats, within
the ±0.0046 limit, but its interval [-0.007118, 0.002290] extends below the
lower limit. Python and candidate entropy seed standard deviations are
0.003411 and 0.005121 nats. These describe five seed means; they do not prove
a cause or justify changing the threshold. Candidate seed 1 is the lowest
entropy candidate seed in this run.

The hash-checked summary and its repeated output are byte-identical, SHA-256
`2e11e2b0aac10d152e58b888a23cda46b0df1b45285019d0ccf847e1f2f93002`.
The summary command exits 1 as required for the failed metric; all ten
captures, ten score reports, command logs and the summary are retained under
`reports/phase4-lm1b-small-heldout-cuda-v1`. The scorer verified its output
against the pinned upstream scorer for every seed. The exact five-seed set is
now observed: any subsequent same-seed run is a retest, not a fresh held-out
comparison. Do not change the margin, bootstrap, oracle or seed set to hide
this result. Controlled two-row parity and development quality are separate
results; the held-out CUDA quality gate, other LM1B gates and Phase 4 remain
open.

## Descriptive Python self-control

As a diagnostic after the held-out run, the same frozen metric comparison was
applied between Python's five already-scored development seeds 11–15 and its
five held-out seeds. The comparison requires its five configured seed labels;
the development rows were mapped in order to those labels only to satisfy its
schema. The bootstrap samples independently, so the mapping does not imply
paired random streams. The control also fails the entropy interval: development
minus held-out is -0.003167 nats, with interval [-0.006221, 0.000180].
Perplexity difference is -2.895863 with interval [-5.477730, 0.044911]. The
control indicates that this observed frozen check can fail between two samples
from the same reference implementation. It neither proves the Rust candidate
correct nor changes or waives the actual failed held-out gate.

`tools/reference/lm1b_python_self_control.py` validates the retained held-out
summary and development capture/scorer hashes before and after the comparison.
Two outputs were byte-identical, SHA-256
`5104bf266d98153b0d8251443a7fb99f0fdb43663fad149c6c9418941abcbbb0`;
existing metric-comparator tests pass.

## Identical reference control

`tools/reference/lm1b_identical_reference_control.py` supplies the same five
retained Python seed scores as both inputs to the unchanged comparator, after
checking each score against the retained summary's digest. This descriptive
control passes with zero point differences and entropy interval
`[-0.003724240511655985, 0.003724240511655097]`, within the frozen
`[-0.0046, 0.0046]` limits. It shows that these reference samples do not force
an identical comparison to fail; it does not qualify the Rust output, change
the independent-resampling protocol, or undo the actual held-out failure.

## Observed reference partition control

`tools/reference/lm1b_reference_split_control.py` uses the five retained
Python development seeds and the five retained Python held-out seeds. It
checks the pinned summary and development capture/score digests, then applies
the frozen comparison to every ordered choice of five reference seeds and its
complement. It reproduces the comparator's deterministic independent bootstrap
draws, checks nine partition results against the comparator itself to `1e-12`
and confirms identical pass/fail decisions, and checks
the input hashes again afterward.

Of the 252 overlapping assignments, 169 fail the entropy interval, six fail
the entropy point bound, and 16 fail the perplexity interval; the union is
169. No assignment fails the perplexity point bound. The assignments overlap
and include reverse orientations, so they are not 252 independent trials or
an estimate of a population failure rate. All ten seeds were already observed;
this post-hoc control is not new held-out evidence, a candidate qualification,
or a change to the frozen protocol. It supports reviewing the protocol's
sensitivity alongside the existing failed Python self-control.

Run with `env PYTHONDONTWRITEBYTECODE=1 .reference/venv/bin/python
tools/reference/lm1b_reference_split_control.py` from the repository root.
The result repeated with 252 assignments and nine comparator cross-checks.

## Intrinsic rotary observed-seed retest

After the intrinsic rotary change, the five already observed seeds were
regenerated with the 1 MiB QKV diagnostic, batch eight, 256 samples per seed,
and CUDA FP64 penalty. The original Python reference captures and scores were
reused after verifying their digests against the retained original summary.
Each new Rust sample was scored against the pinned upstream evaluator, and
each scorer result matched the upstream functions. These are retests, not
fresh held-out samples; the original failed comparison is preserved above.

| Seed | Rust perplexity | Rust entropy |
| ---: | ---: | ---: |
| 42 | 90.975223 | 4.313827975 |
| 0 | 93.514066 | 4.313405072 |
| 1 | 91.916921 | 4.304004038 |
| 2 | 96.494992 | 4.318152621 |
| 3 | 94.144525 | 4.309769375 |

The unchanged comparison **fails** its entropy interval: Rust minus Python
mean is `-0.002871878`, with independent seed-bootstrap 95% interval
`[-0.007821706, 0.002025763]` against `[-0.0046, 0.0046]`. The entropy
point check passes. Perplexity mean difference is `-2.462569`, interval
`[-4.765141, 0.063868]`, and both perplexity checks pass the `5.46` limit.
The deterministic summary was repeated byte-for-byte (SHA-256
`b9e1b0c76a21986fde2e61113d1066983f82e865326955823d50e54b74eba99c`).
This failure does not qualify the CUDA quality gate or change its protocol.

The capture uses the original observed seed guard and generation procedure;
its explicit diagnostic metadata identifies intrinsic reference-order tables
instead of the earlier externally supplied table. The retest scorer differs
from the original only in that identity check and the retest scope label;
its metric calculations and upstream scorer are unchanged. The comparison
uses the original `lm1b_quality_compare.py` unchanged and verifies both
original reference and new Rust identities. New capture binary SHA-256 is
`c1151689daa307bd08f6dbca646bb56e08bd6640a74ee51e14dc7e53e745358d`;
scorer and summary source hashes are
`bc9917f335c05472616fcc11298a4c2410522e4d6978f7c9f226458237c66ae4` and
`b72d6966992cf4c03334b1aca1f41ae4239b7eb98e3a5750c319033d660b3111`.
Captures, scores, both summaries, logs and diagnostic sources are retained in
`reports/phase4-lm1b-intrinsic-rope-v1/lm1b-intrinsic-observed-retest-v1/`.
The capture seed-guard test and feature-specific Clippy passed; both Python
tools compile, the three existing comparator tests pass, and missing input
and altered diagnostic identity were rejected without output. Full performance
and Metal gates remain separate.

From the repository root, the retained comparison can be reproduced without
running inference; choose a new output filename under `reports/`:

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tools/reference .reference/venv/bin/python \
  reports/phase4-lm1b-intrinsic-rope-v1/lm1b-intrinsic-observed-retest-v1/lm1b_intrinsic_retest_summary.py \
  --directory reports/phase4-lm1b-intrinsic-rope-v1/lm1b-intrinsic-observed-retest-v1 \
  --output reports/lm1b-intrinsic-observed-retest-reproduced.json \
  --rust-binary-sha256 c1151689daa307bd08f6dbca646bb56e08bd6640a74ee51e14dc7e53e745358d
```

The expected exit status is 1 because the recorded metric check fails. The
first archive replay failed before comparison because the comparator was not
beside the archived summary script; it was copied unchanged, and the replay
then reproduced the summary digest above.

## Exploratory sample-count planning

A post-hoc simulation examined the sensitivity of the existing two-metric
interval checks to sample count. It used the ten already observed Python seed
scores and five already observed scores from the unqualified combined CUDA
candidate in the [controlled diagnostic](small-controlled-cuda-v1.md#combined-cuda-arithmetic-diagnostic-and-observed-seed-retest).
The Python and candidate entropy sample standard deviations were 0.003204 and
0.004734 nats, respectively. Their perplexity standard deviations were 2.8004
and 2.0508. The script checks input hashes and centers each observed metric
within its runtime before imposing specified hypothetical differences. It
resamples whole seed records independently for each runtime, preserving the
per-seed perplexity/entropy relationship.

For each scenario, 1,000 simulated comparisons used 10,000 independent
bootstrap resamples per comparison and the existing margins and percentile
indices. The simulation used NumPy PCG64 seed 20261117; it does not reproduce
the frozen comparator's Python RNG sequence. With hypothetical
candidate-minus-reference differences of -2.0 perplexity and -0.002 entropy,
the joint checks passed in 611/1,000 simulated comparisons at 20 seeds per
runtime, 858/1,000 at 40, and 990/1,000 at 80. With +0.002 entropy instead,
the respective counts were 628, 908, and 996. The simulated perplexity check
passed whenever entropy did in these scenarios. For zero assumed differences,
the joint counts were 968, 999 and 1,000. These are conditional simulation
counts, not observed rates for new samples.

Only ten reference and five candidate seed means form the empirical
populations. Resampling those finite observations can understate population
variation; the assumed effects, independence and distribution may be wrong.
The 1,000 trials also have Monte Carlo error, and the results say nothing
about candidate correctness, model parity, performance, or other profiles.
They do not choose a new sample count or establish a replacement protocol.
Any separately proposed protocol would require an explicit decision before
new seeds are generated; the original failed frozen gate remains recorded.

The simulation source and two byte-identical outputs are retained under
`reports/phase4-lm1b-sizing-evidence-v1/`. Source SHA-256 is
`411d1ecc7e2460b4eb3b47c498b983e920ecf6a282ae83848dc8c7826d9771d8`
and result SHA-256 is
`c98b02ba01f96b89f53b05cc3d60d8e30bce9622a85f8e952c986ff7c1ea7d91`.
An independent cross-check using the frozen comparator's exact random indices
matched both percentile intervals to `1e-12`; both scripts pass Python syntax
checking. Reproduce the simulation from the repository root with:

```sh
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .reference/venv/bin/python reports/phase4-lm1b-sizing-evidence-v1/ncm-lm1b-joint-planning-v2.py
```

## Fixed-code CUDA decode replay

The replay reads the ten captures and scorer receipts bound by the original
summary SHA-256. For each capture it checks the initial concepts against the
runtime's documented CPU draws from the pinned level-zero unigram. All 2,560
initial concepts match. Each seed has 256 distinct concept rows and 256
distinct raw-ID rows. Recomputing entropy from raw token counts in FP64
agrees with the recorded upstream entropy within `8.82e-7` nats per row.

Each decoder receives all 2,560 retained code rows in the original batches
of eight. The Python path uses the pinned codec and original token-selection
function. The Rust path uses the candidate's CUDA codec and device FP64
penalties. Both paths reproduce their own capture's complete raw-ID stream.
On common inputs, the paths disagree at 393 of 327,680 token positions
across 188 rows.

The following diagnostic uses the same decoder for both groups of code
rows. It computes per-seed entropy in FP64 from the replayed raw IDs and applies
the existing independent-seed resampler.
The original scorer receipts remain the authority for the quality decision.

| Decoder used for both groups | Rust minus Python entropy (nats) | 95% entropy interval (nats) |
| --- | ---: | --- |
| Pinned Python | -0.002113 | [-0.007203, 0.002304] |
| Rust CUDA | -0.002082 | [-0.007081, 0.002258] |

Both intervals extend below the frozen lower bound of `-0.0046`. The
largest absolute per-seed entropy change attributable to the decoder is
`0.000175` nats. The decoder differences therefore cannot account for the
failed entropy interval on these retained rows.

A score observer records the difference between Python's selected token and
the Rust token at the first disagreement in each row. The histories match
through the preceding position. The observer reproduces the original
Python IDs from those batches. Across all 188 sites the median Python
score margin is `0.00504`, with nearest-rank p95 `0.01849` and maximum
`0.04771`.

The archived Python path has cuDNN TF32 enabled and matrix-multiplication
TF32 disabled. The Rust cuDNN convolution specifies FMA math. A separate
fixed-code control disables only Python's cuDNN TF32 setting. This
eliminates all 393 disagreements across the 2,560 rows. The common-decoder
entropy interval remains `[-0.007081, 0.002258]` in this control. Keep the
original Python settings and receipts for the frozen quality gate.

Build and run the decoder audit below. Add `--python-cudnn-fp32` and choose
a fresh output directory for the precision control.

```sh
lib="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib"
link="$PWD/.reference/phase4-cudnn-probe-lib"
setpriv --inh-caps=-all env CUDA_COMPUTE_CAP=86 RUSTC_WRAPPER= \
  LIBRARY_PATH="$link" LD_LIBRARY_PATH="$lib" \
  cargo build --locked --release -p ncm-backend-cuda \
  --features cuda-cudnn,lm1b-experimental --example lm1b_small_fixed_code_decode
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH="$lib" \
  .reference/venv/bin/python tools/reference/lm1b_fixed_code_decode_audit.py \
  --archive reports/phase4-lm1b-small-heldout-cuda-v1 \
  --artifact artifacts/canonical/lm1b-small-inference-v1 \
  --upstream .reference/upstream-ncm --source artifacts/upstream/lm1b-phase4 \
  --rust-binary target/release/examples/lm1b_small_fixed_code_decode \
  --output reports/lm1b-fixed-code-replay
```

The default replay repeats the same code and raw-ID hashes with the score
observer enabled. Both precision control receipts and the first-disagreement
margins are under `reports/phase4-lm1b-decode-precision-v1.L6XPFS/`.
The default and FP32 summaries hash to
`a8e2341ec1bff9d0ddcc13d318a2f38a7b1a892dd840fbd9b7490961be9a3e6e`
and `4d5f34d78d25988a3cb9a8005927674282c0fa996b53e966c69414d8e2e13348`.
Feature-enabled release build and Clippy passed. The Rust probe rejects
wrong input hashes and invalid row shapes. The standard host suite passed
all 532 report tests, with log SHA-256
`eeddd863b5e724e948a1ad0dffba2d7898130212b77ee7eb2aa3c3ec778f350c`.

### Generation with strict Python convolutions

A separate control reruns the five observed Python seeds with only
`torch.backends.cudnn.allow_tf32` set to `False` before generation.
The original pinned capture tool, checkpoint, tokenizer and batch
shape are used through a wrapper. At the original precision, a fresh
seed-1 capture repeats its archived JSON byte for byte.

Every initial concept agrees between the two precision settings.
Across the 1,280 rows, strict convolution changes 69,494 later
concept positions in 1,054 rows and 23,519 raw token IDs in 1,005
rows. The strict-Python entropy diagnostic compared with archived
Rust yields a mean difference of `-0.000391` nats and an independent
95% interval of `[-0.005278, 0.003711]`. This remains below the
frozen lower bound. The control covers entropy on observed seeds.
The original FP32-plus-TF32 Python capture path and complete scorer
receipts remain the baseline for acceptance.

The wrapper, per-seed controls, and summary are under
`reports/phase4-lm1b-generation-fp32-v1.PBBv9e/`. The wrapper and
summary SHA-256 values are
`5a69e6ee9ee35874c034567ce1e881aa2723bba3e2ee06a36311f5fe080b7228`
and `8a5b05fbabf9f151c2809799238c815be063dbd2b4b25fbaf35a5e0a52633cb6`.
