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
