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
