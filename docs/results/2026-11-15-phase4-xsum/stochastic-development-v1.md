# XSum stochastic development baseline and frozen rule

Five independent pinned-Python CUDA runs generated and scored XSum development rows 0–255, using seeds 100–104, 32 rows per process. These are new captures, separate from the earlier exploratory stochastic screen. Their per-seed metrics and score hashes are in [the frozen protocol](stochastic-protocol-v1.json), SHA-256 `2e9d5edd74770b2e3345198fef6c97318d37a3682111fd113bf0993cff594070`. Source-bound manifests and outputs remain in ignored `reports/phase4-xsum-stochastic-qualification-dev-py-{seed}-v1/`; scored rows and per-row evaluator data are in `reports/phase4-xsum-stochastic-qualification-dev-py-{seed}-score-v1.json`. The scorer uses per-row NLL from the pinned evaluator, summed in double precision for resampling; this can produce tiny rounding differences from the evaluator's float32 aggregate.

| Metric | Python seed standard deviation | Frozen margin |
| --- | ---: | ---: |
| ROUGE-1 F1 (percentage points) | 0.30873 | 0.62 |
| ROUGE-2 F1 (percentage points) | 0.31667 | 0.64 |
| ROUGE-L F1 (percentage points) | 0.30730 | 0.62 |
| Generated-text perplexity | 6.25667 | 12.52 |
| Mean output words | 0.14742 | 0.30 |

The [previously recorded design](stochastic-qualification-design-v1.md) defines derivation, independent runtime seeds, held-out rows 256–511, the 10,000-replicate 95% bootstrap rule and its limits. Python and Rust held-out outputs have not been generated or scored as of this freeze. The rule is a scoped quality criterion for CFG 0, not a change to deterministic quality or floating-point tolerances, and does not erase the earlier exact-decision failures. A synthetic regression injected into the analysis rejected the degraded ROUGE-1 while the other unchanged metrics passed; syntax and focused integrity checks also passed. Held-out quality and repeated CUDA performance/memory are pending.
