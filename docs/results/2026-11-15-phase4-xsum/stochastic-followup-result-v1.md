# XSum stochastic independent follow-up: bounded quality passed

The [predeclared follow-up](stochastic-followup-plan-v1.md) evaluated fresh rows 512–2559 for both runtimes, five new seeds and 2,048 summaries per seed. All 10,240 summaries per runtime passed capture and scoring integrity checks. Source-bound batch results and scored per-row data remain under `reports/phase4-xsum-stochastic-followup-{python,rust}-{seed}*`. The [complete comparison](stochastic-followup-result-v1.json) has SHA-256 `4c788e0509a40523a4e1a3b789843f817ee16465470014598150b4a3cb9c79eb` and records every per-seed metric and score hash.

| Metric | Python five-seed mean | Rust five-seed mean | 97.5% degradation bound | Original margin | Follow-up |
| --- | ---: | ---: | ---: | ---: | --- |
| ROUGE-1 | 31.0140 | 31.1431 | 0.2432 | 0.62 | Pass |
| ROUGE-2 | 9.1146 | 9.1414 | 0.2243 | 0.64 | Pass |
| ROUGE-L | 24.1567 | 24.2114 | 0.2783 | 0.62 | Pass |
| Generated-text perplexity | 186.7884 | 187.3440 | 7.5720 | 12.52 | Pass |
| Mean words | 20.6750 | 20.7205 | 0.2285 | 0.30 | Pass |

ROUGE degradation is Python minus Rust; perplexity degradation is Rust minus Python; mean words uses the absolute difference. Each bound meets the original Python-derived margin. This independent larger sample is consistent with similar stochastic quality in the measured setting and narrows uncertainty relative to the first study. It does not prove universal equivalence or establish the cause of individual decision differences. The follow-up was chosen after the first study and therefore does not erase the [original precommitted gate failure](stochastic-heldout-v1.md), where four uncertainty bounds exceeded their limits. Both results must be considered; no tolerance or margin was altered. The prior two forced-history near-boundary controlled-draw differences likewise remain recorded. [Repeated CUDA speed and memory](stochastic-performance-v2.md) favor Rust in all measured cells. Linux gate disposition and XSum MLX work remain pending review of these visible exceptions.
