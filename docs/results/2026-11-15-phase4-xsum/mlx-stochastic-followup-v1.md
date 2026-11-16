# XSum Metal stochastic quality follow-up

The baseline Metal fixed-point sampler at pushed `e14ae02` was compared with the
retained pinned Python CUDA follow-up on 2,048 rows (512–2559), five seeds 4–8,
and independent sampling streams. Before the Metal capture, the [plan](mlx-stochastic-quality-plan-v1.md)
fixed the existing protocol, its unchanged margins, batch seeds and 97.5th-percentile
bootstrap rule. Each Metal seed has 64 source-bound batches, and copied inputs,
receipts and outputs matched the Mac by SHA-256. Capture and scoring used the
previously screened Metal CLI; the owned-loader candidate was not involved.

| Metric | Observed degradation | 97.5% upper bound | Frozen margin |
| --- | ---: | ---: | ---: |
| ROUGE-1 | -0.1364 | 0.2441 | 0.62 |
| ROUGE-2 | -0.0326 | 0.2201 | 0.64 |
| ROUGE-L | -0.0628 | 0.2734 | 0.62 |
| Generative perplexity | 0.7452 | 7.9654 | 12.52 |
| Mean words, absolute | 0.0449 | 0.2276 | 0.30 |

All five bounds satisfy this comparison's rule. Negative ROUGE degradation
means the Metal point estimate is higher. These rows and Python outputs had
already been inspected in the CUDA follow-up: this is an additional backend
comparison, not a previously unseen familywise-controlled study. It does not
erase the original failed CUDA held-out study, the controlled-draw differences,
the [Metal top-1 raw-ID failure](mlx-full-top1-v1.md), or establish Metal
performance or memory acceptance. No margin or tolerance changed.

The Metal-aware scorer used the pinned evaluator and model, validated all 64
input/receipt pairs per seed, and recorded the wrapper and helper hashes. It
exactly replayed every retained Python seed-4 per-row score and aggregate metric;
the strengthened helper-hash version also reproduced the earlier four Metal
seed scores exactly. The bootstrap implementation exactly reproduced both the
observed differences and 10,000-draw 97.5% bounds of the previously published
Python-versus-CUDA follow-up before this result was interpreted. The replay
result SHA-256 is `3483eda3a21df50417c3fc959a5544f803ff89b316d740c9ba26dd5f5b092587`;
the bootstrap replay is `06b90bdd484617691783eadc72f8b0985c5a36f53ba26ce127182bae7e90bd80`.
Synthetic capture-mutation and injected-regression tests also passed. The two
runs of the same deterministic Metal bootstrap were byte-identical, not separate
statistical looks.

The comparison JSON is [mlx-stochastic-followup-v1.json](mlx-stochastic-followup-v1.json),
SHA-256 `23767d8ad0a1b5957dffb34dabb189668a3a0bc86ea7134d62a4fac2a44120e4`.
Raw captures and per-row scores remain under ignored
`reports/phase4-xsum-mlx-stochastic-seed-{4,5,6,7,8}-captured-v1/` and
`reports/phase4-xsum-mlx-stochastic-seed-{4,5,6,7,8}-score-v2.json`; the JSON
lists their score hashes and the pinned Python score hashes. Scoring ran on the
Linux RTX 3060 Ti with driver 620.70.06; capture ran on the recorded M1 Ultra.
Full raw outputs, quality bounds and all limitations should be retained in any
future gate decision.
