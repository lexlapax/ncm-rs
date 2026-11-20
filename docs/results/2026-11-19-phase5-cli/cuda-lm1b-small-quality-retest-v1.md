# LM1B-small CUDA 80-seed nonblind retest

2026-11-20. The approved [80-seed protocol](../2026-11-16-phase4-lm1b/cohort-campaign-v1.md)
was repeated using the same frozen registry: Python seeds 1000–1079, Rust seeds 2000–2079,
256 samples per seed and batch eight. The cohort has already been observed, so the manifest and
summary explicitly say `retest`; this is **not** a prospective or new held-out run. The original
five-seed failure and the approved protocol's limitations remain on record.

The existing campaign captured pinned upstream CUDA Python and the current Rust cohort runner,
then used the pinned FLM scorer and unchanged independent-seed bootstrap (10,000 draws, seed 42).
All 160 score records reported `upstream_exact: true`; summary replay revalidated the captures,
scores, pinned inputs and receipts and reproduced its bytes. Results:

| Runtime | Seeds | Mean perplexity | Mean entropy |
| --- | ---: | ---: | ---: |
| Python | 80 | 93.916013 | 4.312720 |
| Rust | 80 | 94.359900 | 4.313447 |

| Rust − Python | Point | 95% interval | Frozen limit | Result |
| --- | ---: | --- | --- | --- |
| Perplexity | +0.443887 | [−0.271116, 1.145112] | upper bound ≤ 5.46 | pass |
| Entropy | +0.000727 | [−0.000325, 0.001795] | within ±0.0046 | pass |

All 160 per-seed metric records and the comparison equal the earlier
[prospective CUDA result](../2026-11-16-phase4-lm1b/cohort-campaign-v1.md#prospective-cuda-result),
though capture/score receipt identities differ because this is a new manifest. The compact
[summary](lm1b-small-cuda-retest-summary-v1.json) is published without generated text; SHA-256
`0503e852897be9026ff92ea9480ea06dbc2dbfe33a60678dfa9736ed4afb4a95`. Raw captures, per-seed
scores and receipts remain ignored under `reports/phase5-lm1b-small-cuda-retest-v1/`.

| Item | SHA-256 |
| --- | --- |
| Campaign manifest | `0ccbac0502c735730692d8cbdfe0d017bfb9ea5e57a0157362aa60370897821b` |
| Python capture receipt | `9603bdfbf504839f7474bf43e4a7c01a166fc901cb82107b61c1ad0b8b43fd22` |
| Rust capture receipt | `742d4f25e6c85b2037cfd621a86e48948b54bff137e639894330702feac99506` |
| Score receipt | `788c2e02e76ca0ef9ec54c31232328eefc064c979be57dc05fcb06aef1ed7839` |
| Rust campaign runner | `59501cd655040b541965bb3c83c5fbdb205fcab901e2aa5b7c456018173e796a` |
| Campaign tool | `7cb60d7d6fcb2376e0f91c375206a105d86b8ab6527c5bda91f41bb61f0f5ee2` |

The runner and manifest were built/initialized at `f5028c2`. Capture, scoring and summary used
`setpriv --inh-caps=-all` and the environment documented in the campaign reproduction section,
with `--mode retest --size small`; the campaign retained and checked the original artifact,
rotary, scorer and seed registry pins. The GPU was shared with a desktop and not continuously
monitored. This is a scoped CUDA quality retest of the cohort runner, not a performance or Metal
comparison, full CLI corpus-equivalence claim, or Phase 5 completion.
