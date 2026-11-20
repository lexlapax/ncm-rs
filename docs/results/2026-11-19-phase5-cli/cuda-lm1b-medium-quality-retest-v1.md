# LM1B-medium CUDA 80-seed nonblind retest

2026-11-20. Repeated the approved [LM1B 80-seed protocol](../2026-11-16-phase4-lm1b/cohort-campaign-v1.md)
with the original frozen registry: Python seeds 1000–1079, Rust seeds 2000–2079, 256 samples per
seed and batch eight. The seeds have already been observed; the manifest explicitly says `retest`,
not prospective or newly held out. Scorer, seed lists, margins and bootstrap were not changed.

Pinned upstream CUDA Python and the current Rust cohort runner were captured and scored with the
existing campaign tool. All 160 score records reported `upstream_exact: true`; a separate summary
replay revalidated the receipts and reproduced its bytes. Results:

| Runtime | Seeds | Mean perplexity | Mean entropy |
| --- | ---: | ---: | ---: |
| Python | 80 | 88.774846 | 4.339916 |
| Rust | 80 | 88.843234 | 4.340177 |

| Rust − Python | Point | 95% interval | Frozen limit | Result |
| --- | ---: | --- | --- | --- |
| Perplexity | +0.068388 | [−0.481525, 0.614443] | upper bound ≤ 5.46 | pass |
| Entropy | +0.000261 | [−0.000839, 0.001368] | within ±0.0046 | pass |

All 160 per-seed metrics and the comparison exactly match the prior
[prospective medium CUDA result](../2026-11-16-phase4-lm1b/cohort-campaign-v1.md#prospective-cuda-results-medium-and-large).
The compact [summary](lm1b-medium-cuda-retest-summary-v1.json), without generated text, has
SHA-256 `1c95bc7fa63fcc7c175a308b087c97d8a2711c6f2661071f5201a54cfc8d8d99`. Raw captures,
per-seed scores and receipts remain ignored under `reports/phase5-lm1b-medium-cuda-retest-v1/`.

| Item | SHA-256 |
| --- | --- |
| Campaign manifest | `fb3dfe8ab77eede7c7210658b7432533a491136b0b3ec919f29549e54f450c1f` |
| Python capture receipt | `39708f0f180b63010af5afa428839a5e9b60335804c9c74198d8e20ea65332af` |
| Rust capture receipt | `57f89148f93318ba7567bf32b7270d48867b8e719b8095855a24013bab1a20da` |
| Score receipt | `a2ba82ca23f999d53d284fab08b4b2e247b93cc50727cdd41219db5d537b9223` |
| Rust campaign runner | `59501cd655040b541965bb3c83c5fbdb205fcab901e2aa5b7c456018173e796a` |

The manifest was initialized at `3b6d6b3` using `--mode retest --size medium`. Capture, scoring
and summary ran with `setpriv --inh-caps=-all` and the documented campaign environment; summary
replayed byte-for-byte. The desktop GPU was shared and not continuously monitored. This is a
scoped CUDA cohort-runner quality retest, not full CLI cohort equivalence, performance, Metal or
Phase 5 completion.
