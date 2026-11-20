# LM1B-large CUDA 80-seed nonblind retest

2026-11-20. Repeated the approved [LM1B 80-seed protocol](../2026-11-16-phase4-lm1b/cohort-campaign-v1.md)
with the original frozen registry: Python seeds 1000–1079, Rust seeds 2000–2079, 256 samples per
seed and batch eight. The seeds have already been observed; the manifest explicitly says `retest`,
not prospective or newly held out. Scorer, seed lists, margins and bootstrap were not changed.

Pinned upstream CUDA Python and the current Rust cohort runner were captured and scored with the
existing campaign tool. All 160 score records reported `upstream_exact: true`; a separate summary
replay revalidated the receipts and reproduced its bytes. Results:

| Runtime | Seeds | Mean perplexity | Mean entropy |
| --- | ---: | ---: | ---: |
| Python | 80 | 81.184697 | 4.338293 |
| Rust | 80 | 81.500307 | 4.339096 |

| Rust − Python | Point | 95% interval | Frozen limit | Result |
| --- | ---: | --- | --- | --- |
| Perplexity | +0.315610 | [−0.182557, 0.805486] | upper bound ≤ 5.46 | pass |
| Entropy | +0.000803 | [−0.000403, 0.001989] | within ±0.0046 | pass |

All 160 per-seed metrics and the comparison exactly match the prior
[prospective large CUDA result](../2026-11-16-phase4-lm1b/cohort-campaign-v1.md#prospective-cuda-results-medium-and-large).
The compact [summary](lm1b-large-cuda-retest-summary-v1.json), without generated text, has
SHA-256 `4a9b760b61cd127cef434728fc9704613b6fef940c486e9fb0730aac8a63d03d`. Raw captures,
per-seed scores and receipts remain ignored under `reports/phase5-lm1b-large-cuda-retest-v1/`.

| Item | SHA-256 |
| --- | --- |
| Campaign manifest | `b0041c08ebf22b2eb0cb20b51f04ad70d677fe4556f7044fa65141384aeb2000` |
| Python capture receipt | `cffab1c13403d92972fa39d9e53a97a6bd4aad4dbfcf46a9ef303589931a77cc` |
| Rust capture receipt | `166c8e49c3189595c84eb952fe99584e6515ef095e2dbfd720c3628924876a3f` |
| Score receipt | `675b76e5f63fdd5eb930b926e3376e3e9ca7a5e824fb2a8402c1d2aab1b87df9` |
| Rust campaign runner | `59501cd655040b541965bb3c83c5fbdb205fcab901e2aa5b7c456018173e796a` |

The manifest was initialized at `eb3dfa3` using `--mode retest --size large`. Capture, scoring
and summary ran with `setpriv --inh-caps=-all` and the documented campaign environment; summary
replayed byte-for-byte. The desktop GPU was shared and not continuously monitored. This is a
scoped CUDA cohort-runner quality retest, not full CLI cohort equivalence, performance, Metal or
Phase 5 completion.
