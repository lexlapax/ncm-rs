# XSum stochastic held-out quality: not passed

Using the [precommitted protocol](stochastic-protocol-v1.json), Python and Rust each sampled 256 previously unused rows with five seeds, temperature 1, top-k 0, top-p 1 and CFG 0. The runtimes use distinct RNGs. The [machine-readable comparison](stochastic-heldout-v1.json), SHA-256 `407a684ed22224f95b6c03d00026dbb201a0b1839ecb5eec514e32eae70c42ee`, records the per-seed metrics, score hashes, observed degradation and 10,000-replicate bootstrap bounds. The per-row scores, source-bound inputs and sampled IDs are retained under `reports/phase4-xsum-stochastic-qualification-heldout-{python,rust}-{seed}*`.

| Metric | Python five-seed mean | Rust five-seed mean | 95% degradation bound | Frozen margin | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| ROUGE-1 | 31.1869 | 30.9443 | 0.9663 | 0.62 | Fail |
| ROUGE-2 | 9.7775 | 9.6081 | 0.7390 | 0.64 | Fail |
| ROUGE-L | 24.3329 | 23.9235 | 1.1689 | 0.62 | Fail |
| Generated-text perplexity | 196.1287 | 187.4774 | 10.6028 | 12.52 | Pass |
| Mean words | 20.3609 | 20.5008 | 0.5414 | 0.30 | Fail |

ROUGE degradation is Python minus Rust; perplexity degradation is Rust minus Python; word count uses the absolute difference. All four observed mean differences for failed metrics are within the respective margins, but the frozen uncertainty bounds are not. Therefore the result does not establish non-inferiority under the stated rule, nor by itself prove a real quality regression. Do not silently widen the limits or label this a pass. Five-seed variability and input-sampling uncertainty need further investigation; a new independent experiment would be additional evidence, not a retroactive pass of this one. The earlier near-boundary exact-decision failures remain visible. Repeated stochastic performance and the XSum Linux and MLX gates remain pending.
