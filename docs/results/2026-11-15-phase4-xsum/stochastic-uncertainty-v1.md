# XSum stochastic quality uncertainty diagnostic

After the [frozen held-out comparison failed](stochastic-heldout-v1.md), a post-hoc bootstrap separately resampled the five random seeds with rows fixed, the 256 input rows with seeds fixed, and both together. It used 10,000 draws for each mode, independently resampling Python and Rust seed indices and sharing resampled rows. The [result](stochastic-uncertainty-v1.json) and [reproducible script](../../../tools/reference/xsum_stochastic_uncertainty.py) report signed Rust-minus-Python differences; these diagnostics do not change the precommitted gate or constitute a new quality test.

| Metric | Seed-only difference SD | Row-only difference SD | Both resampled SD |
| --- | ---: | ---: | ---: |
| ROUGE-1 | 0.188 | 0.281 | 0.436 |
| ROUGE-2 | 0.169 | 0.216 | 0.344 |
| ROUGE-L | 0.287 | 0.254 | 0.458 |
| Generated-text perplexity | 7.956 | 6.689 | 11.828 |
| Mean words | 0.116 | 0.157 | 0.236 |

Both input selection and sampling seeds contribute to the uncertainty. The decomposition is exploratory, conditional on these rows and five seeds, and its components need not add independently. It does not establish why Rust's observed ROUGE means are lower, rule out a model difference, or show that any larger study must pass. Further evidence should use fresh rows and predeclared methods while preserving the existing failed result and its margins.
