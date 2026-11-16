# XSum pinned CUDA tie selection

The `xsum-inference-v2` reference backend now selects the lowest index among
equal finite maxima for temperature-1/top-k-1/top-p-0 generation. This follows
the [inspected pinned PyTorch 4096-way CUDA sort implementation](cuda-tie-order-screen-v1.md),
including signed-zero ordering. It is scoped to the explicitly selected strict
CUDA/cuDNN reference backend; the earlier artifact profile still rejects exact
ties, and non-finite logits still fail explicitly. No general PyTorch or other
backend tie guarantee is claimed.

The pinned PyTorch sampling sequence and the native kernel agreed on all
1,024 finite synthetic tied rows, with tie counts from 2 to 4,096 and signed
zeros. Their safetensors reference and native result SHA-256 values are
`d7a65015189f546b922df11fe3cf280dae26c743b1c74995b86db5f60ee23be9` and
`1b4c9fceb98cf433f230932a81c1f0375bdff56aa783c25c3ca5b443674b4536`, under
`reports/phase4-xsum-tie-reference-v1/`. The focused real-CUDA regression
also verifies that the older selector rejects ties, the new selector returns
the first index, and both reject non-finite rows. The full `tools/check.sh
--cuda` passed with 18 real-device operator cases; its log is
`reports/phase4-xsum-reference-tie-check-v2.log` (SHA-256
`df599c1164b8b45808dd46e02b205da2ec52d80eaeda2393eef7276b270f3693`).
An earlier full run failed an unrelated process-observer fixture, which passed
in isolation before the successful rerun; both logs are retained.

The existing 256 development rows match Python on all codes and IDs, and
their ROUGE and perplexity scores are identical. The source-bound comparator
output SHA-256 is
`890630f934660d2e0f6df97fbf958d0e759a9784d96f13bf76169b3ba514ee92`.
An initial comparison against a score from an older scorer version was
correctly rejected; the Python outputs were rescored with the current pinned
evaluator before the successful comparison.

One-process, two-warmup/five-trial screens at batches 1/4/32 matched all
checked codes and IDs. Rust/Python median times were 161.32/197.75,
587.77/694.66 and 4531.48/5556.14 ms; whole-process peak RSS was
1062.66/1861.11, 1062.68/1859.46 and 1062.38/1863.07 MiB. They favor Rust
in these cells but are not the repeated performance gate; desktop activity was
not monitored. Raw reports and source/binary/input hashes are under
`reports/phase4-xsum-reference-tie-performance-v1/` and matched postflight.
The full current-source quality and three-process performance comparisons,
broader stochastic qualification and MLX remain pending.

The subsequent full tie-enabled CUDA capture completed all 11,334 rows. Every
raw output ID matches Python; the only intermediate-code difference is the
previously investigated row-2,690 tie. Both shard source and linked-library
checks passed. Their manifest SHA-256 values are
`ae580e8ac25b59c2b782f3aeade72daf24ffbe8f4bd637d882f671607bfb2240` and
`393be26641af3531fef4d14d5129a314e090c52fbe92a9cb2202d9edf448bd2f`, in
`reports/phase4-xsum-reference-tie-full-v1-s{0,1}/`. Full and held-out scoring
and the current-source three-process performance run are in progress or pending;
the capture alone is not a quality or complete runtime gate.

Full and held-out scoring has since passed the frozen quality rule: all output
IDs, decoded texts, per-row ROUGE and aggregate metrics match Python. Full
ROUGE-1/2/L is 34.9236/12.7078/27.9258 and GPT-2 perplexity is 82.6234;
held-out rows 256–11,333 score 34.9312/12.7069/27.9256 and 82.6764. The full
and held-out comparison JSON files under `reports/` hash to
`6c109f0c726d9f40e2e6ce125e2c82c448cd487e997b89e30a6c39177fdd4391` and
`66038ace8f7c67c4c2d9844add596b63a8bf0e5065b75b781eb2be79b04ea2c7`.
The one intermediate-code disagreement remains the investigated Python tie;
the current-source repeated performance run and MLX remain pending.

Reproduce the synthetic comparison with `tools/reference/xsum_tie_reference.py
--output NEW_REFERENCE` under the pinned Python environment, followed by the
strict-cuDNN `xsum_tie_probe NEW_REFERENCE/reference.safetensors` with the
local cuDNN library path. The quality and performance screens use the same
commands and data as the [previous current-source comparisons](cuda-final-repeated-v1.md),
with fresh output paths.
