# XSum CUDA reconstruction, projections and attention

An isolated candidate at base revision `8ba381f` reproduced the pinned CUDA
reference on two retained eight-row controlled-draw runs (CFG 0 and CFG 1,
seed 42). Each run matched all 1,600 codes and 384 penalized IDs. The combined
candidate also matched the retained reference on 04864 and 04928, two
previously investigated 32-row top-1 batches. Default inference still uses
the existing implementation.

The combined result required four changes to model arithmetic and a change
to the controlled inverse-CDF draw.

| Operation | Observed difference and correction |
| --- | --- |
| Linear reconstruction | Fused coordinate and interpolation arithmetic matches every FP32 output in the seven retained input lengths (6, 10, 14, 20, 26, 32, 40 to 48). The corrected forced history matches all nine reconstruction stages. The interpolation change alone worsens the retained guided run. |
| Target projection | A 1 MiB cuBLASLt workspace preference matches pinned PyTorch across 18 tested prefix and target shapes. The existing 4 MiB preference differs in some shapes. |
| Masked attention | Scaling both Q and K by the square root of 0.125 before the score matmul and dividing exponentials by the FP32 softmax sum matches the retained level-eight attention exactly. |
| MLP output projection | For the captured `m=768, k=3072, n=4480` product, 1 MiB matches pinned PyTorch on the compared row. The prior preference changes 404,366 of 430,080 values after the residual. With this correction the full 12-block retained row matches bit-for-bit. |
| Controlled draw | FP64 inverse CDF resolves a retained FP32 boundary discrepancy. It uses the existing SplitMix64 uniform bits and does not implement pinned PyTorch's FP32 top-p filter or its multinomial random stream. |

A paired exploratory performance screen used identical inputs at CFG 0,
stochastic seed 42000, two warmups and five measured iterations per runtime
and batch. Median time changed from 0.5535 to 0.5610 seconds at batch 4 and
from 4.3016 to 4.3869 seconds at batch 32 (1.4% and 2.0% higher). These timings
use an RTX 3060 Ti without activity telemetry. No acceptance benchmark or
Metal experiment is inferred from this screen.

Raw retained artifacts are under ignored
`reports/phase4-xsum-combined-numerics-candidate-v1/`. The isolated CLI build
used `CUDA_COMPUTE_CAP=86 RUSTC_WRAPPER= cargo build --locked --release -p
ncm-cli --features cuda-cudnn` with the existing cuDNN library on the link
path. The CLI SHA-256 was
`70f3307aac41680d5d20cca9edef911dfd13ed2843189e815fbacfdadf1ab137`.
The separately emitted `runtime.patch` and `xsum_linear_fma.rs` retain the
source for that CLI, excluding diagnostic trace hooks.

## Same-input Metal attention

On Mac Studio, three Metal formulations processed the same retained Q/K/V and
additive mask. The captured CUDA row occupied slot 3 in the original batch
of eight. The remaining rows' Q/K/V were zero-filled. For the 430,080
reference coordinates, the GPU results were:

| Formulation | Different FP32 values | Maximum absolute difference |
| --- | ---: | ---: |
| Existing fused attention, scale applied by the kernel | 378,750 | 3.5763e-7 |
| Fused attention with Q and K scaled separately | 369,095 | 2.9802e-7 |
| Separate QK matmul, precise MLX softmax and value matmul with split scale | 306,936 | 2.3842e-7 |

Splitting the scale moves the operation closer to the CUDA result, but the
composed path still differs and has not been timed in the full model.
This is a single operator comparison on an already observed input, not a
fix for the exact Metal top-1 gate. `dotsetup.rmlx` was running and was
not interrupted.

Input SHA-256
`b54748a90c8c7bd3c7e563cea921d3a267b2d005834822a9a918ea70d1c2a46b`.
The receipt SHA-256 is
`347c890c354e592474d9996257c9163243d1e0269699bd729a268a03c58dc716`.
The same-input probe and receipt are retained with this report's ignored
evidence directory.

## Bootstrap self-control

As a post-hoc precision check, both sides of the frozen 256-row bootstrap
used the same original Python score arrays. Resampled rows were shared
and seed indices were drawn independently for the two sides, as in the
frozen comparison. The actual differences between the inputs were zero.
Three upper bounds still exceeded their margins: ROUGE-L 0.7161 against
0.62, generated-text perplexity 20.1768 against 12.52, and mean words
0.3696 against 0.30. The ROUGE-1 and ROUGE-2 bounds were 0.5998 and
0.4396, within 0.62 and 0.64.

This checks what the bootstrap can resolve on this particular set of five
score arrays. It provides no new sampled rows or independent study and
does not revise the frozen comparison. The original scores and protocol
are hash-bound in the retained `python-self-control.json` receipt. The
[existing CUDA advancement decision](user-accepted-exceptions-v1.md)
applies to the prior scoped runtime and comparison.

## Frozen stochastic retest

The isolated CLI and pinned Python regenerated all 256 rows at seeds
`42, 0, 1, 2, 3` using the unchanged [frozen
protocol](stochastic-protocol-v1.json), capture, scoring and comparison
tools. The Python seed metrics reproduce the [original study](stochastic-heldout-v1.md)
exactly. All five seeds and rows were already observed before the candidate
was developed.

| Metric | 95% degradation bound | Frozen margin | Result |
| --- | ---: | ---: | --- |
| ROUGE-1 | 1.020880 | 0.62 | Fail |
| ROUGE-2 | 0.751402 | 0.64 | Fail |
| ROUGE-L | 1.194938 | 0.62 | Fail |
| Generated-text perplexity | 10.694817 | 12.52 | Pass |
| Mean words | 0.535156 | 0.30 | Fail |

The capture manifests bind both runtimes to the same source, binary,
driver and CUDA libraries. All outputs and scores are retained beneath
`cli-retest/frozen-stochastic-retest-v1/` in the ignored evidence directory.
The comparison SHA-256 is
`96d9b702141cc8969886cbf675f1604bc4aafdedb54c1367f41d87e9d51a39fc`.
The controlled arithmetic improvement did not clear the quality
comparison. The candidate remains isolated and has no full performance
or Metal qualification.

## Register-resident masked softmax

A separate CUDA kernel keeps the masked scores and exponentials in registers.
Four warps share a block and each warp still processes one row in the same
reduction order. The arithmetic of the existing CUDA softmax is preserved,
including the FP64 reciprocal rounded back to FP32 before multiplication.

On the RTX 3060 Ti, an isolated paired GPU-event screen found exact FP32
agreement across all five synthetic shapes, including broadcast and
per-batch masks. It used four warmups and ten measured pairs, each timing
ten launches, alternating the order within each pair.

| Batch | Sequence | Original kernel (ms) | Register kernel (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 512 | 0.125 | 0.081 |
| 1 | 516 | 0.108 | 0.084 |
| 4 | 544 | 0.571 | 0.384 |
| 4 | 552 | 0.530 | 0.392 |
| 32 | 560 | 4.931 | 3.284 |

A separate register kernel preserving the experimental split-scale division
also matched on these shapes and reduced latency in the same screen.
The raw operator timings and standalone sources are under `cuda-softmax/`
in the ignored evidence directory.

The runtime dispatch uses four warps per block for sequences 512–560.
Sequences 561–768 use the original kernel. The focused CUDA tests match
the existing operation sequence bitwise for finite results, including the
dispatch boundaries and both mask layouts, and agree on NaNs for fully
masked rows.

A paired full-generation screen used the existing 32-prefix fixture,
stochastic CFG 0 and seed 42000. Each executable ran in a fresh process
with two warmups and five measured requests per batch. All codes and
penalized IDs were identical in each pair.

| Batch | Original median (s) | Register kernel median (s) | Lower latency |
| ---: | ---: | ---: | ---: |
| 1 | 0.152513 | 0.150441 | 1.36% |
| 4 | 0.556258 | 0.543942 | 2.21% |
| 32 | 4.313312 | 4.279571 | 0.78% |

This screen establishes a local timing observation on the tested
workload. Full-corpus output comparison, repeated performance and
process-memory acceptance measurements are separate. The timing reports,
input/binary hashes and GPU snapshots are under ignored
`reports/phase4-xsum-softmax-model-screen-v1.hWdESj/`.

`CUDA_COMPUTE_CAP=86 RUSTC_WRAPPER= tools/check.sh --cuda` passed.
It ran 524 report and data tests and 18 real-device operator cases.
The validation log is
`reports/phase4-xsum-softmax-cuda-check-v1.7Agg2R.log` (SHA-256
`53bc15e6ed980c44e357b540f06157e1f089f5213de8a1a286d27f1edfec6289`).
The source-bound full-corpus retest completed with both shards passing
postflight. All 2,266,800 intermediate codes and all 544,032 raw IDs
across 11,334 rows match the retained tie-enabled CUDA output at the same
batch layout. The preserved CUDA baseline matches every pinned Python
raw ID. The comparison is in
`reports/phase4-xsum-softmax-full-corpus-v1.zYw5qb/comparison.json`
(SHA-256
`dabf28bdb69f1e08fcd937b52f85ba2ec894f034843bec48e7dc80066ce62486`).

## Metal free generation with smaller batches

The same borrowed-upload Metal executable used for the retained full corpus
generated the previously differing rows at batch sizes 1, 2, 4, 8 and 16. Each
sub-batch is a contiguous aligned slice of the original 32-row input. At
batch sizes 1, 2, 4 and 8, row 4883 matches every saved CUDA code and raw ID.
Metal differs by one raw ID at batch sizes 16 and 32. Row 4937 still
differs from CUDA by three codes and five raw IDs at every tested size.
Neighboring rows in these ten runs match CUDA. The probes used ordinary
top-1 generation and did not force any histories.

The Mac executable SHA-256 is
`a580ba2ca6c8a2e1a43d4e7a0b3a7b6645b6576f7424ec2ea3adb711a2429414`,
the same as in the retained batch-32 capture manifest
`4b27beee855cc27738b62f1c5e6fa5dc2c68f1aeaff0788bf254569f711ece6e`.
Input and artifact hashes matched before and after. The comparison receipt
`c7827e0a483b755c6b16c7c647316ee8534aa87644a31e55c56a81149074cf07`,
probe and raw outputs for sizes 1, 2 and 4 are under `metal-batch/` in the
ignored evidence directory. Sizes 8 and 16 are under `metal-batch-8-and-16/`
with receipt SHA-256
`bdeddce2fdaf62c7069b87909b5d969999c1ccace177f7a845ef053b981100de`. The first invocation found that the locally named fixture
directory was absent on Mac. The completed run used explicit hash-checked
copies of both fixture files. The Mac service was left running. This rules
out the tested smaller-batch choices as a fix for both cases.

The repeated top-1 latency and memory comparison against pinned Python
started after full-corpus equality passed. Its manifest and process events
are under `reports/phase4-xsum-softmax-repeated-v1.YUcYdf/top1/`.

## Two Metal controls for row 4937

The six-position latent averages eight consecutive FP32 reconstruction
values at each channel. Metal's current mean and a safe Metal kernel
that adds in the same order as the CUDA continuation agree on all
13,824 compared latent values. The comparison covers the original CUDA,
default Metal, and refined Metal reconstructions, each at batch 1 and
at slot 9 in the original batch-32 geometry. Both Metal formulations
also reproduce the retained latent for their corresponding input. This
rules out the reduction order on these inputs. The upstream
reconstructions can still differ. The hash-checked input and receipt
are under `metal-area-order/` in the ignored evidence directory. The
receipt SHA-256 is
`e30b5a1d7cfdcc3e74cc2b1f02301c5ac11bc65101f23b633ec56a8eb2d0dec9`.

The existing adapted Python MLX model then generated rows 4936–4943 with
its usual masked attention and with both Q and K scaled before the score
matmul. Both forms produced the same 1,600 codes and 384 raw IDs as the
retained Rust Metal batch of eight. Both retained row 4937's three code
and five raw-ID differences against CUDA. The split-scale change improves
the earlier isolated attention comparison but does not change these
generated outputs. The Mac service remained running. The receipt and
both raw results are under `metal-attention-model/` in the ignored
evidence directory. The receipt SHA-256 is
`2e00d050b6979d0858723290e80004a02ede3f56cdc5ff010676069b80e5c4ad`.

A subsequent model probe combined the split attention scale with the existing
refined Welford normalization and the tested two-part target projection. It
used the same adapted Metal model and two retained batches of eight. Row 4937
still differs by three codes and five raw IDs. Row 4883 again differs by one
raw ID, whereas the retained size-eight Metal run matches CUDA for that row.
All neighboring rows match. The isolated combination was left out of the
runtime. The input and source hashes, receipt
`dc5c8bc6b4946437348d7f94d892c78ccc9cf554bdbc6eec59d528252857f88e`,
and outputs are under `metal-combined-model/`. The Mac service was not
interrupted.

## Full-corpus Metal text and ROUGE replay

The pinned scorer's tokenizer and stemmed ROUGE evaluator independently
decoded and scored the retained borrowed-upload Metal outputs. All 11,334
decoded texts and all per-row ROUGE scores match the retained Python CUDA
result. ROUGE-1/2/L is 34.92361687812744 / 12.707781261521596 /
27.925757564995436, and the mean word count is 20.657314275630846. The
pinned data, candidate capture receipts, native Python score report and
scorer assets were verified. The first replay used repeated addition
to aggregate scores and differed at the last decimal places despite
identical per-row scores. The corrected replay uses Python's built-in
sum in the same order as the stock scorer and matches all aggregates
exactly. Both traces are retained in `metal-rouge/`.

The six raw-ID differences remain at row 4883, position 40, and row
4937, positions 34, 42, 44, 46 and 47 (all positions zero-based). They
follow the first EOS at position 25 in both runtimes and rows. The replay did not regenerate Metal
outputs or rerun generative perplexity. The exact raw-ID requirement
still fails. The receipt SHA-256 is
`1ea72ec6974c52c4ce84c0e54e813d9da61f1c00575df386514784e9eac63aa1`.

The pinned perplexity input check independently prepared batch-32 GPT-2
input IDs, attention masks, offset maps and shifted score masks on CPU
using the official empty-prompt settings. These tensors match for all
11,334 rows, and both streams contain the same 277,595 scored tokens
as the retained Python CUDA evaluation. Its perplexity is
82.62335629362389 on these inputs. This verifies that the existing score
uses the same evaluator inputs. It does not constitute a second GPT-2
inference run. The check and receipt are in `metal-ppl-inputs/`, and the
receipt SHA-256 is
`ef8711d294f5eca666fc532fcd783ad26ff8c3ee082774f41a02c66abdc3e25a`.

The optimized CUDA CLI matched all 256,000 codes and 61,440 raw IDs in
the 1,280 retained sampled outputs. It used the unchanged exact five-seed
protocol with seeds 42, 0, 1, 2 and 3 and the same 256 rows per seed. This
establishes output preservation on these observed inputs without
rescoring identical samples. The original stochastic failure and the
separate user-approved advancement decision keep their existing scope.
The comparison is in
`reports/phase4-xsum-softmax-seeded-retest-v1.AIttVa/comparison.json`
with SHA-256
`e71c256a96b0930ea318ba40c75d60be0d58610be8b2e4087fbc97a5490da556`.

## Repeated CUDA top-1 performance

The optimized binary and pinned Python ran the same 32-prefix fixture on the
RTX 3060 Ti. Each size used three fresh processes for each runtime, with
alternating order, five warmups and 30 synchronized generations per
process. Codes and raw IDs matched across all runs at each batch size.
Generation includes code and ID readback and excludes load. Python cleanup
is measured separately. RSS includes startup and load.

| Batch | Runtime | Median ms | Pooled p95 ms | Peak RSS MiB | Sampled GPU peak MiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Rust | 155.74 | 156.30 | 990.22 | 838 |
| 1 | Python | 191.43 | 191.99 | 1780.13 | 990 |
| 4 | Rust | 558.36 | 559.43 | 994.49 | 1062 |
| 4 | Python | 691.53 | 692.68 | 1780.95 | 1624 |
| 32 | Rust | 4327.02 | 4335.09 | 993.21 | 3048 |
| 32 | Python | 5332.18 | 5339.87 | 1785.23 | 5966 |

Medians are medians of process medians. RSS and sampled driver GPU peaks
are medians of per-process peaks. The driver sampled once per second and
can miss brief peaks. Native allocator observations are retained separately
because the CUDA async pool and PyTorch allocator cover different
allocations. The 1,200 telemetry samples had no collector failures.
Four other GPU processes used 327 MiB in total across the observations.
This was not an exclusive run.

Rust reaches 6.42, 7.16 and 7.40 samples per second at the three sizes,
compared with 5.22, 5.78 and 6.00 for Python. The receipt includes per-process
dispersion, raw output positions per second, and text positions per second
after EOS termination and PAD filtering. The full-corpus equality result
and this benchmark concern CUDA. The earlier short previous-Rust screen
and the paired control below measure the kernel change separately.

Reproduce with `tools/bench/run_phase4_xsum_bench.py --profile v2 --batches
1 4 32 --cudnn-dir CUDNN --output NEW_DIRECTORY` under the pinned evaluation
environment. The four receipts are retained in
`reports/phase4-xsum-softmax-repeated-v1.YUcYdf/`:

- `top1/manifest.json`: `0597995556e797abdac06bbe845da329cbe0f7de1a9bcfa6f0fb78f712226f97`
- `summary.json`: `4090ed50367a8d5949ebc4e4c1bddbf33f52c5c49ded0551d02902e144392d6c`
- `driver-memory.json`: `97e1073ce212c80bd46074dd9972b9e36853b133b840030b182a9f6ae31870ca`
- `comparison-cells.json`: `28d2af085cea4aeb9c3bb8ecdb44151d97580e48ab12246f9656dfc34a9124b7`

## Paired previous-Rust comparison

The pinned pre-change commit `8ba381f0853700062fb73ed581193308f8455acd`
was rebuilt from a full Git archive. Both Rust binaries used the same
top-1 fixture and native libraries as the repeated Python comparison.
The optimized binary had the same SHA-256 as in that comparison.
All codes and raw IDs match the retained repeated top-1 results.
Each batch used three alternating fresh-process pairs. Batches 1 and 4
each used five warmups and 30 measurements per process. Batch 32 used
two warmups and five measurements per process to limit runtime.

| Batch | Previous Rust median ms | Optimized Rust median ms | Median paired reduction | Previous / optimized p95 ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 156.47 | 153.62 | 1.70% | 157.90 / 154.86 |
| 4 | 565.86 | 557.51 | 1.76% | 568.57 / 558.69 |
| 32 | 4391.86 | 4322.03 | 1.62% | 4398.35 / 4326.02 |

Medians are medians of process medians. The paired reduction is the median
of each pair's relative change. The three batch-32 pairs had reductions of
approximately 1.63%, 1.62% and 1.56%. Median sampled per-process GPU
peaks are identical at 838, 1,062 and 3,048 MiB. At batches 1/4/32,
median peak RSS was 991.66/990.51/994.78 MiB before and
991.83/991.88/991.65 MiB after. The 347 sampled driver observations had
no failed memory queries. Sampling can miss transient peaks. The change
demonstrates lower latency on the tested device and workload without an
observed increase in sampled GPU memory.

The first baseline link failed with a bus error using a temporary target.
Two reduced on-disk archives also omitted embedded fixtures or tokenizer
metadata. The successful build used the complete pinned archive and an
on-disk target. Both rebuilt binary identities and all failed logs are
retained.

The command, baseline commit, candidate patch, input and binary hashes,
timing, and telemetry are under
`reports/phase4-xsum-softmax-paired-prep-v1.VF955t/`. Its
`paired/manifest.json` SHA-256 is
`8c2a7f4d8edb536aa24868c9acb766b4a558d0477a93c223e425bdcae39459d5`.
The independent `paired-summary.json` SHA-256 is
`7c47ddf9268c19905a793b16d68588828a1fa40724d3b34675ee1440b77bf848`.
The Metal exact top-1 gate still fails separately.
