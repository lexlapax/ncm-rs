# XSum Metal top-1 full-corpus comparison

At pushed baseline `e14ae02`, the Metal CLI generated all 11,334 XSum rows in
contiguous batches of 32. The capture completed on an Apple M1 Ultra (64 GiB,
macOS 26.7.3, MLX 0.32.2); the input dataset, artifact, binary and capture script
hashes matched their declared identities after the run. The report directory was
copied to Linux and every file's SHA-256 matched the Mac. It is retained under
`reports/phase4-xsum-mlx-full-top1-captured-v1/`; its manifest hashes to
`b0e6efa956268e3e424ac5a37a11f60b860bcedf8b2c1ac7661a2f4dc4d91923`.

**The frozen raw-ID equality rule failed.** Against the previously scored Rust
CUDA capture, Metal differs in 10 code positions across rows 2690, 4937, 9813,
10009 and 10015, and six final IDs across rows 4883 and 4937. Against the pinned
Python CUDA scored output, there are nine code differences across rows 4937,
9813, 10009 and 10015, and the same six final-ID differences. Metal matches
Python on row 2690, the previously investigated CUDA tie. The full comparisons
and their retained nonzero exits are `reports/phase4-xsum-mlx-full-top1-compare-v1.json`
(SHA-256 `6bd777d271b5cc774e42bc7b62c9af57d36ec7e4c305d9be7a24834aea62b9b1`)
and `reports/phase4-xsum-mlx-full-top1-python-compare-v1.json` (SHA-256
`c6abd9084889fafc2b6eb804e2d7bbb0f5e37dc9d4af743c022fc3b40472a5d6`).

All six raw-ID differences occur after the first EOS in their rows. Applying the
pinned evaluator's EOS truncation and PAD filtering produces identical token ID
sequences for all 11,334 rows. Therefore the same deterministic decoder receives
identical inputs, but the exact raw-ID rule is still a failure and the Metal scorer
was not independently rerun. In particular, row 4883 has identical generated
codes and one post-EOS final-ID difference; row 4937 has both code and post-EOS ID
differences. Their causes are not established. This comparison used contiguous
Metal batches while the retained CUDA capture used interleaved shards, so batch
shape/order effects also require investigation. No tolerance, quality margin,
reference or failed result was changed.

The [capture setup and smoke result](mlx-top1-capture-progress-v1.md) record the
initial system-Python failure and successful 32-row check. Metal stochastic
quality is being captured separately under the [frozen comparison plan](mlx-stochastic-quality-plan-v1.md).
Full candidate-owned-loader Metal validation, quality resolution and repeated
Metal time/memory gates remain open.

## CUDA fixed-history diagnostic

Pinned Python CUDA was rerun at batch 2 for rows 4883 and 4937, forcing the
previously saved Python codes. Its penalized IDs match the saved Python output
for both rows despite the changed batch shape. The trace SHA-256 is
`42dc76b4ed384a76f3dc375890d86c0e33f8ad74a1e2b0753ba3c3f6524d1895`; the trace
and report were copied to the Mac and hash-verified for a later Metal comparison.

A CPU FP64 penalty reconstruction reproduces all 96 CUDA trace IDs. At row 4883,
position 40, before any raw-ID disagreement and with identical free-generation
codes, CUDA's penalized top two scores differ by approximately `7.35e-5`; the
Metal-selected ID is the second candidate. This establishes a narrow CUDA margin,
not the Metal logit values or the cause of the crossing. Row 4937 has different
free-generation code histories; under *forced Python codes and CUDA token history*
the CUDA winner exceeds the Metal-selected ID at position 34 by 2.59, so this is
not a same-input explanation of its observed free-generation difference. Later
positions also have differing token histories. Full numerical statistics require
the pending Metal fixed-history trace; no tolerance changed.

The calculation is `tools/reference/xsum_top1_cuda_margin.py --output
reports/phase4-xsum-top1-cuda-margin-v3.json`, SHA-256
`ac9fed285300d854eaa76c1e068d9e649b74fa399c62f305139a0f1ab27de04f`.
Its first attempt failed while serializing a NumPy integer and left a partial v1
file; v2 succeeded, and v3 additionally verified the Metal manifest and used the
original unigram directly. All are retained. This diagnostic does not establish
that the full raw-ID gate passes.

A subsequent strict-CUDA CLI diagnostic ran the same contiguous batches starting
at rows 4864 and 4928 used in the Metal capture. Across both 32-row batches,
CUDA matched the retained interleaved Python capture on all codes and IDs; Metal
still differed from this same-contiguous-batch CUDA run in one ID at row 4883
and three codes plus five IDs at row 4937. This narrows a simple input-batching
explanation for these two rows on the current CUDA path; it does not isolate the
Metal operation or cover the other code-difference rows. The candidate-owned
loader was used in CUDA, and the changed CUDA driver is disclosed in its report.
The source-bound result is `reports/phase4-xsum-top1-batch-shape-cuda-v1/result.json`,
SHA-256 `31ee0b0d4a0f521e832ffa82a01f96537a358e6e08e312f498636abe61a579f1`.

For row 4937's first code divergence (global position 8, level 1 position 4),
the saved batch-32 Python choice is 3182 and the batch-32 Metal choice is 3690.
In the separate batch-2 CUDA teacher run with the saved Python history, 3690 is
the argmax and exceeds 3182 by only `2.861e-6`; 3182 ranks second. Thus even the
CUDA teacher's head ordering differs from the saved batch-32 choice at this
near tie. This is evidence of sensitivity to the changed execution context,
not proof that batch size alone caused the Metal result or that subsequent code
and token differences are explained. The source-bound observation is
`reports/phase4-xsum-top1-cuda-head-margin-v1.json`, SHA-256
`236960b6c503c259eaf098d3f81271d67c743507363ea97f7be1f03e0ec5ecbb`.

The Metal fixed-history probe subsequently ran on the two saved Python histories
at batch 2. All 6,799,360 float values were within the existing accumulated band,
all 400 level argmax decisions agreed, and all 96 penalized IDs matched. The
result SHA-256 is `adddda7195dd904dc0be10c54e9ab15425c7013bf840824e2278700fc66b1e86`.
This bounded teacher-forced result does not reproduce or erase the batch-32 free-
generation failures. An initial build failed because `cargo` was absent from the
noninteractive PATH; a second failed because it omitted the documented prebuilt
MLX environment and attempted an unavailable Metal compiler. The existing binary
passed the original teacher control; a subsequent build with the documented
process-local prebuilt environment succeeded and produced the same diagnostic
result. All failures and raw reports are retained under
`reports/phase4-xsum-top1-divergence-mlx-captured-v{1,2,3,4}/`.

A further forced-history harness was prepared to compare one selected row from
the same contiguous 32-row batch on each backend. The strict-CUDA release build
and focused Clippy passed. On real CUDA, the selected penalized IDs matched the
free-generation CUDA IDs for both rows. Source and input hashes passed postflight;
row-4883 and row-4937 trace SHA-256 values are respectively
`dcb52412e2d548d858d9ca742adfc4101db6f289760f6647e44131f955633236` and
`69f7e1a07332738f9448eb9682010af06c2223b78971a1a199d978af12ff8fc8`,
under `reports/phase4-xsum-top1-row-cuda-v1/`. These CUDA traces alone do not
resolve the Metal differences; the subsequent Metal comparison is described below.

## Same-batch forced-history comparison

After the new row tracer passed Linux CUDA validation, pushed commit `d018aa3`
was fast-forwarded into the clean Mac checkout. Full `tools/check.sh --mlx`
passed, including 18 real-Metal operator cases. The CUDA and Metal tracer then
used identical contiguous 32-row prefixes and the saved CUDA code histories,
selecting rows 4883 and 4937 respectively. Each row's 3,233,792 compared head
and final-logit floats stayed inside the existing accumulated band. Independent
NumPy FP64 selection matched each native tracer's selected IDs.

For row 4883, all head argmaxes matched. The first and only forced-history ID
difference was position 40: with the same preceding IDs, CUDA's penalized score
for 50256 minus 3646 was `+6.2059e-5`, while Metal's was `-1.1374e-5`. The
forced Metal IDs exactly match that row's free-generation Metal IDs. This directly
identifies a score-order crossing on this forced execution, while not proving a
specific underlying operation is responsible.

For row 4937, there was one head argmax difference: level 1 position 4. CUDA's
logit for 3182 minus 3690 was `+9.5367e-7`, while Metal's was `-4.7684e-6`.
With CUDA codes forced, the two backends' final selected IDs matched; Metal's
forced IDs differed in five places from its free-generation Metal IDs. This is
consistent with the early head difference contributing to later divergence,
but does not independently prove every downstream cause. The exact full-corpus
raw-ID gate remains failed; no numerical threshold was changed.

The per-row results are [row 4883](mlx-top1-row-4883-v1.json) (SHA-256
`7f00c6519038e2484c96c6d61b034ec1ed14bbe29cf70adfc38d7ebce57b31bf`) and
[row 4937](mlx-top1-row-4937-v1.json) (SHA-256
`aeed8ab90ba430bb818632ca88b4b8cd07258c741fbcf6f3fd510c24f5415003`).
The local and Mac reports, trace hashes, and pre/post input hashes remain under
`reports/phase4-xsum-top1-row-{cuda-v1,mlx-captured-v1}/`. The Mac check log
`reports/phase4-owned-loader-mlx-check-captured-v1.log` hashes to
`10c62f43e88d150f89641cdc4200feacb93b5d0eb1abd0e4da30a04e80a5eaa8`.

While the next CUDA candidate was being checked, both baseline Metal traces
were repeated on the clean `d018aa3` Mac checkout with the same binary, batch
inputs and CUDA histories. The input manifest passed SHA-256 checks before and
after both runs. The JSON and safetensors hashes for each row exactly match the
originals listed in the per-row results above; stderr was empty. The repeated
files remain on Mac under `reports/phase4-xsum-top1-row-mlx-repeat-v1/` and were
copied to local `reports/phase4-xsum-top1-row-mlx-repeat-captured-v1/` with
matching hashes. The
command pattern was `target/release/examples/xsum_top1_row_trace
artifacts/canonical/xsum-inference-rope-v2-final INPUT CUDA_OUTPUT SLOT OUTPUT`,
with `reports/phase4-xsum-mlx-full-top1-v1/input-04864.json`,
`reports/phase4-xsum-top1-batch-shape-cuda-v1/cuda-4864.json`, slot `19`, and
output `reports/phase4-xsum-top1-row-mlx-repeat-v1/row-4883.safetensors` for the
first row; the second used `input-04928.json`, `cuda-4928.json`, slot `9`, and
`row-4937.safetensors` in the corresponding directories. This shows repeatability
for these two forced-history executions, not a resolution of the full-corpus
raw-ID failure.

On the same checkout, `target/release/ncm sample-xsum
artifacts/canonical/xsum-inference-rope-v2-final mlx INPUT` was also repeated for
both `input-04864.json` and `input-04928.json`. All codes and IDs in each
32-row result matched the corresponding original Metal batch result exactly.
The binary SHA-256 (`aaba9fa090bdda88d4b757994ac3b609f292f01bdc0825ec027f7137c517bc69`)
and the prior trace input manifest passed postflight checks. The two new stdout
hashes are `13126af2482cd1a8bb7bd83bd382557b01b10d8fa1d718654ccc339629d4cce5`
and `1662e6869c012e017df154da76d72d83f634123dd86b874010e73e6335738050`,
retained on Mac in `reports/phase4-xsum-top1-free-mlx-repeat-v1/` and copied
with matching hashes to local `reports/phase4-xsum-top1-free-mlx-repeat-captured-v1/`. This is
repeatability against the original Metal output, not equality with CUDA or Python.
