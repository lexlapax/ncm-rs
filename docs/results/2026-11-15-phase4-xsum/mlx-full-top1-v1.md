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
under `reports/phase4-xsum-top1-row-cuda-v1/`. The corresponding Metal harness
has not yet run; these CUDA traces do not resolve the Metal differences.
