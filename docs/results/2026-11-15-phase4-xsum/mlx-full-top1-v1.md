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

The remaining code-difference rows were subsequently checked in their original
contiguous 32-row batches (starts 9792 and 9984). The explicitly rebuilt
borrowed-upload CUDA CLI matched the retained interleaved Python codes and IDs
on both entire batches. Metal differed from same-batch CUDA by one code at row
9813 and five codes across rows 10009 and 10015, with no raw-ID differences in
either batch. The command was `setpriv --inh-caps=-all env OMP_NUM_THREADS=1
TOKENIZERS_PARALLELISM=false .reference/venv/bin/python
tools/bench/check_xsum_top1_batch_shape.py --starts 9792 9984 --cli-result
reports/phase4-xsum-borrowed-upload-cli-v2/result.json --output
reports/phase4-xsum-top1-remaining-batches-cuda-v2`; its result SHA-256 is
`cf6c7cc6a28151393490930bbfccbe309bc04cea99a053787983109063e9c5d2`.
The v1 diagnostic used the old CLI due to an example-only build; it reached the
same numerical result and is retained, but is not candidate-CLI evidence.

For the three remaining code-difference rows, the strict-CUDA same-batch tracer
was rebuilt on the borrowed-upload source and run against the corrected v2
CUDA outputs. Its forced-history selected IDs match free CUDA IDs for all three
rows, and source/input postflight checks passed. The trace SHA-256 values for
rows 9813, 10009 and 10015 are respectively
`40f280fcb906a70d8d0046cdf2f9d423352baebdb8e7cea3578f12c221ff11f6`,
`dc4045147993aca40de76543fb191618e7b54ebbb4b754eaf9c67bdda6ae55f8`, and
`100e2817844d2542d9ce37ad1a60f522489ae90af2582b04748a2bbd5981b26b`.
The command pattern was `setpriv --inh-caps=-all env OMP_NUM_THREADS=1
LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib"
target/release/examples/xsum_top1_row_trace
artifacts/canonical/xsum-inference-rope-v2-final INPUT CUDA_OUTPUT SLOT OUTPUT`.
Inputs were the captured `input-09792.json` and corrected `cuda-9792.json` for
slot 21, and `input-09984.json` with `cuda-9984.json` for slots 25 and 31;
outputs are under `reports/phase4-xsum-top1-remaining-row-cuda-v1/`.
Corresponding Metal traces were subsequently captured on commit `6d72013`
after the repeated benchmark. The CUDA outputs and batch inputs were copied or
checked on Mac against their Linux SHA-256 identities; binary and input
postflight passed. The Metal trace hashes for rows 9813, 10009 and 10015 are
`22ec53f624df36a3096214e60c62203a222d34e377b55f6469b5b50b0bea099d`,
`3166696abc13b15399aeb2686dcd3b954dcead0c0e1a8be371af81c73e2042e7`, and
`e89c73e05cfa72f88076e4dc7627f6944458189d6162f39d050ba15ad55529a6`.
The Metal command used the same `xsum_top1_row_trace ARTIFACT INPUT CUDA_OUTPUT
SLOT OUTPUT` argument pattern, with the copied CUDA outputs under
`reports/phase4-xsum-top1-remaining-row-mlx-v1/`, the Mac capture inputs and
the same respective slots. Outputs and pre/postflight receipts are retained
on Mac in that directory and copied to
`reports/phase4-xsum-top1-remaining-row-mlx-captured-v1/` on Linux.

The row-comparison helper was extended to accept an explicit CUDA trace, result
and expected hash for these rows while retaining its existing pinned defaults;
it also verifies that both trace results identify the requested row slot.
Focused tests passed, and rerunning the two original row comparisons reproduced
their numerical results exactly after excluding the changed input/source hash
inventory. This prepares the comparison; it is not a Metal trace result.

At each row's first free-generation code difference, the forced CUDA head
selects the saved CUDA code and the saved Metal code ranks second. CUDA's logit
margin over that Metal choice is `2.14577e-5` for row 9813 (level 7, position
17), `6.19888e-6` for row 10009 (level 5, position 15), and `3.33786e-6` for
row 10015 (level 8, position 18). These are read directly from the hashed CUDA
traces above; they identify narrow CUDA margins, not Metal logits or the cause
of the crossings. Later code positions can have different histories. The
calculation loaded each `row-ROW.safetensors` with `safetensors.torch.load_file`,
selected `generator.level_LL.logits[0, POSITION]`, and computed the FP32
difference and rank of the CUDA/Metal code pairs `(2854, 3206)`, `(3758, 3032)`
and `(2298, 605)` respectively.

Each subsequent CUDA/Metal forced-history comparison covered 3,233,792 floats,
all within the existing accumulated band, and found one head argmax crossing
at the first differing code above. For the same CUDA-minus-Metal-choice logit
pairs, Metal's margins were `-3.33786e-6`, `-1.57356e-5`, and `-9.05991e-6`
for rows 9813, 10009 and 10015 respectively. The forced final IDs matched
between CUDA and Metal, and each matched that row's free Metal final IDs. These
observations directly identify score-order crossings in the forced executions;
they do not establish a specific underlying operation or prove every later code
difference. Detailed records are [row 9813](mlx-top1-row-9813-v1.json),
[row 10009](mlx-top1-row-10009-v1.json), and [row 10015](mlx-top1-row-10015-v1.json)
with SHA-256 values `52718240472c792b5b871790029b3f1824d7fc0ad00e9d69f0af69812ebf391c`,
`83c562466b804ca6157c31a6fa5756ab7a34bc67674c7f92db0099cc00a2d388`, and
`4ae309ab8c0e6ad7cc3570b5c4880fbdb1a0dfd6b19d04c1885dd62512b49305`.
The existing exact raw-ID gate remains failed; no tolerance changed.

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

## Pinned Python free generation across batch sizes

The pinned Python model ran on the RTX 3060 Ti with driver 620.70.06,
PyTorch 2.15.0+cu132, math SDPA and TF32 disabled. Each free-generation
call used the frozen top-1, CFG-zero settings and the pinned final decoder.
It used aligned contiguous batches around the five rows with saved Metal
differences. Four batch-32 calls reproduced all 128 retained Python rows
exactly. Smaller batch sizes kept every selected row or neighboring row
exact except for the changes listed here.

| Batch size | Rows compared | Intermediate code changes | Raw ID changes |
| --- | ---: | --- | --- |
| 32 | 128 | 0 | 0 |
| 16 | 64 | 3 at row 4937 | 5 at row 4937 |
| 8 | 32 | 3 at row 4937 | 5 at row 4937 |
| 4 | 20 | 3 at row 4937 | 5 at row 4937 |
| 2 | 10 | 3 at row 4937, 1 at row 10015 | 5 at row 4937 |
| 1 | 5 | 3 at row 4937, 1 at row 10015 | 5 at row 4937 |

At each size from 1 to 16, the full Python row 4937 matches the saved
batch-32 Metal row, including its three intermediate code changes and five
raw IDs after EOS. At sizes 1 and 2, Python also exactly reproduces Metal
row 10015. Python row 4883 keeps its batch-32 IDs throughout. This shows
that two of the Metal discrepancies can occur within the pinned CUDA
model from a batch-size change alone. The frozen reference uses its saved
batch-32 outputs and the exact gate still fails.

### Level-one replays

The CUDA run at batch 32 scores code 3182 above 3690 by `9.536743e-7`
at level 1, position 4. At batch 16, the sign reverses. Both native
choices repeat in the two independent replays.

- Replaying the same captured hidden row in either head batch shape keeps
  the selected code. The linear projection changes with shape and the
  batch-32 hidden row ties when scored in the batch-16 head. Feeding the
  same precomputed projection into the remaining head arithmetic reproduces
  all 24,576 logits for each input at both shapes. The native hidden states
  differ in 4,306 of 4,608 FP32 values with maximum absolute difference
  `2.5630e-6`.
- The raw prefix embeddings and target latent match bit-for-bit before
  their input projections. The projected prefix differs in 342,861 of
  393,216 values, and the projected target differs in 3,975 of 4,608. In
  the CUDA batch-16 run, replacing either projected input on its own
  preserves all eight differences at row 4937. Replacing both at the
  first level-one Transformer block restores all 200 codes and 48 IDs
  for that row. The scores tie at the first crossing and the pinned
  selector picks 3182.
- On Mac Studio, the adapted Python MLX model at batch 32 selects 3690
  with the native projections, either CUDA projection, both CUDA
  projections, or the CUDA batch-32 hidden state supplied immediately
  before the head. All five batch outputs are byte-identical. The other 31
  rows match pinned Python. Thus the CUDA projection replay does not
  transfer by itself and the Metal head can differ on the same CUDA
  hidden input. An earlier [split-head diagnostic](level1-cross-backend-diagnostic-plan-v1.md)
  recovered the reference choice on a captured CUDA input but did not
  recover the full batch in free generation.

All inputs were checked before and after. Mac service snapshots show the
same PID. No runtime or comparator changed. Scripts, raw outputs,
safetensors and receipts are under ignored
`reports/phase4-xsum-pinned-python-batch-v1.DIrvoa/`. The batch-size
receipt SHA-256 is
`e4403ceda74669154e8c8474d6255c4c9476c24c9d30c672dadc2f3b92a5ac11`.
The CUDA head and input-projection replay receipts are
`e424e71ca7648ec94f6daec7556f2b21cb5b0d7a32fa0fdd2966961089d30544`
and `71a216bcd2b807b1224a1f2a730e4b4851d6cac98c1a632f34064608048d78f5`.
The Metal projection and hidden-state replay receipts are
`379ed4dc55e5821b08b6101877f96e81f9ee5ae45c1d09d5bb0fda1d3c90d3d1`
and `96f7a1f5f5059ca71bcb34a677004ab6eae90d453ae79526925e20bd1c046c56`.
