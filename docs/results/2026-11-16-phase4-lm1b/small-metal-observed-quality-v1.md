# LM1B-small Metal observed-seed quality retest

The experimental Metal CLI at commit `4fd568b` generated 256 samples for
each already-observed seed 42, 0, 1, 2 and 3, in batch-eight requests.
This is a retest, not fresh held-out evidence. It uses the same fixed
[protocol](heldout-protocol-v1.md), pinned artifact and rotary table,
SplitMix64 request seeds `(seed << 32) | batch`, host FP64 sampler and
penalties. The Mac's `dotsetup.rmlx` service remained running. Each
request ran in a separate CLI process; its output hash is recorded with
the capture. The first seed-42 request exactly matched a separate earlier
batch-eight CLI run. All five transferred capture hashes matched the Mac.

The pinned GPT-2-large scorer ran on Linux CUDA, checking each capture's
Metal binary, source, artifact, rotary, request seeds and hash. Its only
adaptation to the original scorer was Metal identity admission and file
locations; the scoring calculations stayed unchanged. Perplexity and all
individual entropy scores matched the pinned upstream scorer exactly for
each seed. A forged CUDA-penalty identity was rejected. Original Python
capture and score hashes were checked against the previously retained
[CUDA held-out summary](small-heldout-cuda-v1.md).

| Seed | Metal perplexity | Metal mean entropy |
| ---: | ---: | ---: |
| 42 | 90.475563 | 4.31217866 |
| 0 | 95.230335 | 4.31896905 |
| 1 | 92.868411 | 4.30977054 |
| 2 | 96.180485 | 4.31838078 |
| 3 | 93.282320 | 4.31022133 |

The unchanged independent seed-bootstrap comparison **fails** the entropy
interval. Metal minus Python mean entropy is `-0.0007996202`, within the
`±0.0046` point limit, but its 95% interval is
`[-0.0050203402, 0.0036893424]`, extending below `-0.0046`. Perplexity
difference is `-2.264292`, with interval `[-4.675775, 0.202392]`; its point
and interval checks pass. The failed original CUDA quality result and the
Metal controlled-sampling failures remain separate and unchanged. This
observed-seed retest does not qualify Metal or complete Phase 4.

The identity-bound summary repeated byte-for-byte with SHA-256
`1aedd2aed0de83c242fd6f9d346cb15610bf142ce28278a26adc23c5e4591b13`;
the unchanged metric comparator's three unit tests passed. The capture,
Metal scorer and summary script hashes are respectively
`054d3618bd28b3acda13a5761e3cfd6ca579b1bbfd014fe1014339f137893070`,
`a56b9b2310a3b44c03d6c0856ed565f7ec77d28d414fcb9586a99b03a6d90251` and
`fcce66e001a2e30a44e613d59ef98eddda6eebf8fefd4559f51a4462c2fb611e`.
All captures, per-request identities, scores, sources, logs and repeated
summary are retained under `reports/phase4-lm1b-metal-observed-quality-v1/`.
The CLI binary SHA-256 was
`fd09fa0ba4f5ede495fa7ab52b3c1d9ef83ce1990dbdeb82738fa1b1173072c2`.

## Same-request CUDA comparison and initial reconstruction

The five retained Metal captures were compared with the existing efficient
CUDA captures using identical request seeds, batch shape, artifact and
reference rotary table. The CUDA comparison used its already verified capture
hashes; Metal used the hashes in the summary above. Across 1,280 rows, all
initial codes match, but every final code row and every final ID row differs.
There are 754,079 differing codes out of 820,480 and 154,818 differing IDs
out of 163,840. These later differences are cascading continuations, not
independent same-history decisions. The first differing code in each row is
at level 1 for 337 rows, level 2 for 916, level 3 for 26 and level 4 for one.
This comparison does not establish the cause of the quality interval failure.

For seed 42's first batch, bounded diagnostic runs on both devices exactly
reproduce the respective retained codes and penalty-adjusted IDs. Initial
codes match. The efficient CUDA and Metal initial reconstructions differ by
at most `0.0465850830` (RMS `0.0042393702`); level-one logits differ by at
most `0.0101852417` (RMS `0.0021086611`). Three of the 64 level-one choices
differ. The first changed choice in request draw order is row 3, position 2:
the common uniform `0.5883136381` selects CUDA code 4737 and Metal code
4738. Replaying the two captured logit rows with that draw reproduces both
choices.

A separate strict-FP32 cuDNN CUDA control on that same request has the same
initial codes. Its initial reconstruction differs from Metal by at most
`0.0001449585` (RMS `0.0000045145`), and all 64 level-one choices match
Metal; 2,257 later codes still differ. Efficient and strict CUDA use
different convolution and generator backends, and the captured initial
reconstruction is produced before the transformer. This control localizes a
large part of the early cross-device gap to the initial codec path and is
consistent with the known convolution-precision difference; it does not
isolate a single convolution or prove that precision alone caused all later
choices. It does not change the evaluator settings or waive either gate.

The five-seed comparison result SHA-256 is
`0fd63d3428f456afe225c0132095ef8b59aa5b68dd4cef02d057d85dcbc2ef55`.
The first-level replay analysis and precision-control result hashes are
`48412144214c9ebbc8afeb81a9b89cc6947880937a3d715a5f1253099c4011da`
and `8ec719ebedd6ec9594aa7ebab260c774b8608b00db63c314ebcceaaedc1fd986`.
Sources, traces and results are retained under
`reports/phase4-lm1b-metal-cuda-observed-comparison-v1/` and
`reports/phase4-lm1b-firstlevel-cross-device-v1/`. The first strict-control
launch was interrupted after detecting that its temporary source still selected
the efficient backend; the corrected strict run produced the recorded result.
Temporary example sources were removed and the Mac checkout restored.

## Bounded Metal convolution precision experiment

The initial exploratory counts in this section use the accumulated tolerance
(`5e-4 + 2e-4 * abs(reference)`). The controlled-trace comparator classifies
`reconstruction.level_*` as a component and uses the tighter `1e-4` absolute
term. The accumulated-band counts below therefore do not establish a
controlled-trace pass. The corrected candidate checks are recorded in
[the component-policy follow-up](#accumulation-candidate-and-component-policy).

Using the same seed-42 batch and authenticated artifact, an adapted Python
MLX diagnostic reproduced the recorded Metal level-zero reconstruction
bit-for-bit. Rounding both residual-convolution operands to a TF32-like
10-bit mantissa with halfway values away from zero reduced error against the
efficient CUDA reconstruction from RMS `0.0042393702` to `0.0000965048`
and maximum `0.0465850830` to `0.0007629395`. However, 548 of 393,216
reconstruction values still exceed the existing accumulated tolerance.
Nearest-even rounding left 1,027 values outside, and truncation was worse.
These are formulation tests on one input, not a change to the oracle or the
accepted numerical policy.

With the same rounded reconstruction, a separate adapted Python Metal
level-one calculation reproduced all 64 CUDA sampled choices; its baseline
reproduced Metal's choices and differed from CUDA in three. The adapted
full-generation diagnostic still differed from CUDA in all eight rows:
3,491 of 5,128 codes differed when rounding all codec refinements, and 3,823
when also rounding generator post-convolutions. Its unmodified baseline
already differed from Rust Metal in 356 codes in one row, limiting any
conclusion about the Rust backend. The candidate does not pass the numerical
or complete-generation checks and was not added to production.

The precision, level-one and full-generation result SHA-256 values are
`6f9483d5b5dd218aa9b4ce8de8eadab8ae6ba4063af3254dc1c97f2bcb572f8c`,
`1c89cc097ce9b6645072e46818803bf368cacdb40c95243b62bbd5221c8dac84` and
`e31a0e00df40479f56b0312f6c5d2566eae68660d95c562c589195ce7a53eee7`.
Their source hashes are
`48e86f56ccdd58afb3dbb335bf62019afa781909e324962b01137c860652e232`,
`56747efdfe2f0e41d9e10022298c7f16b3cf430f2adb38cb66182aa73fc8ff13` and
`5ddc3562b7a48c5b5ac34f9694f2d0f50c25cad7e1d4324c8caa46d122cb2bf9`.
Sources, authenticated inputs and results are retained under
`reports/phase4-lm1b-metal-tf32-precision-v1/`; the Mac results match the
transferred files. The Mac checkout stayed clean and its service remained
running.

### Same-input and accumulation controls

For the same eight initial codes, public PyTorch CUDA with the evaluator's
convolution settings reproduced the retained efficient-CUDA initial
reconstruction bit-for-bit. Its captured upsample tensor is also bit-identical
to the Metal upsample tensor, including after the tested halfway-away TF32-like
rounding. Feeding that exact CUDA upsample into the adapted Metal residual
convolution leaves the same 548 out-of-band values. Thus the remaining error
in this fixture is not explained by different upsample values.

On the same rounded operands, a single windowed matrix multiplication left
497 values outside the accumulated tolerance; three separate matrix
multiplications with tested addition orders left 504–505. An independent CUDA
FP64 convolution of the same rounded operands, followed by FP32 blending,
left 470 outside. These controls do not establish the precise CUDA convolution
arithmetic or a passing Metal formulation. They also do not alter the failed
full-generation or quality results.

The public CUDA upsample capture SHA-256 is
`a984ad228c25139b82f100f616631918f041cda3b8ed3dd4237f6e08f25a5b89`.
The same-input, accumulation-order and FP64-control result hashes are
`f433bee7ad27c4d662e49e564234b8121a518c42723ab16eb10de6b6493868b1`,
`7dc730da7525d7acad3b24f60b8ba075292bf38fb3ac07c7e11c0f7a6d1ee0e6` and
`35b98b3aa27a5a396b7e089417595ad08093ff30aa4352ebbba6dcd0c475e14a`.
Sources, logs and captures are retained under
`reports/phase4-lm1b-metal-tf32-input-v1/` and
`reports/phase4-lm1b-metal-tf32-precision-v1/`. Transferred Mac inputs and
results matched their local hashes; the Mac checkout remained clean and its
service running.

### Operand-rounding and chunked accumulation checks

On the same identical upsample and weights, the public CUDA convolution
reproduces its original level-zero reconstruction bit-for-bit when either
or both operands are explicitly rounded to TF32-like precision with
halfway values away from zero. This establishes the effect of that rounding
for this fixture; it does not specify CUDA's internal reduction order.
Nearest-even and halfway-toward-zero variants do not reproduce the CUDA
output exactly.

A separate CPU FP64 diagnostic compared four input and four weight rounding
modes at 2,516 positions, including all 470 prior FP64-control failures.
No combination passed the unchanged tolerance at all sampled positions;
the closest RMS result was halfway-away rounding for both operands, which
retained all 470 original failures. This targeted sample is not a full-tensor
qualification. An initial attempt used an incorrect local input path and was
rerun after correction.

On Metal, explicitly accumulating rounded products in chunks of 32, 64,
128 and 384 left respectively 484, 476, 486 and 504 of 393,216 values
outside the existing tolerance. The single windowed matrix multiplication
left 497. No candidate passed or changed production behavior. These results
are consistent with a remaining difference in convolution accumulation or
related arithmetic, but do not establish its exact implementation.

The CUDA rounded-operand result, CPU sampled result and Metal chunked result
SHA-256 values are
`c9ff004a6580f6a4e502ce563325c96ea7420a2f47a3af3035465b9206f7517e`,
`a6fe4c9c5471267215f5cb242eab9d0e7153beda20fa57f175ed3e04c446ffed` and
`cdfdc06539c895408f9485f859009f11b75164e47534dafe545c4c9f89bf5acf`.
Sources and reports are retained under
`reports/phase4-lm1b-tf32-rounding-identification-v1/`; the transferred
Metal result matched its Mac original. The Mac checkout was clean and its
service remained running.

### Accumulation candidate and component policy

An exploratory candidate rounds both operands halfway away to TF32-like
precision, starts from the bias, adds eight-product FP32 partial sums and
clears one low mantissa bit after each addition. This is a tested arithmetic
formulation, not a claim about CUDA's internal implementation. It passed the
looser accumulated band on the original eight-row initial reconstruction
and two additional two-row evaluator fixtures. It does **not** pass the
controlled-trace component policy: respectively 443, 78 and 121 values
are outside `1e-4 + 2e-4 * abs(reference)`. The CPU full-tensor control
also has 432 component-band failures. Earlier in this investigation the
accumulated-band result was described as passing the initial reconstruction;
that was incorrect for the actual controlled-trace policy.

For the retained eight-row request, the adapted Python MLX candidate differs
from CUDA in 2,756 of 5,128 codes when applied to codec refinements, and
2,495 when also applied to generator convolutions. Every row differs after
continuations diverge. The unmodified adapted Python baseline already differs
from Rust Metal on one row, so these counts do not directly characterize a
Rust implementation. A same-request early trace uses the repository's actual
controlled comparator: six of 12 captured tensors fail, including 443, 428
and 406 out-of-band values in reconstructions 0, 1 and 2. This candidate
has not passed numerical, controlled-sampling or quality gates and was not
added to the runtime.

The independent seed-11 and seed-16 component-policy result SHA-256 values
are `8db3b6f52877f2b9c001d6987b57606fa20f827724e746bb2720bdf2f1574c9e`
and `a0d782dd9883fe97143eeb8dd60a44faae247e373636500bdc0c09b2eca7a923`.
The early controlled-policy comparison SHA-256 is
`550828738bb77e64f7496b566596d639f04e44d5f5577dcbbcc3f0b9f624183b`;
the full-generation refinement-only and combined candidate result hashes are
`02fceb6ef2ebf4c12874051908091eb83abc9dd8a6af2a00bf4951eea71fbdcf`
and `894622c9fe1dccb3249a1a2ee3eb2d3286c4fe1720a1e28f05cad4194a5c94b3`.
Sources, fixtures, traces and comparisons are retained under
`reports/phase4-lm1b-metal-accumulator-generalization-v1/` and
`reports/phase4-lm1b-tf32-rounding-identification-v1/`. The temporary CUDA
probe was removed; the Mac checkout remained clean and its service running.
