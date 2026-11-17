# XSum level-one cross-backend diagnostic

The unchanged full Metal raw-ID gate fails, and the retained same-batch
forced-history trace for row 4937 first crosses at level 1, position 4.
This diagnostic exposes the level-1 transformer output and head projection
for a forced history, alongside the previously captured logits. It records
one row from the same 32-row CUDA/Metal inputs and saved CUDA code histories.
The existing row trace remains unchanged and still filters out extra tensors.

First build and check on CUDA, then compare the new CUDA output's existing
logits/IDs with the retained same-batch trace. After the CUDA check and
reviewed commit, fast-forward the clean authorized Mac checkout and run the
same diagnostic there. Compare hidden values, projection and logits with the
existing floating-point bands and retain exact source/input/output hashes.
These observations do not waive the raw-ID gate or identify causation merely
from a near tie. No thresholds or reference artifacts are changed.

The first CUDA launch used a Mac-only input path and failed before inference;
the error is retained under `reports/phase4-xsum-level1-cuda-v1/`. The corrected
local capture uses `reports/phase4-xsum-mlx-full-top1-captured-v1/input-04928.json`
(SHA-256 `ea64637b3ca4176f2025af1f27611b46f71cda5a3219a027abb41a92cf39f515`)
and the retained CUDA history (SHA-256
`5fd01efcf2563a17910337bbd0debf28da20daf1167ecc2246c47961d8c132e8`).
All ten original captured tensors and selected IDs are bit-identical to the
retained CUDA row trace. The new tensors have shapes `[1, 6, 768]` and
`[1, 6, 384]`; the capture SHA-256 is
`a1db24ca2c4cabd9f9f7c87c81f36f5ae697a3e1ffde465cbb3daf4ea2628894`.
This is a diagnostic CUDA reproduction, not a Metal result.

`setpriv --inh-caps=-all env RUSTC_WRAPPER= ./tools/check.sh --cuda` passed,
including 18 real-device operator cases (log SHA-256
`4aa2d4e1ea86a256939e825659892536b9b0ab2d60469250de087cef916243de`).
The feature-specific CUDA diagnostic also passed Clippy with `-D warnings`.

## Metal result

The clean Mac checkout fast-forwarded to `e01d455`, then `./tools/check.sh
--mlx` passed, including 18 Metal operator cases. The diagnostic was built in
release mode and run with the same 32 prefixes and forced histories; input
SHA-256 values above matched before and after. Its Metal safetensors SHA-256 is
`7cc97d8548e76788a8bd24cd152b4842edaa94f7e1a4ff2beb6303b642b42692`;
JSON SHA-256 is
`3f269850c0709ee4165f58c23dd8fbd19701957b915bc005dd879b723c952a40`.
The copied files match remote hashes. All ten retained Metal trace tensors
are bit-identical to the earlier Metal trace, and selected IDs match.

The new hidden tensor already differs between CUDA and Metal in 4,515 of
4,608 elements (maximum absolute difference `8.2850e-6`; at the divergent
position 4, `5.0068e-6`). The subsequent projection differs in 2,242 of
2,304 elements (maximum `1.4782e-5`; position 4, `9.5367e-6`). Thus the
backends differ before the head projection; these observations do not locate
a particular transformer operation or prove the later near tie's cause.
The existing logits retain their documented in-band crossing, and the exact
full Metal raw-ID gate remains failed. The Metal check log SHA-256 is
`26d588bdc1198731da3cba950b29e48a7a94c88c6de0d0870d9ad2e62a244834`.
Raw evidence is under ignored `reports/phase4-xsum-level1-{cuda-v2,mlx-captured-v1}/`.

## Transformer block follow-up

The next forced-history diagnostic adds the concatenated level-1 input and
all 12 transformer block outputs, preserving all earlier observations. It is
only enabled for the forced level-1 trace. A CUDA capture with the same inputs
contains 25 tensors, and all 12 previously captured tensors match bit-for-bit.
Its safetensors SHA-256 is
`c9d7c3d95e0177e89db35ad0f1c109b90451323acd058e056b42a95ab05f419f`.
Feature-specific strict Clippy passed. The full CUDA regression passed, including 18 operator cases (log SHA-256
`f3ea33da28fe8d3c3ba1494cab25469e8c2f89dd9e07710e3e992be420699208`). The matching
Metal trace remains pending; the added observations are diagnostic only.

The follow-up Metal check at `5025e97` passed, including 18 operator cases;
its log SHA-256 is
`1ae00c77d98dce231f58dc1693dfd5218f9b68be4a8398ee2c5f761452614f86`.
The capture SHA-256 is
`cb113f515bc6512ca0b8c511ee0bf7f4c7a780f8c5a1a5fc2e8788d99a7c4edd`;
the local copy matches the Mac hash, input hashes were unchanged, and all 12
previous Metal tensors are bit-identical. The selected IDs remain unchanged.

At transformer entry, the 512-position prefix is bit-identical across CUDA and
Metal, but the six target vectors already differ, with maximum absolute
`7.1526e-7`. After block 0, maximum differences are `1.5259e-5` in the prefix
and `1.4305e-6` in the target. After block 11 they are `1.8311e-4` and
`4.5776e-5`, respectively. These are raw diagnostic differences, not a new
tolerance or an assertion about the first causal operation. Input reconstruction
and block arithmetic both need investigation; the exact raw-ID gate remains
failed. Raw traces are retained under ignored
`reports/phase4-xsum-block-{cuda-v1,mlx-captured-v1}/`.

## First-block follow-up

A further forced-history trace observes first-block normalization, QKV
projection, merged attention, and attention residual. The CUDA capture retains
all 25 prior tensors bit-for-bit and adds four observations; SHA-256
`74986c330e7a2076f835475f5f4e52d4021588633cb358bbee2721cecdc51b77`.
Strict feature-specific Clippy passed after correcting a style failure, whose
initial log remains retained. The full CUDA check and matching Metal capture
are pending. This traces arithmetic but does not alter selection or tolerances.
The first full CUDA check stopped in the unrelated
`test_default_budget_handoff_preserves_deadline_and_all_eighteen_real_observations`
process-observer fixture after an unexpected live executable identity; it did
not reach CUDA checks. The failure is retained in
`reports/phase4-xsum-first-block-cuda-check-v1.log`; a full rerun is in progress.
The full rerun passed, including 18 CUDA cases; its log SHA-256 is
`8f6b5fcf4f3920f6e6f0aaf8d23b52575bc61fd9955d75b37f542191676eb13e`.

The matching Metal check at `14e900c` passed, including 18 operator cases;
its log SHA-256 is
`214c3ce3830d795b6b81b123d3016049d668239aa73e2e34500edb2c750a283b`.
The Metal trace SHA-256 is
`f84c98e40865eaec2a8cc5270a9fcbc46119c5637e76073d538720fd5c84a133`;
its local copy matches the Mac hash, inputs were unchanged, and all 25 prior
Metal tensors match bit-for-bit. Selected IDs remain unchanged.

For the prefix, transformer input was exactly equal; first layer normalization
already differs in 158,444 elements, maximum `1.1921e-7`. The maximum prefix
difference then is `2.3842e-7` after QKV projection, `4.6194e-7` after
attention, `1.3113e-6` after the attention residual, and `1.5259e-5` after
the entire first block. Target inputs already differed before this block.
This suggests a bounded normalization comparison as a next experiment; it
does not prove the downstream code near tie is caused by this operation.
The exact Metal raw-ID gate is still failed. Full raw traces remain under
ignored `reports/phase4-xsum-first-block-{cuda-v1,mlx-captured-v1}/`.

## Target-input follow-up

The forced trace now additionally records the initial reconstruction, its
level-1 resized latent and input projection. CUDA's row capture contains 32
tensors; all 29 previous tensors are bit-identical. Its SHA-256 is
`f1192dd8e9a1b6ea68a63930109b66146c83a2b044f59f0354e5a08282bb0d2b`.
Feature-specific strict Clippy passed. The full CUDA check and matching Metal
capture remain pending. This is diagnostic and does not change selection or
qualification rules.
The full CUDA check passed, including 18 operator cases; log SHA-256
`870f5aaf57d082ac64fee66c467e798cdc2968b588ba85317dcf51d1d41c30c6`.

The corresponding Metal capture at `09d6c04` passed the full Metal check,
including 18 operator cases; log SHA-256
`338328af45b060beefdb7073d2d700296f023d619a266396fb2a694cc621dc97`.
The Metal trace SHA-256 is
`4eae7ba83c8a945e4c9c7beb850e963db0a6a4367a57dab3c00df14448cbf1d5`;
the local copy matches the remote hash, inputs were unchanged, and all 29
previous Metal tensors match bit-for-bit. The CUDA/Metal first reconstruction
already differs in 17,521 of 18,432 elements (maximum absolute `1.9073e-6`,
RMS `2.3264e-7`). The resized latent differs in 2,206 of 2,304 elements
(maximum `3.8743e-7`), and the target projection in 4,401 of 4,608 (maximum
`7.1526e-7`). These observations locate a difference before the resize but do
not identify the first differing codec operation or prove causation for the
head near tie. The exact Metal gate remains failed. Raw evidence is retained
under ignored `reports/phase4-xsum-target-{cuda-v1,mlx-captured-v1}/`.

## Level-zero cascade follow-up

A forced-history diagnostic now records the codebook output, both upsampling
convolutions, and the intervening group normalization for level zero. The
same-batch CUDA capture contains 36 tensors and reproduces all 32 previous
CUDA tensors bit-for-bit; SHA-256
`155d66bbed2905dd3d99cd2a255bb55601dcbb181b06a057ce0e70710726f867`.
Feature-specific strict Clippy passed. Full CUDA check and the corresponding
Metal trace remain pending. The default untraced cascade is unchanged.
The full CUDA check passed, including 18 operator cases; log SHA-256
`effe8c19e8d9a4e792a3f04ef0f71f3e97c7764fdc8970256bf72c303f4946e6`.

The matching Metal trace at `4749514` passed the full Metal check, including
18 operator cases (log SHA-256
`97292be7872830927e4c744b2d5feb3056933e38c56992419f027dcb87c2067b`).
All 32 previous Metal tensors are bit-identical; input hashes were unchanged.
The trace SHA-256 is
`c56fef74e46b6a301c5eb794e66cdf79012dafc635af277589cfdab0592c6242`
and the local copy matches the Mac. The codebook output is exactly equal to
CUDA, but the first upsampling convolution already differs in 1,309 of 4,608
elements (maximum absolute `2.3842e-6`). The following group normalization
has maximum difference `3.5763e-6`, and the second convolution `2.4885e-6`.
These observations locate the first captured difference but do not prove its
causal responsibility for the later near tie. The full Metal raw-ID gate
remains failed. Raw traces remain under ignored
`reports/phase4-xsum-cascade-row-{cuda-v1,mlx-captured-v1}/`.

After the later three first-block MLP traces were added, the corrected
39-tensor CUDA forced-history capture reproduced every tensor shared with the
retained CUDA cascade capture bit-for-bit. Its SHA-256 is
`3723d9fda929e4d503614961f2f2126a18b6cefa563accb51c9cb3b8eb54ce0f`;
the capture and build log are retained under `reports/phase4-xsum-mlp-cuda-v1/`
and `reports/phase4-xsum-mlp-cuda-build-v1.log`. Compared with the experimental
Metal group-normalization model trace, maximum absolute differences across the
selected row were `1.3113e-6` at the first residual, `2.0266e-6` at the
second norm, `8.5831e-6` at the MLP input projection, `5.7220e-6` after GELU,
and `1.5259e-5` after block 0. The largest block-level difference is in prefix
positions. These stages receive already-different inputs, so this comparison
does not assign causality to an individual operation or establish a fix.

The first MLP projection was isolated on the captured CUDA second-norm input.
A row-only CUDA projection differed from the retained 32-row trace in over
1.4 million values; zero-filling the other 31 rows while retaining row 4937
at slot 9 restored all 1,591,296 CUDA values exactly. Using that same input,
weights, bias and 32-row geometry, the standalone Metal projection at
`415cdd3` also matched all 1,591,296 retained CUDA values bit-for-bit. The
hash-bound input and receipt are retained locally and on Mac, with SHA-256
`bb6efb7f11a4cfe569bd7d3faaa341b5b42f4f2bc772ba499beca0f2dfa00aa2` and
`25a99094984c669db3ada0c287530f381ed731b47a2192063506c00e33871d36`.
This rules out a difference for this isolated projection and input, not for
all shapes or values; earlier normalization differences and the full Metal
gate remain open.

The second normalization was isolated using the 512 captured CUDA prefix
vectors, its trained affine weights and the retained CUDA outputs. PyTorch
CUDA layer normalization reproduced all 393,216 reference values exactly.
The existing standalone Metal probe matched 245,194 values with MLX fast
normalization (maximum absolute difference `7.1526e-7`) and 299,356 with
safe-math Welford (maximum `4.7684e-7`). The input and receipt SHA-256 values
are `9c99649d0b8e9c0d65e33a2c379fecbfad3e15b3f60a6c96f20ab3dd91b883ec`
and `67f24926e41b166672502cb4df20f3a77ec6557da76de2afc03f18236680cef5`;
local and Mac copies match. These are previously inspected prefix vectors,
not a causal demonstration or full-quality test; earlier model-level
normalization experiments did not resolve the known output crossing.

The captured CUDA first-MLP input and GELU output were packed as a same-input
probe (SHA-256 `ac8b934c98ca34d9590afdededad8dfba2bb5b4e3c555d382190819f0f9c95d0`).
PyTorch CUDA reproduced all 1,591,296 retained GELU outputs exactly. At pushed
commit `a1a70f0`, standalone Metal composed GELU matched 1,034,783 values;
maximum absolute difference was `7.1526e-7`, mean `6.2985e-8`. A separately
compiled Python MLX graph matched 324,499 values, maximum `5.9605e-7`, mean
`7.0696e-8`. The result SHA-256 is
`164b7bd0703b147a8b12819460eee8dc1e00efe6a8395d552d4f18b4cbce8731`;
local and Mac copies match. These comparisons isolate the operation on one
captured row. The Python compiled graph uses a row-sized input, not the full
32-row Rust graph, so its statistics do not prove exact Rust backend behavior
or causality for the code crossing. Full output and quality gates remain open.

The standalone Metal GELU probe next tested the CUDA expression's multiply
order on this identical input. The first attempt failed because `erf` was not
declared; the second failed because the JIT could not resolve the installed
header. At `8c3ed81`, the probe loaded the installed MLX `erf.h` and
`expm1f.h` into the JIT header with upstream notices preserved and recorded
their hashes. Safe, relaxed and fast math each matched 1,180,744 of 1,591,296
CUDA values, maximum absolute difference `5.9605e-7` and mean `5.9422e-8`.
The original composed operation matched 1,034,783. The receipt SHA-256 is
`765d1ff6daf714ad0485356fd2aff5ac70210f81463c97737f4f50fe94d9e92d`;
Mac and local copies match. This is one inspected row and a Python Metal
kernel, not a validated Rust model change, performance comparison or gate pass.

An explicitly selected model diagnostic at `ae1b8bf` combined the prior
experimental upsampling and group-normalization operations with an uncompiled
MLX GELU graph using multiplication by `1/sqrt(2)`. It is not the standalone
custom kernel, and the default backend remains unchanged. Full Linux CUDA
validation passed, including 18 operator cases. The Mac release build and
row-4937 forced-history capture completed; the candidate still selects code
3690 where CUDA selects 3182 at level 1 position 4. Its 3182-minus-3690
margin is `-3.8147e-6`, versus `-4.7684e-6` for the earlier Metal candidate;
forced final IDs match CUDA. The captured tensor SHA-256 is
`0d9364a314d78c6609e41261873c3d143f6d6c68b13ee811ab162d1594940dc7`,
with matching local and Mac copies. This does not resolve the crossing or
qualify a full generation, quality or performance gate.

A fresh synthetic standard-normal `[1, 518, 3072]` input generated with Torch
CPU seed `735027` was scored by pinned PyTorch CUDA GELU and compared using
the existing Metal probe. The input SHA-256 is
`7b1dd78ef1062804010e816fa6564bda7240706526ad933950a2100a1f414bfb`;
the result SHA-256 is
`a13e47c6a9a552db13097d6371549ddfee32b888e159e6a2ad8de6ce7939f170`.
Composed Metal matched 1,080,669/1,591,296 values exactly; the standalone
CUDA-order Metal expression matched 1,203,197, with maximum absolute
differences `7.1526e-7` and `5.9605e-7`, respectively. All three Metal math
modes had the same reported results. This one synthetic input is not a
model-quality, performance or full-parity result.

At `80bbc5c`, the standalone probe compared compiled and uncompiled MLX GELU
graphs using multiplication by `1/sqrt(2)`. On the captured row, compiled
multiplication matched 810,907/1,591,296 CUDA values versus 324,499 for the
original compiled division graph and 1,180,744 for uncompiled multiplication.
On the synthetic input, corresponding counts were 892,589, 368,231 and
1,203,197. The receipts hash to
`c4deead62a37f129aea3bc31ef8d24dcce67b9b74cb144699773a9e6cb75c4f0` and
`299049f1474e86c8446cc0a6c60072c6da873ce177fcd3a3d156f28fef758fa6`,
and local and Mac copies agree. A separate explicitly selected model backend
is prepared to test the compiled multiplication graph with the prior cascade
diagnostics. The full CUDA check passed, including 18 device cases
(`reports/phase4-xsum-compiled-gelu-cuda-check-v1.log`); Metal model results
remain pending. This is not a quality or performance gate.

The compiled multiplication model diagnostic built and ran on Metal at
`fdd7a24`. Its row-4937 forced-history tensor capture hashes to
`0d9364a314d78c6609e41261873c3d143f6d6c68b13ee811ab162d1594940dc7`,
identical to the prior uncompiled candidate; the JSON hash also matches. It
therefore retains the same level-1 code crossing and matching forced final IDs.
This result applies to the captured row, not all inputs or a performance gate.

The explicitly selected all-operation diagnostic at `bcf03b7` combines the
experimental upsampling, group normalization, Welford layer normalization
and compiled GELU. Full CUDA checks passed, including 18 device cases; the
Metal release build and row-4937 forced-history capture completed. The
candidate still selects 3690 instead of CUDA's 3182 at level 1 position 4,
with margin `-5.7220e-6`; forced final IDs match. The tensor SHA-256 is
`6bf249c311a636410eb9aafba54e4ee9e4c1ef822f5a4da09eab99e8c0554630`,
and local and Mac copies match. This fails to resolve the known crossing and
does not qualify full generation, quality or performance.

At `9847c47`, the bounded free-generation example compared the default Metal
backend with the upsample, groupnorm, GELU, compiled-GELU and all-operation
diagnostic backends on the same 32-row batch beginning at row 4928. Every
capture was byte-identical (SHA-256
`1662e6869c012e017df154da76d72d83f634123dd86b874010e73e6335738050`),
and the default capture also reproduces the retained default Metal batch.
Compared with the retained same-batch CUDA capture, each has three code and
five raw-ID differences, all on row 4937. Input hashes passed before and after
the runs; local and Mac captures match. This explicitly tests free generation
on one previously inspected batch, not the full corpus or a quality gate. It
rules out these candidates as exact fixes for this batch.

For row 4937's level-1 position 4, a CPU FP64 diagnostic recomputed the
unrounded squared-distance score difference between codes 3182 and 3690 from
each captured projection and the pinned FP32 codebook values. After applying
the positive CUDA-derived mean-norm and temperature scales, its margin was
`-3.4309e-6` on the CUDA projection and `-3.0220e-6` on the experimental
Metal projection. The recorded FP32 margins were respectively `+9.5367e-7`
and `-3.8147e-6`. This calculation deliberately uses FP64 codebook norms and
does not reproduce the reference's FP32 reduction/rounding semantics. It
shows that replacing the reference decision with this higher-precision
recomputation would select 3690 even for the captured CUDA state; it is not a
valid parity fix or evidence to change the frozen decision rule.

At `faad73b`, a controlled same-input head comparison used the captured CUDA
projection at row slot 9 of the original 32-row geometry, CUDA-derived
codebook norms and FP32 scales, with other rows zero-filled. The controlled
PyTorch CUDA calculation reproduced the reference pair margin `+9.5367e-7`
but differed from the retained reference in 5,692/24,576 logits; it is not an
exact replay of the reference computation. On the same controlled input,
Metal's margin was `-2.8610e-6`: CUDA chose 3182 and Metal 3690. Metal matched
6,528/24,576 controlled CUDA logits exactly, maximum absolute difference
`2.1935e-5`. This isolates a crossing in the controlled head arithmetic even
with equal projections and norms, without proving it is the sole cause of the
model-level crossing. Input and receipt SHA-256 values are
`bb6d1badf44ce7b9e82c37585291d39ce314388f989cb4f92366a1ce77e63aa9` and
`1894c758a706ef8076c3ce5469f0aebf6741ee911c71f748a876dcf72178d73d`;
local and Mac receipt copies match. No gate or runtime change follows.

At `ca79e11`, the controlled screen also captured the CUDA head dot products
and applied the remaining scaling and subtraction on Metal. Metal reproduced
all 24,576 controlled CUDA logits exactly when given those CUDA dot products,
including the `+9.5367e-7` pair margin. Independently computed Metal dot
products matched 1,937/24,576 CUDA values, maximum absolute difference
`5.3406e-5`; for codes 3182 and 3690, CUDA produced
23.597530365/23.695962906 and Metal 23.597528458/23.695970535. The
hash-bound input and receipt SHA-256 values are
`d4a435ff62e472de4d575cfa8ee5da38054caa347116ab0fe276dc45f8b321da` and
`fd214d65ec37ddc229acd3251b496f45f77154fe4eac9096cbb495dcbf2e3115`;
local and Mac receipt copies match. These results localize this controlled
crossing to the dot product, but the controlled CUDA calculation is not an
exact replay of every retained reference logit and this is not a gate pass.

The controlled head probe tested equivalent Metal dot-product layouts. At
`98c621a`, a sum of two 192-wide matrix products matched all 24,576 controlled
CUDA dot products and selected code 3182 on the inspected row; flattened,
row-only and four-part alternatives retained the wrong choice. The receipt
SHA-256 is `a378f6940da332599aad365b98b96ae498f141c6a4b1ba7c292e3a51283254b2`.
The same two-part layout also matched all 24,576 controlled CUDA dots in each
of four fresh synthetic input screens (seed `841627`, captured codebook and
32-row geometry), retained under `reports/phase4-xsum-head-layout-synthetic-v1/`
on both machines. An explicitly selected Metal model diagnostic combines
two-part XSum head projection with the earlier upsampling candidate; the
default backend is unchanged. Full CUDA checks passed (18 device cases;
`reports/phase4-xsum-head-split-cuda-check-v1.log`). Rust Metal compilation,
free-generation, full quality and performance remain pending.

At `2abb48a`, the split-head diagnostic built and ran on Metal. On the forced
row-4937 history it still selected 3690 instead of CUDA's 3182, with a
`-9.5367e-7` margin. Free generation matched all CUDA codes and raw IDs for
the 4864 batch, but retained three code and five raw-ID differences on the
4928 batch. Its free outputs are byte-identical to the earlier upsample
candidate on both batches; hashes are
`8a7ef47db698d66906f238d61085b43ff2e49d7d29154754e8e05eeb7026beb6` and
`1662e6869c012e017df154da76d72d83f634123dd86b874010e73e6335738050`.
The forced tensor SHA-256 is
`0ff9cd6bdc43e1bef1679273617f0f5fd714607a6f2acda197a756789f32468d`;
local and Mac copies match. This is a previously inspected bounded screen,
not full corpus quality or performance evidence. The exact gate remains failed.

At `7509110`, three additional explicit combinations paired the split head
with experimental group normalization, compiled GELU, and all prior
operations. Full CUDA validation passed including 18 device cases. On Metal,
all three free-generation outputs were byte-identical to the earlier split
candidate on both 32-row batches: the 4864 batch matches CUDA exactly, while
the 4928 batch retains three code and five raw-ID differences. Input hashes
passed postflight and local and Mac capture hashes agree. This eliminates these
combinations for the inspected batches, not all possible approaches; full
quality, performance and the exact Metal gate remain open.

The same six backends were screened on the 32-row batch beginning at 4864.
Default Metal matched all CUDA codes but retained one raw-ID difference at
row 4883 position 40. Each of the five experimental variants matched all
CUDA codes and raw IDs on that batch; their byte-identical capture SHA-256 is
`8a7ef47db698d66906f238d61085b43ff2e49d7d29154754e8e05eeb7026beb6`,
versus default Metal SHA-256
`13126af2482cd1a8bb7bd83bd382557b01b10d8fa1d718654ccc339629d4cce5`.
Input hashes passed before and after; local and Mac copies match. This confirms
the partial row-4883 improvement in free generation, while all variants still
fail the previously described row-4937 batch. Neither screen is full quality.
