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

The level-1 hidden-to-head projection was tested separately on the captured
CUDA hidden state. PyTorch CUDA reproduced all 2,304 retained projection values
with the original 32-row geometry but not with batch 1. On the identical input
and geometry, Metal's ordinary projection matched 271/2,304 CUDA values,
maximum absolute difference `2.0981e-5`. Splitting the 768-wide product into
two 384-wide products improved the count to 439/2,304 and reduced the maximum
to `5.7220e-6`, but did not reproduce the reference. The input and split-probe
receipt SHA-256 values are
`f3facfd746c8be6b5d05d3370efcbce0c00eb1de9104a64fc26fb0a4f6c879d7` and
`2de3a86dadcc4ab09fdea13e5771d9e3c96dd3c69bce90c51f603effae871273`;
local and Mac receipt copies match. These bounded comparisons do not establish
a model-level fix or justify changing the default projection.

At `24279d2`, the same controlled head projection was also tested as sums of
three, four and six equal-width products. They matched 350, 420 and 502 of
2,304 CUDA values exactly, with maximum absolute differences `7.6294e-6`,
`5.7220e-6` and `3.8147e-6`. The prior two-part result matched 439 values.
The hash-bound receipt is
`abfc9115e2a29370051eb564cc2d4840297b0f311dee79e52d1fdf546f120303`;
local and Mac copies agree. None establishes reference parity or a model fix.

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

At `88a9ce9`, the controlled probe also computed codebook squared norms on
Metal. They matched 2,725/4,096 PyTorch CUDA norms exactly (maximum absolute
difference `3.0518e-5`); applying the Metal norms and their mean scale to the
controlled CUDA dot products still selected 3182 with margin `+9.5367e-7`.
The receipt SHA-256 is
`0fa7a3f859fd59cbd7d7265f0e248112f4c80aa5cf690dcaaa5742ca586d3130`,
with matching local and Mac copies. These CUDA norms are PyTorch-derived and
were not captured from the actual Rust CUDA backend, so this is controlled
diagnostic evidence rather than exact runtime attribution.

The same controlled CUDA head arithmetic was then run on the captured Metal
split-head model projection. With the CUDA projection it chose 3182 at margin
`+9.5367e-7`; with the Metal projection it chose 3690 at margin `-9.5367e-7`.
The CUDA dot products for codes 3182/3690 were 23.597530365/23.695962906
for the CUDA projection and 23.597530365/23.695968628 for the Metal projection.
Thus upstream projection differences can independently change this decision,
even with CUDA head arithmetic. This is one controlled row, not a quality gate.

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

At `0584442`, an explicit diagnostic combined the two-part hidden-to-head
projection with the previous two-part codebook head and upsampling candidate.
Full CUDA validation passed including 18 device cases; the Metal build and
both free-generation screens completed. Outputs were byte-identical to the
earlier split-head candidate on both batches: 4864 matches CUDA exactly,
while 4928 retains three code and five raw-ID differences. Input hashes passed
postflight and local and Mac output hashes agree. The reduced isolated
projection error therefore did not resolve the observed free-generation
failure; this is not full corpus quality or performance evidence.

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

At `9c3db94`, a controlled chain used the captured CUDA hidden row, the
original 32-row geometry, CUDA-derived norms and scales, and a split codebook
dot product. Splitting the preceding hidden projection into six products
reproduced the controlled CUDA pair margin `+9.5367e-7`; two, three and four
parts produced ties, and the unsplit projection chose 3690. On four synthetic
hidden inputs (seed `427193`), six parts had the most exact score matches of
the screened layouts, but still differed on most of the 24,576 logits. The
captured-row receipt SHA-256 is
`327409f8ea274dab2e3949dc5ec23406cc2b26c8d635f3099e83e5f04f7585a2`;
synthetic inputs and receipts are retained under
`reports/phase4-xsum-head-chain-synthetic-v1/`. This is controlled diagnostic
evidence, not an exact reference replay or quality gate. A six-part model
variant passes the full CUDA check, including 18 device cases; Metal
free-generation and performance results remain pending.

At `d4b37b6`, the same controlled chain was screened using the retained Metal
split-head hidden state instead of the CUDA hidden state. All five hidden
projection layouts selected 3690; their pair margins ranged from
`-9.5367e-7` to `-3.8147e-6`. CUDA-derived norms and scales were held fixed.
The CUDA score array in the input belongs to the *different CUDA hidden state*,
so the receipt's exact-match counts are not same-input accuracy measurements.
The input and receipt hashes are
`7cee53e126b21ba646b132b2104d707923de7ad5a61794c0c31770cf42e82e18` and
`b5c6a30a82cb4ba9a84046b4763d9f9727b508c91455dbfa1e62d86bfad1792b`;
local and Mac receipt copies agree. This is one inspected forced-history row,
not a full model or gate result.

At `6b8a156`, the six-part model variant built and ran on Metal. Its free
outputs on batches 4864 and 4928 were byte-identical to the earlier split-head
candidate: batch 4864 matches CUDA, while batch 4928 retains three code and
five raw-ID differences. Input hashes passed postflight; local and Mac output
hashes agree. The controlled head-chain improvement did not resolve this
free-generation failure. Full corpus quality and performance remain pending.


## CUDA intervention replay of the row-4937 crossing

A local diagnostic replay uses the pinned upstream `Block`, captured row in
its original slot 9 of a zero-filled batch of 32, the original 518-token
geometry, captured rotary tables and the actual prefix padding mask. The
initial PyTorch math-SDPA replay exactly matched the first normalization and
QKV, but its attention differed in 324,028 values (maximum `2.5239e-7`),
so it was not an exact baseline. Replacing SDPA with the repository CUDA
masked-softmax kernel and the same contiguous matrix products made block 0
match all 397,824 captured outputs exactly, including the recorded internal
stages. The extracted kernel uses the repository build's `compute_75` and
`-O3` options; these are bounded diagnostics, not changes to inference.

Using that replay through the remaining blocks, captured CUDA states at the
input and after blocks 0, 5, 10 and 11 each reproduced the captured CUDA final
hidden exactly and the reference head pair margin `+9.5367e-7` (choice 3182).
For each corresponding Metal state, replay chose 3690, with respective margins
`-9.5367e-7`, `-1.9073e-6`, `-1.9073e-6`, `-2.8610e-6`, and `-2.8610e-6`.
The head projection and codebook dot products must use flattened batch geometry;
a preliminary three-dimensional projection yielded an incorrect CUDA baseline
and is not used for attribution. These experiments do not prove a unique cause.

The captured prefix input is identical between backends. Substituting only the
Metal prefix into CUDA input preserves the CUDA result; substituting only the
six Metal target vectors changes 4,232 input elements (maximum `9.5367e-7`)
and flips the replayed choice to 3690. Separately, the CUDA input projection
of the captured CUDA latent exactly reproduces the captured projection and
input, whereas projecting the Metal latent with CUDA and then running CUDA
transformer/head also chooses 3690 (margin `-1.9073e-6`). The captured
level-zero codebook embeddings and first cascade convolution match exactly,
while the subsequent cascade normalization and reconstruction differ. These
findings establish that upstream target/latent differences can independently
cause this one crossing; they do not establish which operation to change,
whether a candidate fixes free generation, or full-corpus parity.

The ignored local artifacts are under `reports/phase4-xsum-block-replay-v1/`.
The exact-block script, kernel source and log SHA-256 values are respectively
`c6f2d1ff0257c617444135dc1ce4b7b35548c50abfeed4e9d057ed83bf6ac2e8`,
`c760c3efeb761498705a22a129ff990fa4bf2afe8e5aaf37af203f4bc3844b83` and
`8938b5298ec743fcb8f09a3561ec3a02254b8a4eb63890bd6bdc673faea7a135`.
Boundary intervention script/log hashes are
`9cf85cdceee1b5a90e9093d151c50cf7c0db123b63d6a2c5564f592c8cd4b9d4` /
`aff41abb1160024184f6e29a3efe015d8c8d1a1facf73e306ab0b909856405ff`;
input split script/log hashes are
`5fa4967c22e793cb1a584b7fe83a55af64dbab64aab529c244142cb8ef51fdf0` /
`a845d575b2c672cd44af2e47d743f00b7b4df57e79f0eb50e7aad5d5edc3d607`;
latent intervention script/log hashes are
`8b0e32ff0b20a7160946ef4c75f645a87c01ec5baf74ba3680b6fd69a7dc31a5` /
`c4c47be44dd166c658dbf18ade95f24ca2d0beec443b6cf770001a74b160e8bd`.
Run the Python probes with `setpriv --inh-caps=-all env OMP_NUM_THREADS=1
.reference/venv/bin/python` followed by the script path; compile the extracted
kernel with `/opt/cuda/bin/nvcc --shared -Xcompiler -fPIC -O3
--gpu-architecture=compute_75 softmax.cu -o softmax.so`. The full quality
and performance gates remain open.


A further intervention reconstructs the remainder of the CUDA cascade from
its captured normalization output using the original batch geometry. CUDA
transposed convolution and refinement reproduce all captured reconstruction
values exactly; the 48-to-6 area reduction requires the CUDA operator's
sequential eight-value FP32 sum, which reproduces all captured latent values.
Ordinary PyTorch reduction differs and is not used for the intervention.
With either the Metal cascade normalization output or the Metal second cascade
convolution output substituted, the remaining CUDA computation still chooses
3182 with margin `+9.5367e-7`. With the Metal initial reconstruction
substituted, CUDA area reduction and the remaining model choose 3690 with
margin `-1.9073e-6`. All three corresponding CUDA-state controls reproduce
the captured CUDA target input and final hidden exactly. Thus, for this
particular state and CUDA continuation, the reconstruction difference is
sufficient and the earlier captured differences alone are not; this narrows
investigation to refinement and interactions, not a unique arithmetic cause.

The additional ignored script/log SHA-256 pairs are: normalization
`efb784add347422e90e9d6572b349f724d657c3a729126901e68a921415c4cbe` /
`1659068d81efd6388d4cccd20c105832660187e4d0db1e2eac4d78e4667d3ca8`;
second cascade convolution
`4093370aeb9d832b0563709eb8597d12104c40f682c95b01433c27eac5aa6c52` /
`74d74cc1e5eb53a97406cc0569f987bb208520cf26f63a1c46e8cea0009196e9`;
and reconstruction
`7c69d171ccc4b7981f058e7852f33755cc5a86f9b3b82063c8094bc9a2bd8416` /
`e6e575ccf935bd58f00521512e51d4b06ca94aab1e025d9ae7a459b74fbdea0d`.


The [standalone Metal refinement probe](../../../tools/reference/xsum_refinement_metal_probe.py)
then used the captured Metal cascade convolution as the *same input* to CUDA
and Metal, each with original batch 32 and the row in slot 9. CUDA used the
control described above. Metal independently reproduced every value of the
captured Metal reconstruction. Its refinement convolution differs from CUDA
in 16,653/18,432 values, maximum `3.8147e-6`; the resulting reconstruction
differs in 16,378 values, maximum `1.9073e-6`. The input SHA-256 is
`e6a550e694cea63fce4c393290f89a84ea45a009076dafad5ab18b444aae0f57`
and receipt SHA-256 is
`f2db0d8648866fb93732be3b4bacd66835393373237b937d058ff5a75e7b45e8`;
local and Mac receipt copies agree. Mac command: `.reference/mlx-venv/bin/python
tools/reference/xsum_refinement_metal_probe.py --input
reports/phase4-xsum-refinement-comparison-v1/input.safetensors --input-sha256
e6a550e694cea63fce4c393290f89a84ea45a009076dafad5ab18b444aae0f57
--output reports/phase4-xsum-refinement-comparison-v1/result.json`. The
receipt checks the input hash before and after. This locates same-input
refinement arithmetic differences for one captured row; no default, tolerance,
quality or performance gate has changed.


The [refinement layout screen](../../../tools/reference/xsum_refinement_layout_probe.py)
compared the same captured input and batch geometry. A single flattened
`[batch*48, 1152]` matrix product in channel/kernel order exactly matched all
18,432 CUDA convolution and reconstruction values. Three separate kernel-tap
products, summed in either direction, differed in more than 17,500 convolution
values. The receipt SHA-256 is
`97ee4c6e5e5ce80418cc31fc334977c74dfdd8e0e5108e6a6b8ddbb033b6ebd4`;
its captured candidate tensors hash to
`c49c708f1afde50f95ea378c0a1276563b8dabc8625de9d8ec5d0553bf45ec88`.
The input hash is the one above, verified before and after. This is a single
controlled row, not a full-model or quality result. An explicitly selected
Metal candidate uses the flattened layout together with the previous split
head/upsampling candidate; the default path is unchanged. Full `tools/check.sh --cuda` passed, including 18 device cases
(`reports/phase4-xsum-refinement-layout-cuda-check-v1.log`); Metal
free-generation screens are pending.


At `18a7a7c`, the candidate was built with the MLX feature and screened on
macstudio. The 4864 free-generation batch matches all CUDA codes and raw IDs
(output SHA-256 `8a7ef47db698d66906f238d61085b43ff2e49d7d29154754e8e05eeb7026beb6`).
The 4928 batch still has three code and five raw-ID differences, all on row
4937 (output SHA-256
`1662e6869c012e017df154da76d72d83f634123dd86b874010e73e6335738050`).
Input hashes passed postflight and local/Mac output hashes match. The first
build attempt omitted the required `mlx` feature and failed before compilation;
the corrected build passed. These are bounded screens, not full corpus quality
or performance. The exact Metal raw-ID gate remains failed. A forced-history
trace was subsequently captured as described below.


The forced-history candidate trace records the same level-one choice 3690
(margin `-1.9073e-6`) versus CUDA choice 3182 (margin `+9.5367e-7`). Its
first cascade convolution is exact, while its reconstruction differs from
CUDA in 16,838/18,432 values (maximum `9.5367e-7`), and its input differs
in 4,166/397,824 values (maximum `5.9605e-7`). The trace and receipt SHA-256
values are `12bc470fecf2367e1535b87fed4709cf24395d8165ca1052769897d765ee06c5`
and `97fc103ef17c2b1851bf2a0a88341a44da795ef1eb03d1d1190f14805a30a6ea`;
local and Mac copies agree.

The exact CUDA continuation of the candidate's captured transformer input
selects 3690 (margin `-9.5367e-7`), whereas projecting its captured latent
with CUDA and continuing in CUDA selects 3182 (margin `+9.5367e-7`). The
CUDA-latent control exactly reproduces the retained CUDA input and hidden.
Thus target projection or its interaction with the target embedding is a
further independently sufficient difference in this one controlled history.
The ignored intervention script/log hashes are
`d726d821829e630eed46f1529a3be721dc06e2b878e2a2e9551a7e731d52eea1` /
`9f48f17b45c0f69e9b5727ac4c613fd85d852ad5deee06a8b30b80e5fd6f04eb`
for block boundaries, and
`d3f06f3764161e9444fa81569b5f755dbb49b830d9aa75345d708d437638539e` /
`b79e27b1aa215f919c23c107f28f1cf15102a8c9c95e3b37660b988edab1744a`
for latent intervention. This is not proof of a unique cause, a candidate
fix, or full-corpus evidence.


The [target projection screen](../../../tools/reference/xsum_target_projection_metal_probe.py)
used the candidate's captured latent and identical batch-32 geometry. Metal
`addmm` and `matmul` plus bias both exactly reproduce the candidate's captured
projection but differ from the CUDA projection in 3,866/4,608 values, maximum
`5.9605e-7`. A two-part projection reduces the mismatch to 1,528 values,
maximum `1.1921e-7`, but the exact CUDA continuation still chooses 3690. A
three-part projection yields an exact tie in the CUDA continuation, which
chooses 3182 by its tie behavior; four and six parts choose 3690. All are
bounded arithmetic interventions, and a tie is not evidence of a robust fix.
Input, Metal result and output-tensor SHA-256 values are respectively
`beba7c1c7672482d4f49b5629b4bc0d208a6412ad291b2e2a473c0a3cae6c953`,
`2fc8a4f03a54730324e15d032a00c04682110ff7ab8603a719d2bd25b0c4da44` and
`88228a241002174a4dd361972bc4601b2efd9b3ff3ef429edbe69e3ff5e92e14`.
Local and Mac copies agree; input hashes passed before and after. The CUDA
continuation script/log hashes are
`bf75a0f4804660d880e50efc68650bc71862686b75ef1daa0a031f0a79f70659` /
`5cea3bac4c4aacea5f27b30fc21d509c5e9572e5ca35201c114d09e75e979312`.
No default inference or quality gate changed.


At `77ba69f`, an explicit diagnostic combined the refinement candidate with
three 128-wide target projection products for the six-vector level. Full
`tools/check.sh --cuda` passed, including 18 device cases
(`reports/phase4-xsum-target-projection-cuda-check-v1.log`). The Metal build
passed and both free-generation batches are byte-identical to the prior
refinement candidate: batch 4864 matches CUDA; batch 4928 retains three code
and five raw-ID differences on row 4937. Input hashes passed postflight.
Output hashes are `8a7ef47db698d66906f238d61085b43ff2e49d7d29154754e8e05eeb7026beb6`
and `1662e6869c012e017df154da76d72d83f634123dd86b874010e73e6335738050`.
The controlled CUDA tie did not translate into a free-generation fix. The
path is experimental; the default inference and frozen exact gate are
unchanged.


A [standalone Metal FMA screen](../../../tools/reference/xsum_projection_fma_metal_probe.py)
compared 20 explicit accumulation orders against PyTorch CUDA at the same
projection geometry, using the captured latent and four CPU-seeded synthetic
inputs. Two contiguous 192-element FMA accumulators summed before the bias had
the fewest differences: 1,528/4,608 on the captured input and 1,488–1,528
on the synthetic inputs. For a separate CUDA capture with zero bias, this
same order matched every one of 23,040 values. The nonzero-bias CUDA results
still differ, so bias handling or an algorithm difference remains; this is
not an inference change or a gate result.

The nonzero-bias input/result SHA-256 values are
`3ad2d6a984197e6a10390eac8964512568af717b5443e08271b7e3469eb1699b` /
`c029b48b416eab3724e6edb20988b9ee2c64b306dd1fd79aeb61e6b7f66560e9`;
the zero-bias input/result values are
`fa81cc3bd7d185feec621753021692a554224255aee246fe1f518ea325a0edd8` /
`ff1364abd5a580731d26f6ec7dbc5fa27478fe37ea0071341fc00b32575542f4`.
Mac input hashes were checked before and after, and local and remote receipt
hashes match. The ignored preparation scripts hash to
`a76fabcc8c36a525a7c632520d7398d4517f7f44f3140f893725b37f54d0061d` and
`4c2e50109dba515d3bbf531107f2a3fb58fe1a5321c42bebb4c746698d35ed81`.
A separate Metal bias-epilogue screen matched every CUDA value on all five
inputs when adding bias to the first accumulator before adding the second.
The other two parenthesizations differed in at least 1,287 values per input.
The receipt SHA-256 is
`f5de92e011c48b86d30bdcbc872919fb105bc63945ef408f514c8bb89998a212`;
local and Mac copies agree, and the input hash passed before and after.
This is a bounded custom-kernel result, not a full-model or quality result;
equivalent native Metal matrix products also match all five controlled
inputs exactly; the receipt SHA-256 is
`5ab1fbdf234461130f857e214d0596df95a1581352052bc268f5b5c1a2f95a62`.
An explicit runtime diagnostic passes `tools/check.sh --cuda`, including 18
CUDA device cases (`reports/phase4-xsum-reference-projection-cuda-check-v1.log`);
Metal regression and free-generation checks remain pending. The default
backend is unchanged.


At `82346ed`, the reference-order candidate passed full
`tools/check.sh --cuda` including 18 CUDA cases and its Metal regression
`experimental_xsum_projection_preserves_reference_bias_order`. The
Metal release build passed. Both free-generation batches are byte-identical
to the preceding refinement candidate: batch 4864 matches CUDA, while 4928
retains the three code and five raw-ID differences on row 4937. Their hashes
are `8a7ef47db698d66906f238d61085b43ff2e49d7d29154754e8e05eeb7026beb6`
and `1662e6869c012e017df154da76d72d83f634123dd86b874010e73e6335738050`;
input hashes passed postflight. The forced-history trace is described below. This is not full-corpus quality
or performance evidence, and the exact gate fails.


The candidate forced trace SHA-256 is
`005f759f7f2d41fcdbe051f4ce3b2776e9c1d0c32de2221e58a9e771756f4772`;
its receipt is `97fc103ef17c2b1851bf2a0a88341a44da795ef1eb03d1d1190f14805a30a6ea`.
Local and Mac copies agree. Its latent and target projection match the
previously prepared same-input CUDA projection input and output exactly,
although its latent differs from the retained CUDA latent. An exact CUDA
continuation from this candidate input selects the CUDA reference code,
whereas substituting its Metal block-0 output selects the Metal code. The
CUDA controls at all inspected boundaries exactly reproduce the retained CUDA
hidden and reference margin. Script/log SHA-256 values are
`9e6d9cc36699bb81522fb26da929ed77ade373fc1f016976f21013df8885dc96` /
`2ced4aecbc1aa65a9e91267f0b00db3dc78c33d4980615d11ee9528bb1af8c75`.

With the candidate input, separately substituting captured first-block Metal
normalization, QKV, attention, second normalization, MLP input or GELU output
into CUDA also selects 3690; substituting the first residual selects 3182.
Replacing only the prefix or only the target part of first normalization each
selects 3690. These are distinct interventions, not a monotonic causal chain
or proof of a unique cause. Stage script/log SHA-256 values are
`05d5fa01e9d5327da610f58065f5917beda37bc55856b6455ae8101ffa1aeb49` /
`0515c0c00cdf04e7aa8999db1df9928612c46d05bf5302ffce1c31bdd2cdb750`;
prefix/target script/log values are
`e06fbbb64b3a2f5e6828bc79039898e09068e921bdadaf89de16db2e11a15e4b` /
`1bfb0100593f5db1b36fa225beb812d88737209e77a744a337b0f206c2ebc47a`.
The same-input [normalization comparison](../../../tools/reference/xsum_candidate_norm_metal_probe.py)
reproduces the captured fast Metal output exactly; it differs from CUDA in
160,441/397,824 values, while the Welford alternative differs in 31,190.
Substituting either output into the exact CUDA continuation still selects 3690,
with margins `-9.5367e-7` and `-2.8610e-6` respectively. The input, receipt
and tensor SHA-256 values are
`168e9f614032fb145ea76d22bf14e404f54311ba222cb04346f3838c98cb23f5`,
`a8bf6be4e9b3f4a70524e031a9d9335384f1cb59931d88ffc672413a2f6836b4` and
`5e848dd35d1db16ebfa9f54bef6bf756ae05b28c7fe19bc20a693db7f0c679ee`;
local and Mac copies agree. The continuation script/log hashes are
`7e428a5d4c6320a64ea8397cbce337ae829bcd277c30463184f5f6e3663b9eab` /
`02b1e98636dae03974a54cd86f024e7b6137efd602c8ec84ccb98cdd793a9732`.
A statistics comparison is prepared. These do not qualify a model change or
a gate.


The safe Welford [statistics comparison](../../../tools/reference/xsum_candidate_norm_stats_metal_probe.py)
matched all 518 CUDA means; 53 of 518 reciprocal standard deviations differed
(maximum `4.7684e-7`), and 31,190 of 397,824 outputs differed. Relaxed and
fast math each differed in 60 reciprocal standard deviations and 198,459
outputs. The input/result SHA-256 values are
`249ed8d48f81283b2b8df34150d6a857975319903ac33e0daa57790664f5dba0` /
`088873a918959e8ea5ea8a03cfb0b30ba09de28815fd720d70893527b93ecc18`.
The receipt input was checked before and after; the forced-statistics control
is pending. These observations cover one candidate history, not a gate.
