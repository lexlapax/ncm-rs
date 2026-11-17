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
