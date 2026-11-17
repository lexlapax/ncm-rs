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
