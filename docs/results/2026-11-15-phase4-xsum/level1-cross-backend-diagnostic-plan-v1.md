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
