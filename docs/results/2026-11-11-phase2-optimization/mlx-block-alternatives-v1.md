# MLX decoder-block alternatives — isolated diagnostics

The fresh Apple matrix for `2df27c3` exposed a substantial batch-4 resident
block regression. Two isolated candidates were evaluated; neither changes the
production implementation. Each passed real-Metal operator/component tests
and all 642 trained traces / 2,500 decisions against the unchanged oracle, then
ran three matched processes per binary with five warmups and 30 trials across
all existing codec scopes. Output ID hashes matched. These are diagnostics,
not acceptance runs.

- **RMS-only fast primitive:** resident-block median ratios versus the current
  candidate at batches 1/2/4 were **0.984/0.919/0.951**. Batch-4 remains far
  from the accepted baseline; the same small diagnostic showed large encode
  variation (batch-2/4 ratios 1.331/1.243) even though encode does not call
  RMS normalization. This does not justify promotion. [Raw RMS evidence](raw/mlx-fast-rms-poc-v1.tar.xz)
  SHA-256 `07f1f2ea2485e4c3b7dfdbf7e34ba0a7af066f41a05d49d54fd1796b82d63352`.
- **Pinned fast attention:** resident-block ratios were **0.968/0.923/0.924**;
  batch-1 decode and batch-4 decode ratios were 1.134 and 1.160. Encode also
  varied substantially despite not calling the changed decoder attention. The
  result does not establish a sufficient, non-regressing fix. [Raw attention evidence](raw/mlx-fastattention-poc-v1.tar.xz)
  SHA-256 `08f7bf7765c498b6d0de072db94c707d2e506cd93f94f3cb7ee4bcddb14ef9cd`.

A separate layout investigation is in progress. None of these diagnostic
results replaces the failing Apple initial audit, changes acceptance limits, or
establishes why the regression occurred.
