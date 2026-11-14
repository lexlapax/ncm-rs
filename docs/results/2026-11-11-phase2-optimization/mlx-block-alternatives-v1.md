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

The separate layout and process-state investigations are recorded below and in
[state diagnostics](mlx-block-state-v1.md). None of these diagnostic
results replaces the failing Apple initial audit, changes acceptance limits, or
establishes why the regression occurred.

## Retained latent layout

A separate instrumented example evaluated all 12 retained encoder snapshots at
batches 1, 2 and 4 in isolated `3307dff` and `2df27c3` worktrees. All 36
snapshot shapes and exact float-byte hashes matched. The final resident-block
input had shape/strides `[1,89,256]` / `[22784,256,1]`, `[2,89,256]` /
`[22784,256,1]`, and `[4,89,256]` / `[22784,256,1]` in both versions. Ten
intermediate batch-1 snapshots differed only in the stride of the degenerate
batch dimension; no batch-2/4 or final snapshot strides differed. These
observations do not rule out differences in internal graphs, allocation state,
or subsequent operations, but do not support a changed final latent stride as
the explanation for the block regression.

[Raw layout reports, diagnostic source and build logs](raw/mlx-latent-layout-v1.tar.xz)
have SHA-256 `fab07d08dc4bd2d499b43fb611c2179a6ea83c77bb1954de846cbb038d586f9e`.
The Apple-only Q/K/V alternative below did not justify a production change.

## Apple-only unpacked Q/K/V

An isolated candidate restored the earlier Q/K/V slicing layout only on macOS
while keeping the Linux packed path. It passed Metal component/operator tests
and all 642 trained traces / 2,500 decisions. Three fresh matched processes
per binary showed resident-block median ratios of **1.002/1.218/0.987** at
batches 1/2/4 relative to the current combined candidate. The batch-4 result
is still far above the accepted baseline observed in the initial gate, and the
batch-2 diagnostic regressed. It is not promoted; the small sample also shows
substantial variation in unrelated scopes. [Raw reports, patch and qualification](raw/mlx-unpacked-poc-v1.tar.xz)
have SHA-256 `57863d0ead951603741c0c22388802e65192ad07fbd4336caa4e1f2c20f95932`.

## Single-gather interpolation

An isolated interleaved single-gather implementation passed component tests
and all 642 trained traces / 2,500 decisions. Three matched processes per
binary showed block median ratios versus the current candidate of
**1.011/1.120/1.038** at batches 1/2/4; reconstruction ratios were
**1.011/1.215/0.971**. Output ID hashes matched. The small diagnostic showed
variation across other scopes, so it does not establish a sufficient,
non-regressing improvement and the candidate was not promoted. It does not
replace the failed Apple gate.

[Raw reports, patch, source and qualification](raw/mlx-interleaved-poc-v1.tar.xz)
have SHA-256 `0720b9d8d6a9743465f2f34b6838c607fb40234b0b05a29ae020c402deca198b`.

## Decoder-specific fast normalization

An isolated candidate used the pinned fast normalization primitives in the
decoder middle block while keeping the encoder normalization implementation.
Metal component and operator tests passed, as did all 642 trained traces with
zero ID disagreements against the pinned CUDA reference. Three matched fresh
processes showed block median ratios of **0.814/0.684/0.790** versus the
current candidate at batches 1/2/4, with reconstruction ratios of
**0.976/0.820/0.827**. Encode ratios were **1.079/1.088/1.061** even though
the encoder's normalization path did not change. Process medians varied
substantially (for example, candidate batch-4 encode ranged 5.57–10.05 ms).
Output IDs matched, but this small mixed diagnostic is insufficient for
promotion or acceptance. A refined candidate is being examined separately.

[Patch, tests, qualification and raw reports](raw/mlx-blocknorm-poc-v1.tar.xz)
have SHA-256 `7585d9cf826af9482287223d689326b2a76bf562e7aaf20aace3d41f006eb837`.

## Decoder fast normalization and attention refinement

The refinement leaves the ordinary encoder normalization call unchanged and
uses the previously tested fast attention primitive with fast decoder
normalization. Metal component/operator tests and all 642 trained traces
passed, with zero ID disagreements. Three paired processes per binary gave
block median ratios versus the current candidate of **0.771/0.734/0.744**,
encode ratios **0.989/1.075/1.138**, and reconstruction ratios
**1.023/1.202/0.947**, at batches 1/2/4. Output IDs matched. For context,
the candidate's sampled batch-4 block median was 1.245 ms; the earlier
initial gate's accepted baseline median was 1.206 ms. Those measurements were
collected at different times and are not an acceptance comparison. The mixed
results, particularly encode and reconstruction variation, require the complete
fixed gates. A production candidate based on this experiment is being
validated; the 69/78 Apple audit above remains the latest acceptance result.

[Patch, tests, qualification and paired reports](raw/mlx-blocknorm-poc-v2.tar.xz)
have SHA-256 `9976049633f2938085d52b47e65070cddd80553f2db933f2f63d35d5f0e9b760`.

The formatted production implementation passes `tools/check.sh --cuda` with
`RUSTC_WRAPPER=` and umask 077: 483 report tests and 18 real-CUDA operator
cases pass. The initial wrapper-enabled check could not start `sccache` in this
environment. The CUDA codec executables changed, so the prior Linux 78/78 gate
and trained capture do not establish acceptance for this implementation; new
Linux quality and performance measurements are required before the final Apple
checks. The newly added direct Metal normalization regression has not run yet.
[Linux check, CUDA report and binary identities](raw/mlx-blocknorm-linux-check-v1.tar.xz)
have SHA-256 `39bf25bce3931ca3cde02f641c850139c7bc4c0b16f0a6477055d2ebf022c80e`.
