# Linux synchronized live-pool observations

The published Linux 78-cell audit passes its timing, quality and **host** RSS/HWM
checks. It does not adjudicate the separately declared 1.05 comparable
steady-state live-memory band. An offline inspection of the same retained Linux
matrix and control found a material native CUDA pool-used difference.

The codec harness retains the model and an evaluated latent, drops each returned
output, synchronizes the CUDA device, then reads its default async pool's
`used_current`. Both Rust builds use this pool. All three matrix and nine
control processes per runtime and all four scopes at each batch produced the
same values within a batch:

| Batch | Previous Rust pool used | Candidate Rust pool used | Candidate / previous | Increase |
| --- | ---: | ---: | ---: | ---: |
| 1 | 59,677,184 | 63,897,168 | 1.070714 | 4,219,984 |
| 2 | 59,768,320 | 63,988,304 | 1.070606 | 4,219,984 |
| 4 | 59,950,592 | 64,170,576 | 1.070391 | 4,219,984 |

Thus every one of the 24 matrix/control scope snapshots exceeds 1.05 for this
same-allocator native comparison. The observed increase equals 4,194,304 bytes
plus 25,680 bytes. The previously approved [strict-FP32 projection](fused-projection-review-v1.md)
retains a 4 MiB workspace per backend on this Ampere GPU; the earlier
[prepared-state diagnostic](../2026-11-11-phase2-linux-controls/README.md)
measured 25,680 additional logical pool-used bytes. This is strong accounting
consistency, not independent attribution of each allocation in these reports.
The workspace patch was approved with its memory cost disclosed; that approval
was not recorded as a waiver of the 1.05 band.

This is a comparison of **default CUDA pool used bytes**, not complete GPU
memory: non-pool context/cuBLAS allocations are excluded. The Python reports use
Torch's allocator counters with different coverage and rounding, so this review
records those bytes but does not claim a cross-allocator ratio. Output IDs,
fixture identities and retained-output element counts match; the reports do not
hash the retained native latent values in these snapshots. These limitations do
not make the observed native increase disappear. Whether this pool observation
is sufficient to adjudicate the contract's comparable steady-state live-memory
criterion must remain explicit; it cannot be treated as a pass based on the
78-cell host-memory audit. No performance or memory limit was changed.

The [review script](linux-live-pool-review-v1.py) rehashes the retained archive,
the published audit and all 36 codec reports, verifies workload/output identity,
and reports every scope. No hardware was accessed or rerun. Raw evidence remains
in [the original Linux gate archive](raw/tiny-borrow-linux-gate-v1.tar.xz).
