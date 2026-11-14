# Apple gate and isolated MLX interpolation prototype

Candidate `cd49989` passed the real-Metal check, the four real tokenizer
profiles, and the trained qualification: 642 traces and 2,500 decisions,
zero ID disagreements, with unchanged oracle and tolerances. The full Mac
check ran 373 selected report tests (166 platform-skipped) and explicitly
excluded 110 tests in Linux-only diagnostic modules. A preceding check failed
28 tests and errored on 27 because those modules used Linux interfaces; the
failure is retained. The first direct Metal check also passed 18 operator cases.

A fresh 45-process Apple matrix and separate nine-process lifecycle capture
produced an initial **76/78** audit. Both BERT decode cells passed point
estimates but not the process envelope. One predeclared BERT control measured
nine fresh processes per runtime, every BERT scope and checked load. Its plan
SHA-256 is `9b0b0928683856113eca360a942e81815879833e5752f602306d891a5889d93f`.
The controlled audit is **77/78**: BERT encode versus accepted Rust has median
ratio 0.98186 and p95 ratio 1.02533, but its whole-process p95 interval upper
bound **1.11420** exceeds the unchanged 1.10 limit. The earlier decode cells
pass. No follow-up was retried. The historical audit reproduces byte-for-byte;
its SHA-256 is `459b08aa3222248b8c94f38c9a52c7563256b12d3f2759edebfa92c0f9d36a17`.
An existing `rmlx` server remained running but was idle in activity snapshots;
macOS process telemetry is retained. This result is not an Apple gate pass.

The isolated `b30fd31` interpolation example compares the production scalar
MLX graph with gather operations using identical host f64 coordinates and f32
weights. Three fresh processes per mode used five warmups and 30 trials across
six input/output shapes. All measured outputs matched exactly. For batch 1,
13 to 89 positions, median-of-process graph time was **0.1806 to 0.0057 ms**;
graph-plus-evaluation was **1.0656 to 0.2966 ms**. Across the sampled shapes,
evaluated ratios ranged 0.270–0.691. Peak MLX counters for that case increased
from 82,776 to 137,248 bytes; batch 4, 13-to-89 increased from 407,684 to
802,592 bytes. These per-process allocator peaks are not host RSS or a full
model measurement. A first command exited at its usage check due to shell
argument splitting, before collecting any sample; the declared six processes
were then run once. This isolated result is not full-model performance or parity.

A production candidate replaces per-position linear graphs with the same
f64-coordinate gather formulation and adds expanded Metal comparison cases.
It now passes the expanded Metal tests and the unchanged trained oracle (642
traces / 2,500 decisions, zero ID disagreements). Its Linux CUDA check also
passes. A six-process matched full-model diagnostic compared the candidate
against the immediately preceding native implementation. Median-of-process
encode ratios for batches 1/2/4 were 0.329/0.326/0.552; decode ratios were
0.323/0.533/0.610; reconstruction ratios were 0.480/0.497/0.585. Output ID
hashes matched. These are diagnostic results, not the final acceptance gate.

Peak MLX allocator counters rose by about 9.5 MB in batch-1 reconstruction
and 26.3 MB in batch-4 reconstruction; maximum observed host HWM decreased
in this small run. Separately measured resident-block ratios were
1.078/1.025/1.308 for batches 1/2/4 despite no block algorithm change. The
small sample cannot settle that variability; complete acceptance and resource
checks remain necessary. The BERT failure above must not be erased by retrying
the unchanged tokenizer candidate.

[Candidate validation and matched codec reports](raw/mlx-gather-candidate-v1.tar.xz)
have SHA-256 `0df59284aec00bf51a91e95f942e77b0c2a093dfbc712dece5a9455e37fd5932`.
The exact tested source hashes are `7ecbdb8fd77838bc17e6375c40a48245a26a0b84afef813cdf4eac0b052df190`
for `ops.rs` and `650cb609bbe8a53f0c8f999d6299cb7652e7ca713d5942391889fbb113d59a92`
for the expanded operator test.

[Raw Apple gate and prototype evidence](raw/vendored-mlx-gate-and-linear-poc-v1.tar.xz)
has SHA-256 `7d48133e3f44d02bdb8365a577c9acbf994bbea1b1d422193bd85acafb784364`.
The archive includes telemetry, plans, logs, qualification, lifecycle, initial
and controlled audits, and six prototype process reports; trace tensors remain
outside Git. Extract to the repository root and run the existing audit with
`reports/phase2-vendored-mlx-gate-v1/controlled-config.json` to reproduce the
expected failing result.

## Historical Apple audit: clean-export reproduction

Exported tested commit `cd49989` with `git archive` into an empty temporary
directory, extracted the published Apple gate archive there, and ran the
existing audit with its archived controlled configuration. It returned the
expected failure and reproduced the **77/78** audit byte-for-byte, with the
SHA-256 above. Independently verified every path and SHA-256 in all **48
native Rust report source inventories** across the Apple matrix and BERT
control against the clean candidate export or clean `3307dff` baseline
worktree; no mismatches were found. This strengthens provenance of the
historical failure; it does not qualify the later combined candidate or
replace the pending Apple gate.
