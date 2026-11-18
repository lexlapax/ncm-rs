# WMT14 Metal corpus and bounded retry

The [first Metal CLI comparison](mlx-cli-smoke-v1.md) found that a four-row
guided batch starting at 2656 failed the existing head guard. Running
each row independently matched Python. The follow-up kept the original
four-row grouping to identify the failure and then evaluated both full
3,003-row modes with the frozen inputs and references.

## Finite guided tie

The guided head probe completed all 11 levels on CUDA. Two Mac runs
reproduced a finite tie between codes 1813 and 1922 at row 2658, level 9,
position 8. Their common FP32 score was `11.906917572021484`. Pinned
Python chooses 1922 on its own scores. Replaying the captured Metal
score bits in the pinned Python CUDA sampler instead chose 1813 in
every tested run, across six seeds and repeated-score batch widths 1,
2, 4, 16 and 32. Accepting a tie by that rule would change the output.

With the captured Metal conditional and null branches, separate FP32
operations reproduce every guided score bit. FP64 mixing selects
1813 by about `5.1e-7`. One fused form selects 1813 and the other
fused form retains the tie. The difference-form FP32 expression
also ties. The captured branch values therefore already favor the
other code in higher-precision mixing. The two-row Metal case
completes and differs on row 2658 by two codes and one raw ID.

## Native Metal corpus

The native recorder loads one model and preserves every successful
batch or failed range. It exits nonzero when any batch is rejected.
Each mode attempted the same 751 batches as pinned Python, with the
three-row tail. The resulting comparisons are exact over returned rows.

| CFG | Returned rows | Rejected batch start | Intermediate code differences | Raw ID differences |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2,999 | 1376 | 13, all in row 632 | 0 |
| 1 | 2,999 | 2656 | 0 | 0 |

At row 632, the first difference is code 2602 versus 270 at level 4,
position 17. It persists at every tested native batch size from 1
through 32, with identical raw IDs. Base batch 1376 passes at sizes
1, 2 and 8, rejects at size 4, and has 36 code and 17 raw ID
differences at sizes 16 and 32. The guided 2656 range matches
in tested native batches of 8 and 16.

## Scoped candidates

The [explicit stride-four projection variant](../../../crates/ncm-backend-mlx/src/codec.rs)
changes the three nonoverlapping WMT14 transposed convolutions to
matrix projections. It recovered the rejected base batch exactly
on the targeted fixture. In the guided batch it returned the same
two code and one raw ID differences as the smaller native case.
Row 632 retained its 13 code differences. The candidate left the
ordinary backend on its native path.

Duplicating rows to reach eight internal inputs recovered both
rejected batches exactly. The external grouping remained four,
and both three-row tails matched their pinned outputs. The
experimental corpus recorder retries only after the original head
guard returns an error, records the retry batch size, and retains
the original guard on the recomputed scores.

In the complete base-mode retest at `6d5f07f`, batch 1376 was the
only retry. All 750 previously successful batch records were
byte-identical to their native baseline records. The resulting
3,003 rows match every pinned raw ID, with row 632 retaining its
13 intermediate code differences. The frozen full and held-out
scorers pass in two independent offline environments with
byte-identical results. Both hypothesis streams are identical, so
the paired bootstrap differences are zero.

In the complete guided retest, batch 2656 was the only retry.
The other 750 records were byte-identical to their native baseline.
All 939,939 codes and 192,192 raw IDs match pinned Python across
the 3,003 rows. Both frozen scorers pass the held-out and full
checks. Their results are byte-identical and match the prior
CUDA scorer output. The replay preserves the original rejected
captures and uses previously inspected data.

Run evidence is retained under ignored
`reports/phase4-wmt14-guided-head-v1.vo6Wpv/`. Relevant receipt
SHA-256 values follow.

| Evidence | SHA-256 |
| --- | --- |
| Native CFG-zero corpus comparison | `0ba5dcb84a31dfc1f9997f9b9787e10aecc47b1ec279745f67223c902b8dd3de` |
| Native CFG-one corpus comparison | `e60aab723354eda8429621cdb9e29c23edc2c807922320aa77e45cb7b56dfeaa` |
| Metal conditional and null score bits | `6272d997ccfe8ed2961f8a5e508a4e331bed1065a5a9ac59afd0fd1605ee2213` |
| Python tied-score replay | `e5bc1d2520f560088a66145a4a2944b1bcda2dfd38848111a2aeec75cb2fce8c` |
| Guidance mixing control | `115a0feb643301e859da5045b21501cc4e9418fcb338c82c52d324ad3deda19b` |
| Base retry exact comparison | `5b94302fcb7001bb0ba1130b78f78899d0100bc3d05c511309de0d47c2f57582` |
| Base retry frozen scoring | `5598f2d2d1a484d077c875874b8105f004ad9af68149e801bdc58246f81ee62b` |
| Guided retry exact comparison | `874c1ee5ac482b3b17ec4918cd1c50d2dc33ced522be70a4dd5565aaf462287c` |
| Guided retry frozen scoring | `915cae3bf2351db1f90bd290ebbe79353a4b9bdb91a67cb494d30791403f89bc` |

The [Metal teacher-forced comparison](teacher-cuda-reference-v1.md#metal-comparison)
passes all 26 tensors. The CLI exposes the same bounded retry through
the experimental `mlx-retry-padded` device. The original corpus
capture predates the shared implementation. The targeted Metal CLI
and capture comparison below passes. Paired latency and memory
qualification remains open.

## Shared CLI and capture verification

At `854cfd6`, the CLI and shared recorder agreed on four ranges in
each mode. The eight controls contain 1,920 raw IDs, all matching
pinned Python CUDA. Guided codes also match exactly. Base row 632
retains the 13 previously observed intermediate code differences.
The native device continued to reject base range 1376 and guided
range 2656. The opt-in device recovered both, reported the recovery
on standard error, and wrote the expected codes and IDs for all
eight rows. The other ranges, including the three-row tail, produced
byte-identical CLI output in both device modes. Five-row retry
requests failed before producing any output.

Feature-enabled release build and Clippy passed on Mac Studio.
Full Linux CUDA and Mac Metal checks each passed 18 operator cases.
Mac service snapshots show the same PID. The CLI controls are in
`corpus/owned-loader/phase4-wmt14-mlx-retry-cli-v1.08O1c4/`,
and the full Metal log SHA-256 is
`b345620fdb03449f831b0c210102ab39a39a25d66d5792bb389e1df7ee65ed12`.

## Startup memory preparation

The experimental CLI now takes ownership of the checked inference
artifact and releases the codec payload after upload, before
uploading generator weights. The original borrowed API continues
to be available. The updated CUDA CLI reproduced all 24 targeted
outputs across both modes. Focused CUDA build and Clippy passed,
as did the full Linux CUDA suite with 18 device cases.

Three alternating fresh Linux CLI processes per implementation
used the same fixed base batch of four. Each old/new pair produced
byte-identical stdout. GNU time reported median peak RSS of
`1768.69 MiB` with the borrowed loader and `1462.30 MiB` with the
owned loader, a reduction of `306.39 MiB` (17.3%). A Linux build
was active, so the overlapping wall-time ranges do not qualify
a speed change. The cold observation receipt hashes to
`df9ee27828a8cf3d7300178ee9cbad2396b973d22c8abed60d4090793872b35e`.

On Mac Studio, three fresh runs of each loader in each guidance mode
also produced byte-identical outputs. Both loaders peaked near
`2110 MiB` of process RSS. Heap snapshots found approximately
`1.0 GiB` resident in empty large-allocation regions in the warmed
Rust process versus `552.1 MiB` in the warmed Python process. In a
separate control that allocated and freed the two artifact-sized Rust
buffers, Apple allocator pressure relief returned zero bytes in all
three runs. The snapshots are local to the measured processes.

The Metal timing probe can opt into a reusable host buffer for the
two bundles. It verifies both payloads and their tensor inventories
before uploading weights, then rechecks file size and hash against
the same manifest on each reread. A changed file fails the load. The
normal CLI currently uses the owned loader. The reuse test checks
allocation identity, byte identity and rejection of a same-size
changed file.

Three alternating fresh processes per loader at batch 4 matched the
pinned codes and raw IDs in each mode. Median peak process RSS fell
from 2122.84 to 1664.75 MiB at CFG zero and from 2123.06 to 1664.77 MiB
at CFG one. Native MLX peaks were identical per mode. The median model
load increased from 1.35 to 2.00 seconds in both modes. The Rust-only
screen used one warmup and two timed batches per process. Its RSS
landed about 7 MiB above the nearby repeated Python baseline, which
used longer runs. The same service PID was observed before and after
the test. Raw evidence is under
`corpus/owned-loader/phase4-wmt14-mlx-reusable-v1.xGtw9Z/`.

A follow-up used the actual artifact sizes to test scratch release in
isolation. Three Rust `Vec::shrink_to_fit` trials reduced live RSS by
about 1.7 MiB with the same pointer. In three anonymous-mapping trials,
releasing the unused tail lowered live RSS by 91 MiB, and dropping
the remainder returned to about 1.5 MiB. The device timing probe uses
this copied anonymous scratch on Apple Silicon, trims the whole
unused pages after the final checked generator reread, and drops the
remaining pages following upload. Weight files are still read and
hashed before use. The ordinary owned loader remains available for
paired comparison.

At `fe55e5b`, Mac artifact Clippy and tests passed, including a check
that trimming preserves bytes and that a subsequent larger read can
grow the buffer. Across four short paired cells, all retained outputs
match pinned Python CUDA and native MLX allocation peaks are identical.
Peak process RSS falls from a range of 2,122–2,123 MiB to 1,561–1,562 MiB.
The reusable loader is paired against the ordinary loader with one
warmup and two measured batches per process. The raw evidence is under
`corpus/owned-loader/phase4-wmt14-mlx-vm-scratch-v1.8Brqq9/`. The summary
hashes to `01ddaf042cd0a6d7e4ac1a70abdcf49835366222be5eaec8c83db64b20e56aae`.

The allocation controls are retained in the ignored run evidence
under `corpus/owned-loader/phase4-wmt14-scratch-{shrink,vm}-v1.*`.

## Repeated small-batch baseline

Three alternating fresh processes per runtime each ran five warmups
and 30 measured generation-and-readback trials. All 24 process outputs
matched the pinned prefix codes and raw IDs. Native Rust used the owned
loader. The [Python MLX comparison](../../../tools/reference/wmt14_mlx_reference_bench.py)
adapted the pinned implementation to Metal. The service stayed running. RSS and MLX peak are separate
process and allocator measurements. Startup is excluded from trial time.

| CFG | Batch | Median ms Rust / Python | p95 ms Rust / Python | Peak RSS MiB Rust / Python | MLX peak MiB Rust / Python |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 160.08 / 107.71 | 190.54 / 108.46 | 2122.53 / 1657.59 | 1161.50 / 1174.55 |
| 0 | 4 | 235.62 / 237.16 | 260.00 / 241.63 | 2123.94 / 1657.80 | 1621.24 / 1665.64 |
| 1 | 1 | 212.25 / 151.77 | 246.03 / 155.11 | 2123.69 / 1658.16 | 1315.58 / 1319.22 |
| 1 | 4 | 362.11 / 364.60 | 367.53 / 368.90 | 2124.66 / 1657.55 | 2153.81 / 2142.62 |

Batch 1 remains slower in both modes and process RSS remains higher
at both sizes. The per-process measurements and timing dispersion are
retained under `corpus/owned-loader/phase4-wmt14-mlx-small-paired-v1.dAjuOA/`
in the ignored run evidence directory above. The comparison SHA-256 is
`e71648f7030c5cef0e798dffe30b45524fd6dd21415dbba5a4e09e7e346f6002`.


## Parallel head reduction screen

At `b640f8a`, an opt-in Metal selector distributes each 4,096-score
row across 32 lanes. Its two real-device tests passed, covering ties,
signed zero, nonfinite scores and a unique maximum at the last index.
All 12 alternating batch-1 timing processes matched pinned codes
and raw IDs. With three warmups and 12 timed trials per process, the
pooled medians changed from 155.37 to 170.06 ms at CFG zero and
from 201.16 to 207.25 ms at CFG one. The experiment remains reproducible at that commit. The slower
selector was removed from current source.

The ignored evidence under
`corpus/owned-loader/phase4-wmt14-mlx-simd-head-v1.KX8PET/`
includes the unchanged service snapshots, compiler logs, test log
and timing samples.

## Deferred head validation screen

At `0ac559b`, the explicit Metal probe uses the same unique-maximum
selector. It accumulates the raw results and checks all sentinel
codes once before decoding. Until then, bounded embedding lookups
clamp invalid indices for safe intermediate evaluation, leaving the
original codes available for the final check. Both known rejected
ranges still fail with the same error and produce no output.

Three alternating fresh processes per setting each ran three warmups
and 12 timed trials. All 24 process outputs matched the pinned codes
and raw IDs. The owned loader was used in both settings.

| CFG | Batch | Median ms immediate / deferred | p95 ms immediate / deferred | Peak RSS MiB immediate / deferred |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 146.99 / 95.85 | 189.81 / 96.59 | 2122.58 / 2134.11 |
| 0 | 4 | 228.27 / 216.01 | 238.85 / 221.32 | 2123.73 / 2135.38 |
| 1 | 1 | 201.21 / 137.56 | 241.83 / 138.84 | 2123.16 / 2134.84 |
| 1 | 4 | 371.77 / 354.57 | 377.11 / 358.51 | 2124.86 / 2136.28 |

MLX peak allocation was unchanged in each pair. The native service
PID stayed the same. The short screen compares two Rust variants
and does not complete the repeated Python comparison. The ignored
receipts and compiler logs are under
`corpus/owned-loader/phase4-wmt14-mlx-deferred-head-v1.dTSmQ0/`.

A wider short screen used two fresh processes per setting, two warmups
and six measured trials. All 12 process outputs matched pinned Python
CUDA. Deferred validation reduced pooled medians from 479.39 to
472.83 ms at base batch 16, 858.89 to 831.70 ms at base batch 32, and
847.47 to 840.82 ms at guided batch 16. MLX peak changed by +22.44,
-29.06 and +29.45 MiB, respectively. The measurement uses the owned
loader. The retained summary under
`corpus/owned-loader/phase4-wmt14-mlx-deferred-head-large-v1.5Ol4Vj/`
hashes to `090d66892f28353159b85afd96744b192549ec6bd28fe07dc28ccf360b1a2a2b`.

## Long-run latency follow-up

The combined Rust candidate passed all pinned output checks at base
batch 16 in the first repeated comparison. Its three process medians
were 473.23, 2689.08 and 847.64 ms, compared with 629.93, 629.79 and
663.85 ms for adapted Python MLX. Rust's pooled median and p95 were
681.85 and 4479.36 ms, versus 631.73 and 729.33 ms for Python. In the
slowest Rust process the initial trials were around 0.47 seconds
and the final ten were above four seconds. Peak process RSS and
native MLX peak were lower for Rust in the cell.

The three Rust processes retained about 13,911 MiB of MLX cache
after their final trial. The ending cache measurements are close
across Rust processes with very different latency profiles.
The service snapshots and system counters do not identify the source
of the tail. The timing probe now accepts a process-local
`cache-mib=N` setting and a separate `release-outputs` mode. The latter
drops logits before readback and includes code/ID buffer release and a
GPU barrier in timing. Trial start timestamps can be compared with
read-only activity snapshots. These controls are opt-in.
