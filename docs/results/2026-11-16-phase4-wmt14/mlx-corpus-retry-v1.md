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

## Repeated combined comparison

The combined Rust candidate at `fe55e5b` used the reclaimable
buffer and deferred head validation. Three alternating fresh
processes per runtime ran five warmups and 30 synchronized trials.
Every retained output in the six completed cells matches the
pinned Python CUDA codes and raw IDs. The Python performance
baseline was the separately adapted Python MLX implementation.
Model load was about 2.02–2.06 seconds for Rust and 0.66–0.67 for
Python. The generation measurements exclude model load.

| CFG | Batch | Median ms Rust / Python | p95 ms Rust / Python | Peak RSS MiB Rust / Python | MLX peak MiB Rust / Python |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 96.33 / 107.43 | 97.02 / 108.59 | 1561.23 / 1657.39 | 1162.73 / 1174.55 |
| 0 | 4 | 216.87 / 237.22 | 220.68 / 241.40 | 1561.45 / 1657.91 | 1621.24 / 1665.64 |
| 0 | 16 | 681.85 / 631.73 | 4479.36 / 729.33 | 1561.27 / 1660.73 | 2599.19 / 3274.29 |
| 0 | 32 | 1310.04 / 1284.05 | 1788.67 / 1686.31 | 1561.56 / 1658.77 | 2835.18 / 3925.68 |
| 1 | 1 | 138.30 / 150.32 | 139.48 / 154.65 | 1561.27 / 1658.25 | 1315.58 / 1319.22 |
| 1 | 4 | 354.56 / 382.90 | 358.87 / 386.30 | 1561.30 / 1658.88 | 2153.81 / 2142.62 |

Base batches 16 and 32 fail the repeated latency comparison.
Guided batch 4 has a higher native MLX allocator peak. The user
directed work to continue on Linux, so the last guided batch-16
cell and the queued optimized Metal corpus replay are unfinished.
The first six cells are preserved in the three local snapshots under
`corpus/owned-loader/optimized-*-snapshot.*/`. Their exact comparison
SHA-256 values are `b564720c9897d20d1619fd048b41c7113ccee8fb1cfc6c22425cbf478a9693dd`
for the four small cells,
`d562282be73245202dae9a4f2547e68b05512b88db667fc699a14f6b1f423b02`
for base batch 16, and
`04c36285087a8ea604b3fc7cbb034e0f0d88182ffc908a4dbbb6462d3e731521`
for base batch 32.

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

## Replaying a batchwise capture

Use the checked artifact and the pinned 3,003-row prefix fixture. The
recorder writes one JSONL record per group of four prefixes, including
the final three-row group. The verifier binds the base or guided
reference to its frozen SHA-256, checks the requested range, and
records exact code and raw-ID disagreements, rejected batches and
padded retries. It writes the receipt before exiting with status 1
for a valid capture that has any disagreement or rejection.

```sh
target/release/examples/wmt14_corpus_capture ARTIFACT mlx-retry-padded PREFIXES CAPTURE.jsonl 1 0 3003
python3 tools/reference/compare_wmt14_corpus_capture.py REFERENCE.json CAPTURE.jsonl RECEIPT.json --cfg 1
```

For a targeted replay, pass the recorder's start and count to the
verifier with `--start` and `--count`. The verifier also accepts
CUDA recorder output and runs with the Python standard library.

The WMT14 CLI and recorder now release their own decoder-logit
handle before reading codes and IDs. A backend can still hold the
tensor through the ID selection dependency. The Metal timing
example's `drop-logits` option reproduces the CLI and recorder's
release point. Its separate `release-outputs` option also measures
output release and a GPU barrier. Real-device Metal performance
and allocation effects of this change remain unmeasured.

On Linux CUDA, the release was verified on 24 targeted CLI rows
and 30 recorder rows in both modes, including the three-row tail.
The CLI also matched all 48 reference rows at the maximum base
batch of 32 and guided batch of 16. Feature-specific release build
and Clippy passed. The full Linux check passed 18 real CUDA
operator cases. Raw results are in
`reports/phase4-wmt14-release-logits-cuda-v1.hRpXNM/`.

The verifier reproduced both earlier full CUDA comparisons and
both retained native and padded Metal comparisons. It retained
the native rejected batches and the 13 intermediate base-code
differences at row 632 after retry. The focused tests covered
missing groups, reordered rows, malformed retries and output
values outside the declared vocabularies. Supplying the base
reference for a guided comparison was rejected.

## Repeating the Metal timing comparison

Use the pinned native MLX install and Python MLX environment, the
checked WMT14 artifact and prefix fixture, and copies of the two
frozen Python CUDA reference captures. The runner checks reference
hashes and compares every Rust and Python MLX output with the
selected reference before recording a timing cell. Each cell uses
three alternating fresh processes per runtime, five warmups and 30
measured batches.

```sh
export MLX_SYS_PREBUILT="$PWD/.reference/mlx-native"
RUSTFLAGS="-C link-arg=-Wl,-rpath,$MLX_SYS_PREBUILT/lib" \
  cargo build --locked --release -p ncm-backend-mlx \
  --features mlx,wmt14-experimental --example wmt14_perf
.reference/mlx-venv/bin/python tools/bench/run_phase4_wmt14_mlx_bench.py \
  --output reports/wmt14-mlx-drop-logits
```

The default uses early logit release, as the CLI and recorder do.
Repeat `--cell CFG:BATCH` to run selected cells. On the same cells,
`--lifetime retain` supplies the original output-lifetime control,
`--lifetime release` includes the output release and GPU barrier,
and `--cache-mib N` applies a process-local allocator limit. Run
each setting into a fresh directory. Use `--base-reference` and
`--cfg-reference` if the frozen captures are in different locations
from the default Linux report paths.

The manifest marks the seven-cell matrix complete only after every
required process in every requested cell has a verified receipt and
the watched inputs match postflight. The summary records slower
cells and higher memory separately. A complete summary can contain
performance regressions. Linux can use `--replay DIRECTORY` to
verify retained raw results before a new Metal run.

The Linux replay checks passed on the six retained Metal cells.
The verifier rejected a missing guided batch-16 cell and refused
to launch a new timing run on Linux. At base batch 16, 41 of 90
Rust trials exceeded the Python pooled p95 of 729.33 ms. They
occurred across two of three processes, with 20 in the second and
21 in the third. At base batch 32, 16 of 90 Rust trials exceeded
the Python pooled p95 of 1686.31 ms, all in the third process.
The retained samples warrant repeated process controls across the
large batches. The available data does not identify the cause.

The Linux replay bundle is
`reports/phase4-wmt14-mlx-runner-final-v1.yPzwZi/`. Its summary
SHA-256 values for the four small cells, base 16, and base 32 are
`d73994472a36fdc4e40c8b93473438a7f7ad0b69c7bfbf33b422aed4b6252334`,
`96c979ca3fd8d5a41301253321e728eda7873b14023411e18fc85a4a02461420`, and
`f0578e5f273c15320e33d24d4d6bb4eeb5e6d24632282f6a1beb36f2e754b3c2`.

## Measuring an intermediate evaluation checkpoint

With deferred head validation, the backend can build a graph for the
whole hierarchical generation before the final code readback. To
test the cost of that schedule, the probes accept an optional
`materialize-after=N` setting for N from 1 through 10. It evaluates
the cumulative reconstruction after N completed levels, with the
same operators, batch shapes, and selector checks. On Metal, this
gives the runtime an opportunity to release preceding intermediates
before appending the next level. The setting adds synchronization
and its latency and allocator effects need measurement.

Screen the two base latency failures with `--with-control`. The
runner holds logit lifetime and cache policy fixed, and measures Rust
with and without the checkpoint against a shared Python baseline. Each
configuration gets three fresh processes, five warmups and 30 measured
batches per process. The order rotates so each configuration runs in
each position once per cell. Across base batches 16 and 32 the runner
uses all six possible process orders.

```sh
.reference/mlx-venv/bin/python tools/bench/run_phase4_wmt14_mlx_bench.py \
  --cell 0:16 --cell 0:32 --materialize-after 5 --with-control \
  --output reports/wmt14-mlx-tail-level-five
```

The screen uses 18 process launches across the two cells, compared
with 24 in two separate paired runs. The six Python launches it
eliminates took 209.81 seconds in the retained Metal runs. Actual
elapsed savings depend on the new run. The summary compares both
Rust configurations with Python and the checkpoint with its Rust
control. Use the ordinary two-configuration schedule with the selected
Rust settings for the full timing matrix.

On Linux, a fixture exercised the nine-launch schedule, replayed its
summary byte for byte, and rejected a Rust control with a different
logit lifetime. All six archived Metal timing cells reproduced their
prior summaries byte for byte. Full Linux validation passed with
530 report tests. The log SHA-256 is
`fd55b3315a4eb6596a1c4f61d44a2f5e0889bac642447e86adef7c26a4ba95b2`. These checks are recorded under
`reports/phase4-wmt14-shared-mlx-screen-v1.MBYFBQ/`. Fresh shared-baseline
timing still requires Apple Silicon.

The recorder takes the same option for a corpus comparison with a
measured configuration. For example, capture the guided corpus
with the padded retry and the level-five checkpoint, then run the
same JSONL verifier as for the original padded replay.

```sh
target/release/examples/wmt14_corpus_capture ARTIFACT mlx-retry-padded PREFIXES CAPTURE.jsonl 1 0 3003 materialize-after=5
python3 tools/reference/compare_wmt14_corpus_capture.py REFERENCE.json CAPTURE.jsonl RECEIPT.json --cfg 1
```

### Local CUDA validation

On Linux, the release CUDA recorder used `materialize-after=5` with the
original four-row batches and immediate head checks. All 3,003 base and
3,003 guided rows match the frozen intermediate codes and raw IDs. There
were zero rejected or retried batches. Both JSONL files are byte-identical
to the [early-release CUDA captures](cuda-fast-argmax-plan-v1.md#full-cuda-replay-with-early-logit-release).
The recorder rejected level 0 and the final level 11 before creating an
output file. The CUDA release build and feature-specific Clippy passed.

The local evidence is in `reports/phase4-wmt14-materialize-cuda-v1.Lf0P1g/`.
The SHA-256 values for `comparison-0.json` and `comparison-1.json` are
`57431f5d21af3e4b91066a78eeec4e7ad933907b23785fbe86033ec3ba1ca409` and
`6343ea547f36d047e5df3c4f5a141ca2b2a3c2c9c1c9af46b2a60cd22aaba585`.
The tracked runner reproduced the archived base batch-16 summary and
rejected adding a checkpoint to replayed evidence.

Both frozen offline scorer environments reproduced their prior passing
full and held-out receipts byte for byte. The base receipt hashes to
`84a87dcf08a7ca8e9498aab4cc198fa3e53bca341637d8124d83ea5e1e8e7eda`
and the guided receipt to
`915cae3bf2351db1f90bd290ebbe79353a4b9bdb91a67cb494d30791403f89bc`.

The real CUDA capacity runs combined `deferred-head` and
`materialize-after=5` at batch 32 base and batch 16 guided. Each run used
five warmups and 30 trials. All codes and IDs match the corresponding
frozen rows. These unpaired runs check outputs at maximum batch sizes.
The paired Metal measurements are pending. The SHA-256 values for
`capacity-0-32.json` and `capacity-1-16.json` are
`94a3bf4de72f0c580605d751b2596cc352f88b88b186ff4aff1abc3151e66a45` and
`257798f5c016fa590dd32f26a384398998cf5e4af363cf2925181628a8ed9d45`.

The first full check and an isolated retry failed the existing process
lifecycle fixture while the GPU corpus replay was active. The unchanged
`tools/check.sh --cuda` rerun after the replay and capacity checks passed,
including all 528 Linux report tests and 18 real CUDA operator cases. The
failed logs are preserved alongside the passing `full-check-serial.log`,
which hashes to `6925e29fb8eda7f36a2979d3c0c0cf5e92eecca29d0b622d30d99d40c79ea4d8`.

## Locating the first base code difference

The retained Metal batch starting at row 632 has 13 differing intermediate
codes and exact raw token IDs. Its first difference is at row 632,
zero-based level 4, position 17. The pinned Python CUDA batch and the
native CUDA probe both select code 2602 where the retained Metal batch
selects 270. The pinned Python capture verifies all four rows against
the frozen corpus. The native probe checks that holding an input and
logits for tracing produces the same full codes and IDs as ordinary
generation.

| Projection input and method | Score for 2602 | Score for 270 | Winner margin |
| --- | ---: | ---: | ---: |
| Pinned Python CUDA FP32 | 3.3386700153 | 3.3386690617 | 0.000000954 |
| Native Rust CUDA FP32 | 3.3386716843 | 3.3386702538 | 0.000001431 |
| Pinned Python input, common FP64 projection | 3.3386706206 | 3.3386692630 | 0.000001358 |
| Native Rust input, common FP64 projection | 3.3386721419 | 3.3386711681 | 0.000000974 |

Code 922 is a distant third in both CUDA captures. Across the selected
input vector the largest absolute difference is `5.72e-6`. Across its
4,096 projected scores it is `2.10e-5`. Both CUDA paths still pick the
same code, and the same FP64 projection preserves the winner on both
captured inputs. The remaining question is whether projecting the
Metal input in FP64 also picks 2602. If it does, measure the native
projection on the same input shape. If it picks 270, compare the
preceding reconstruction and transformer activations.

Generate the pinned base trace on CUDA with the original four-row
fixture. The `--base-head` arguments select the batch start and the
zero-based level.

```sh
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  .reference/venv/bin/python tools/reference/wmt14_cfg_row_trace.py \
  --upstream .reference/upstream-ncm --source artifacts/upstream/wmt14-phase4 \
  --prefixes reports/phase4-wmt14-tokenizer-reference-v1.json \
  --full reports/phase4-wmt14-base-cuda-full-v1/python.json \
  --base-head 632 4 --output reports/wmt14-base632-reference
```

To repeat the native CUDA capture with cuDNN from the pinned reference
environment, link through the existing unversioned cuDNN library path.
The final three probe arguments select the absolute row, level, and
position. The probe records the full projection input and the 4,096
scores at that position.

```sh
setpriv --inh-caps=-all env CUDA_COMPUTE_CAP=86 RUSTC_WRAPPER= \
  LIBRARY_PATH="$PWD/.reference/phase4-cudnn-probe-lib" \
  cargo build --locked --release -p ncm-cli --features cuda-cudnn,wmt14-experimental \
  --example wmt14_base_head_probe
setpriv --inh-caps=-all env \
  LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" \
  target/release/examples/wmt14_base_head_probe \
  artifacts/canonical/wmt14-inference-v1-final cuda \
  reports/phase4-wmt14-tokenizer-reference-v1.json 632 4 17 \
  > reports/wmt14-base632-cuda.json
```

On Apple Silicon, build the same probe with the pinned MLX installation.

```sh
export MLX_SYS_PREBUILT="$PWD/.reference/mlx-native"
RUSTFLAGS="-C link-arg=-Wl,-rpath,$MLX_SYS_PREBUILT/lib" \
  cargo build --locked --release -p ncm-cli --features mlx,wmt14-experimental \
  --example wmt14_base_head_probe
target/release/examples/wmt14_base_head_probe \
  artifacts/canonical/wmt14-inference-v1-final mlx \
  reports/phase4-wmt14-tokenizer-reference-v1.json 632 4 17 \
  > reports/wmt14-base632-mlx.json
```

Compare the two capture files in the pinned Linux reference environment.

```sh
OMP_NUM_THREADS=1 .reference/venv/bin/python tools/reference/compare_wmt14_head_site.py \
  reports/wmt14-base632-reference reports/wmt14-base632-mlx.json
```

The Linux evidence is under `reports/phase4-wmt14-base632-cuda-v1.O5g5Bs/`.
Both native CUDA runs produced the same JSON. The probe rejected an
out-of-range position and the offline comparison rejected a different
head level. The default guided reference command reproduced its archived
tensor payload byte for byte. The retained Metal capture confirms all
13 code differences and the same first divergent position. The new
Metal head capture is pending.

| Evidence | SHA-256 |
| --- | --- |
| Pinned CUDA trace | `4179dba1011cf3fad60a2b79f10d2405ab6b3606264be183c1a98d935c6bc14b` |
| Native CUDA site capture | `3ac0cd680fd54f6447edc95ca523cb23f27c232b24ddb8db0a419e54f9d3e9f7` |
| CUDA comparison | `1471b67ee4e72de6035757f6ec55da0d55fcdd0135a268dd691ac95b24b5da4d` |

Feature-specific Clippy passed for the CUDA and device-free builds. The
two focused fixture tests passed. Full `tools/check.sh --cuda` passed
all 528 Linux report tests and 18 real CUDA operator cases. The
full log hashes to `0693ae901c25d024ce56d81d312b12784f8176650bf426012b257f56575602f3`.

## Tiling the decoder vocabulary projection

The WMT14 sampling probes and corpus recorder accept `decoder-rows=N`
for 1 through 32 rows per tile. The decoder still builds its
hidden state for the full request. It then projects a contiguous
group of rows to the 151,680-entry vocabulary, uses the existing
first-maximum selector, and reads the selected IDs before starting
the next projection. This keeps the full vocabulary axis for every
token. The tiled API returns host codes and IDs. Logit and teacher
traces use the full-projection path.

Two rows require a 74.06 MiB FP32 score tensor, compared with
148.13 MiB at batch four and 1,185 MiB at batch 32. These sizes
describe individual projections. Backend allocations and the
additional tile readback barriers affect the measured peak and wall time.
The projection's row dimension changes, so frozen output checks are
required on each device.

Run the existing paired timing schedule on Apple Silicon to measure
guided batch four and both large base batches with two-row tiles.

```sh
.reference/mlx-venv/bin/python tools/bench/run_phase4_wmt14_mlx_bench.py \
  --cell 1:4 --cell 0:16 --cell 0:32 --decoder-rows 2 \
  --output reports/wmt14-mlx-tiled-decoder-two
```

Build the probe and prepare its inputs using the [Metal timing
procedure](#repeating-the-metal-timing-comparison). Use
`--materialize-after 5 --with-control` in the same command to screen
the earlier reconstruction checkpoint with two-row tiling fixed
across both Rust configurations. The runner rejects
`--decoder-rows` with `--lifetime retain` because the tiled API
returns codes and IDs.

The recorder supports the same decoder setting for a full comparison
with the selected runtime build. The padded retry repeats the same
input rows on a head rejection and trims the recovered output to the
original batch. It uses the selected tile size for both attempts.

```sh
target/release/examples/wmt14_corpus_capture ARTIFACT mlx-retry-padded PREFIXES BASE.jsonl 0 0 3003 decoder-rows=2
target/release/examples/wmt14_corpus_capture ARTIFACT mlx-retry-padded PREFIXES CFG.jsonl 1 0 3003 decoder-rows=2
```

### Local CUDA validation

The two-row projection reproduced all 3,003 base and 3,003 guided
rows, with zero code or raw ID differences, retries, or rejections.
Both JSONL files are byte-identical to the
[earlier CUDA captures](cuda-fast-argmax-plan-v1.md#full-cuda-replay-with-early-logit-release).
The resulting comparison SHA-256 values are
`57431f5d21af3e4b91066a78eeec4e7ad933907b23785fbe86033ec3ba1ca409`
for base and
`6343ea547f36d047e5df3c4f5a141ca2b2a3c2c9c1c9af46b2a60cd22aaba585`
for guided.

The six targeted ranges per mode reproduced 46 rows, including both
original Metal rejection ranges and each three-row tail. The recorder
rejected a zero-sized tile before creating a capture. The focused
retry test verified the eight-row input expansion and confirmed a
decoder error propagates directly. The runner's option and replay
tests passed, and the retained Metal base batch-16 summary reproduced
byte for byte. The full-logit teacher trace reproduced all 26
tensors from the earlier native capture and passed against the pinned
teacher oracle. Its SHA-256 is
`cff57cecc1ba1e486fdad62ed1050754266cdfff126e94111063d3dd9db718ee`.

The Linux receipts are in
`reports/phase4-wmt14-tiled-decoder-cuda-v1.p60hTp/`. The CUDA recorder
and capacity probe passed feature-specific Clippy and release builds.

Both offline scorer environments reproduced their prior passing
full and held-out receipts byte for byte. The base and guided
receipts hash to
`84a87dcf08a7ca8e9498aab4cc198fa3e53bca341637d8124d83ea5e1e8e7eda`
and `915cae3bf2351db1f90bd290ebbe79353a4b9bdb91a67cb494d30791403f89bc`.

The CUDA capacity probe used `deferred-head materialize-after=5
decoder-rows=2` at batch 32 base and batch 16 guided, with five
warmups and 30 measured trials per process. All frozen codes and
IDs match at both maximum batches. The receipts
`capacity-0-32.json` and `capacity-1-16.json` hash to
`c09dc8c69d1953ae693049a7a33f407c154bf57d4a39bc359c004d6f4e5d8f2c`
and `d295c1e054eeee5c06e17aaa11c766ec9f91b38f90088461267ee086d5c11a57`.
These are correctness controls. Metal allocator peak and latency for
the tiled decoder still need paired measurements.

The first full check encountered the existing process-lifecycle
fixture failure during the GPU corpus capture. After the WMT14
device work finished, the serial `tools/check.sh --cuda` pass
completed all 530 Linux report tests and 18 CUDA operator cases.
Both logs are retained. The passing log hashes to
`d0e76c78373d7789f298da9a2b652a0f88e53a393ec9bb9cbe81bf0520b99203`.
