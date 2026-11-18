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

### Allocator trace for slow trials

The retained batch-16 and batch-32 `time -l` receipts contain
whole-process system CPU time and peak footprint. Rechecking
those receipts and their pinned output comparisons shows two
different batch-16 tail patterns. Run 2 slows after its first
ten trials. Run 3 starts slow and partly recovers.

| Base batch | Rust process | First five median (ms) | Last five median (ms) | System CPU (s) | Trials above Python p95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 1 | 474.2 | 472.6 | 1.15 | 0 |
| 16 | 2 | 473.9 | 4,515.5 | 4.99 | 20 |
| 16 | 3 | 1,389.3 | 653.0 | 6.00 | 21 |
| 32 | 1 | 853.8 | 831.4 | 1.58 | 0 |
| 32 | 2 | 1,139.2 | 1,585.5 | 12.74 | 0 |
| 32 | 3 | 1,773.5 | 1,588.9 | 22.90 | 16 |

The peak footprint spans only 1.34 MiB across the three batch-16
processes and 3.00 MiB across batch 32. Ending MLX cache spans
0.22 MiB at batch 16 and is identical across the batch-32 runs.
Each process records 11 page faults and zero swaps. These
whole-process counters describe resource use across startup and
generation. The retained receipts lack time-resolved allocator
samples, so the cause of the latency tail remains open.

For a new Metal diagnostic, add `--allocator-trace` to the tracked
runner. It records Rust active bytes, process peak bytes, and cache
bytes before the first measured request, between requests after
the previous Rust outputs leave scope, and after the final
request. Samples sit outside the timed region. The summary retains
the matching trial start timestamps, and each `.meta.json` records
whole-process system CPU time. The allocator diagnostics can
perturb work between trials. Keep the paired qualification
measurements on the ordinary schedule.

```sh
.reference/mlx-venv/bin/python tools/bench/run_phase4_wmt14_mlx_bench.py \
  --cell 0:16 --allocator-trace --output reports/wmt14-mlx-tail-allocator-default
```

If the tail recurs, compare a fresh run with the same command and
`--cache-mib 2048` in a new output directory. Compare the trial
sequence with the 31 recorded allocator boundaries and the
whole-process system CPU time. Both runs must match the frozen
outputs. Linux fixtures verify launch, replay and malformed-counter
rejection. All six archived timing summaries replay byte for byte.
The old process observations and raw receipt hashes are in
`reports/phase4-wmt14-metal-tail-replay-v1.pbyM04/`. The SHA-256 of
`tail-observations.json` is
`d0c770814d14b7a3af80c92676e1c81878e6f6256c4000a87198b52e9cdb5df7`.
The full Linux suite passed all 531 report tests. Its log hashes
to `bd0eb60fe10c271f43c85ddad158dceb13c23ec7e9c7cfb49524f9e52904e2c1`. A live trace requires Apple Silicon.

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

### Scoring a complete batchwise capture

Replay the complete base and guided JSONL files on Linux with the
pinned tokenizer environment `.reference/tokenizers-venv` and both
[offline scorer environments](cuda-development-quality-v1.md)
`.reference/wmt14-scorer-venv` and `.reference/wmt14-scorer-replay-venv`.
Choose a fresh output directory for each mode.

```sh
python3 tools/reference/replay_wmt14_corpus_quality.py BASE.jsonl \
  reports/wmt14-metal-base-quality --cfg 0 --runtime metal
python3 tools/reference/replay_wmt14_corpus_quality.py GUIDED.jsonl \
  reports/wmt14-metal-guided-quality --cfg 1 --runtime metal
```

The command uses the frozen CUDA output and decoded-text references in
`reports/`. Use `--reference` and `--reference-text` if those files are
stored elsewhere. The same hashes apply. Use `--runtime cuda` for a CUDA
recorder capture.

Keep `batch-comparison.json` with the scorer receipts. It records
rejections, retries and all code and raw-ID differences. A rejected
batch stops the replay before decoding and preserves that receipt.
For a complete capture, the command verifies both scorers' full and
held-out results and their byte equality. A passing replay is a
frozen quality result. Intermediate code differences remain visible
for numerical review, and device performance has its own gate.

The tracked command reproduced the archived padded Metal captures
and combined CUDA captures in both modes. For each of the four
captures the normalized JSON, decoded text, comparison and both
scorer receipts are byte-identical to the corresponding earlier
artifacts. The Metal base receipt retains 13 differing codes at
row 632 and no raw-ID differences. Replaying the unpadded native
base capture retains its rejected batch at 1376 and exits before
decoding. The evidence is under
`reports/phase4-wmt14-scoring-replay-v1.wS1Msf/`.
The focused replay, tokenizer and scorer tests passed. The full Linux
suite passed all 531 report tests and its log hashes to
`6595c6274db7eec83f01275a8619c19766bee0796674275e34718d197c02098c`.

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

On Apple Silicon, compare two-row tiles with a full-projection
Rust control at guided batch four and both large base batches.
Both Rust configurations share one Python baseline, and the runner
rotates their process order through three repetitions per cell.

```sh
.reference/mlx-venv/bin/python tools/bench/run_phase4_wmt14_mlx_bench.py \
  --cell 1:4 --cell 0:16 --cell 0:32 --decoder-rows 2 --control decoder \
  --output reports/wmt14-mlx-tiled-decoder-two
```

Build the probe and prepare its inputs using the [Metal timing
procedure](#repeating-the-metal-timing-comparison). An optional
`--materialize-after 5` applies the checkpoint to both Rust variants.
To measure the checkpoint with two-row tiling held fixed, select
`--control checkpoint --decoder-rows 2 --materialize-after 5`.
The selected treatment is recorded in the manifest and must
be the only Rust setting that differs. Linux fixture tests
verify live command construction and replay for both control
modes. The runner accepts historical checkpoint receipts and
checks archived two-configuration runs. For tiles, select the
default output lifetime or `--lifetime release`.

The same nine-launch screen can compare two tile widths directly.
The manifest binds both widths, and replay checks both arms'
settings and exact outputs. For base batch 32, compare eight-row
tiles with a two-row control.

```sh
.reference/mlx-venv/bin/python tools/bench/run_phase4_wmt14_mlx_bench.py \
  --cell 0:32 --control decoder --decoder-rows 8 --control-decoder-rows 2 \
  --output reports/wmt14-mlx-tile-width
```

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

### Paired CUDA timing control

On the RTX 3060 Ti with driver 620.70.06, the same release binary
ran the full projection and two-row tiles at guided batch four
and base batch 32. Both used deferred head validation. Each
configuration used three fresh processes, with five warmups and
30 synchronized trials per process. The order alternated across
pairs, starting with the tiled run for guided and the full run
for base. Every receipt matched the exact frozen codes and raw
IDs. Timing includes host readback and excludes model load.

| Mode and batch | Full projection median / p95 (ms) | Two-row tiles median / p95 (ms) | Tiled change median / p95 |
| --- | ---: | ---: | ---: |
| Guided 4 | 309.86 / 313.30 | 312.45 / 316.08 | +0.84% / +0.89% |
| Base 32 | 1,210.55 / 1,213.50 | 1,216.15 / 1,219.52 | +0.46% / +0.50% |

Calculate the median and nearest-rank p95 over the 90 measured
trials per configuration. The tiled median was higher in each
of the three process pairs in both cells. This comparison
measures the scheduling cost on CUDA. Metal latency and native
allocator peak still need device measurements.

Build the CUDA timing example and use the pinned reference
environment's cuDNN library. These commands use the same
library paths as the native CUDA head probe above.

```sh
setpriv --inh-caps=-all env CUDA_COMPUTE_CAP=86 RUSTC_WRAPPER= \
  LIBRARY_PATH="$PWD/.reference/phase4-cudnn-probe-lib" \
  cargo build --locked --release -p ncm-backend-cuda \
  --features cuda-cudnn,wmt14-experimental --example wmt14_fast_argmax_perf
wmt_cudnn="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib"
wmt_artifact=artifacts/canonical/wmt14-inference-v1-final
wmt_prefixes=reports/phase4-wmt14-tokenizer-reference-v1.json
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$wmt_cudnn" \
  target/release/examples/wmt14_fast_argmax_perf "$wmt_artifact" "$wmt_prefixes" 32 0 deferred-head
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$wmt_cudnn" \
  target/release/examples/wmt14_fast_argmax_perf "$wmt_artifact" "$wmt_prefixes" 32 0 deferred-head decoder-rows=2
```

Use arguments `4 1` for the guided cell. Preserve each process's
JSON, compare its outputs with the matching frozen reference,
and verify that the reports bind the selected tile size.
Raw results, input hashes, process order, and the summarizer
are under `reports/phase4-wmt14-tile-paired-cuda-v1.LlZKIe/`.
The complete `summary.json` SHA-256 is
`58a75d50494a05154b9eb66f0b04a7c046e7c04f6b02a14f1d71fefbd6ae598f`.

The six archived Metal timing cells reproduced their prior summaries
byte for byte. The Linux host suite passed all 530 report tests.
Its `full-check.log` hashes to `2bc376362f2cd0cc6ffdc2995a6107e48533cac78c936ab7d0fc3a2a840ac728`.

### CUDA async-pool high-water measurements

The CUDA memory probe measured the full projection and two-row tiles
at guided batch four, base batches 16 and 32, and guided batch 16.
The three large-batch comparisons also include eight-row tiles.
Every arm used deferred head validation, three fresh processes,
three warmups and one measured request per process. The launch order
rotated across repetitions. All 33 receipts matched the frozen codes
and raw IDs, and each arm's used peak repeated exactly across its
three runs.

| Mode and batch | Full used / reserved peak (MiB) | Two-row used / reserved peak (MiB) | Eight-row used / reserved peak (MiB) |
| --- | ---: | ---: | ---: |
| Guided 4 | 1,162.23 / 1,216 | 1,104.51 / 1,152 | — |
| Base 16 | 1,608.49 / 1,824 | 1,196.05 / 1,248 | 1,312.24 / 1,536 |
| Base 32 | 2,204.01 / 2,624 | 1,379.14 / 1,472 | 1,379.14 / 1,472 |
| Guided 16 | 1,610.01 / 1,824 | 1,379.14 / 1,440 | 1,379.14 / 1,536 |

The figures are medians of process high-water counters for CUDA's
default async pool. They include model load and warmups. After
synchronized output release, live usage is 1,012.97 MiB in all arms.
At base batch 32, both tile sizes lower peak used memory by
824.87 MiB and peak reserved memory by 1,152 MiB. Eight-row tiling
issues four projections there. Two-row tiling issues 16. At
base batch 16 and guided batch 16, two-row tiling has the lower
reserved peak. Use the same Metal runner with `--cell 0:32
--control decoder --decoder-rows 8` to screen eight-row tiles.
The [three-width CUDA comparison](#cuda-tile-width-latency) measures eight-row latency. Metal peak and latency still need device measurements.

Build and run the allocator probe from the repository root using the
same pinned inputs and cuDNN runtime as the timing probe above.

```sh
setpriv --inh-caps=-all env CUDA_COMPUTE_CAP=86 RUSTC_WRAPPER= \
  LIBRARY_PATH="$PWD/.reference/phase4-cudnn-probe-lib" \
  cargo build --locked --release -p ncm-backend-cuda \
  --features cuda-cudnn,wmt14-experimental --example wmt14_fast_argmax_memory
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$wmt_cudnn" \
  target/release/examples/wmt14_fast_argmax_memory "$wmt_artifact" "$wmt_prefixes" 32 0 deferred-head
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$wmt_cudnn" \
  target/release/examples/wmt14_fast_argmax_memory "$wmt_artifact" "$wmt_prefixes" 32 0 deferred-head decoder-rows=8
```

Read `used_peak` and `reserved_peak` under
`cuda_pool_after_output_drop` for both arms. The full projection
also reports counters while holding its device output. Both the
original invocation without options and the new option validation
were exercised. Feature-specific Clippy and the CUDA release build
passed.

The raw results, watched input hashes, exact process order, shell
runner and summarizer are under
`reports/phase4-wmt14-tile-cuda-pool-v1.mHFlWE/`. The summary SHA-256
is `3b9d1d05ff4df8aaf99063ac855b2dcc3e51e0071ea3e87e93da6eaac54e396e`.
The binary SHA-256 is
`65d419822342c2939a04996754382bf1abb0d3b276e9b3eddb1588ab7e49cd6c`.

The Linux host suite passed all 530 report tests. The full check log
hashes to `79592dba2ae63249431e1a87a58352552dc27790b3b1ac7c8d024319eaeb88ad`.

### CUDA tile-width latency

At base batch 32, the same release CUDA probe compared the full
projection, two-row tiles and eight-row tiles. Every arm used deferred
head validation, five warmups and 30 measured requests in each of three
fresh processes. The order rotated across rounds so every arm ran
first, second and third once. All nine receipts match the exact
frozen codes and raw IDs. Per-request timing includes host readback
and excludes model load.

| Projection | Pooled median (ms) | Pooled p95 (ms) | Process medians (ms) |
| --- | ---: | ---: | --- |
| Full | 1,211.34 | 1,213.80 | 1,187.07 / 1,211.34 / 1,212.30 |
| Two rows | 1,215.00 | 1,220.36 | 1,205.10 / 1,215.00 / 1,218.45 |
| Eight rows | 1,214.86 | 1,218.70 | 1,210.96 / 1,215.20 / 1,216.44 |

Relative to the full projection, two rows added 0.30% at the
median and 0.54% at p95. Eight rows added 0.29% and 0.40%.
The eight-row median is 0.01% lower and p95 is 0.14% lower
than two-row tiling. The differences between tile sizes are small
relative to the spread in process medians. In the earlier allocator
screen both widths had identical peak used and reserved CUDA pool memory
at this batch size. A shared-baseline Metal screen can measure the
same alternatives under the native allocator.

Reproduce the full and two-row runs using the
[paired CUDA timing commands](#paired-cuda-timing-control).
Use `decoder-rows=8` for the third arm. Rotate each arm through all
three positions and pool 90 measured trials per arm.
The binary, input hashes, process order, shell runner, and
frozen-output verifier are under
`reports/phase4-wmt14-tile-width-cuda-v1.crnFJZ/`. The summary SHA-256
is `85a40874ec75ded8ca1dffb3e465652669d144d28ad3dc84930865857a9e5b55`.

The six archived Metal cells reproduced prior summaries byte for byte.
The full Linux suite passed all 530 report tests. Its log hashes
to `05ac6127c7bac95c5a2feded857953acc5746b4c5cf2f2b0ca47285fd13473c8`.

### Combined CUDA corpus replay

The recorder accepts `deferred-head` independently of padded retry.
Use it on CUDA to combine the deferred head check with the
level-five reconstruction checkpoint and two-row decoder tiles.
This exercises the three settings together across the original
four-row corpus grouping and the three-row tail.

Use the CUDA library and fixture variables from the
[paired timing commands](#paired-cuda-timing-control).

```sh
setpriv --inh-caps=-all env CUDA_COMPUTE_CAP=86 RUSTC_WRAPPER= \
  LIBRARY_PATH="$PWD/.reference/phase4-cudnn-probe-lib" \
  cargo build --locked --release -p ncm-cli \
  --features cuda-cudnn,wmt14-experimental --example wmt14_corpus_capture
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$wmt_cudnn" \
  target/release/examples/wmt14_corpus_capture "$wmt_artifact" cuda "$wmt_prefixes" \
  reports/wmt14-combined-base.jsonl 0 0 3003 deferred-head materialize-after=5 decoder-rows=2
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$wmt_cudnn" \
  target/release/examples/wmt14_corpus_capture "$wmt_artifact" cuda "$wmt_prefixes" \
  reports/wmt14-combined-guided.jsonl 1 0 3003 deferred-head materialize-after=5 decoder-rows=2
```

Replay each JSONL with the [frozen output verifier](#replaying-a-batchwise-capture)
and the corresponding pinned reference and scorer. On Metal,
`mlx-retry-padded` already uses deferred head checking, and the
recorder accepts the same checkpoint and tile settings for that path.

The combined CUDA replay matched all 3,003 base and 3,003 guided
rows exactly, with no retries or rejections. Both JSONL files
are byte-identical to the preceding two-row-only capture.
The frozen comparison SHA-256 values are
`57431f5d21af3e4b91066a78eeec4e7ad933907b23785fbe86033ec3ba1ca409`
and `6343ea547f36d047e5df3c4f5a141ca2b2a3c2c9c1c9af46b2a60cd22aaba585`.
Both offline scorer environments reproduced their previous
passing full and held-out receipts byte for byte. Base and
guided scorer hashes are
`84a87dcf08a7ca8e9498aab4cc198fa3e53bca341637d8124d83ea5e1e8e7eda`
and `915cae3bf2351db1f90bd290ebbe79353a4b9bdb91a67cb494d30791403f89bc`.

The original invocation also matched its three-row tail,
and a repeated `deferred-head` option failed before creating
a capture. The recorder hash, postflight manifest, verification
scripts, and receipts are under
`reports/phase4-wmt14-combined-cuda-replay-v1.sCukmN/`.
The combined configuration still needs its Metal corpus,
latency and allocator comparisons.

The build without device features exposed an inferred option type
that needs an explicit `usize`. With the annotation, Clippy passes
with and without CUDA and the rebuilt CUDA executable is byte-identical
to the executable used for the full replay. Both hash to
`4d95c5af29e3a903313c80cc7b49fc1bc58fdc43c3eaf51fccc9d9e2c4424f6c`.

The full Linux suite passed all 530 report tests. Its log hashes to
`7da1e9a810f8c688779995cc98910ce899c47028ab29f556f34e37c41f4667be`.

## Combined Metal verification

Mac Studio fast-forwarded to `0e99ce4` and built the four feature-enabled
release probes against the existing MLX libraries. The run used the same
pinned artifact, prefixes and CUDA output references as the Linux replay.
The `dotsetup.rmlx` service was resident during the run. The timing runner
does not reserve the GPU.

### Targeted checks and row 632

The fresh fixed-history Metal teacher trace passed all 26 tensor checks.
Its final-logit maximum absolute difference was `3.624e-5`, with no
greedy ID or forced code differences in the two-row control. The trace
tests a forced history. The generated corpus uses its own history.

The Metal recorder used deferred head validation, the level-five
checkpoint and two-row decoder tiles. Across the three targeted
four-row ranges, all raw IDs matched. Base range 1376 and guided
range 2656 matched all codes after their single padded retries.
Base row 632 retained its 13 intermediate code differences.

The first difference remains at level 4, position 17, with no prior
level-code differences. CUDA chooses 2602 over 270 and Metal chooses
270 over 2602. Re-evaluating the captured CUDA and Metal head inputs
using the same FP64 projection preserves each device's winner.
The CUDA margin is about `1.36e-6` and the Metal margin is about
`4.65e-6`. The maximum input difference at this position is
`1.45e-5`. This locates the disagreement in the inputs to the
head; higher-precision projection alone cannot align those two choices.

The raw Mac run is under `reports/phase4-wmt14-metal-resume-v1.6LwmUb/`.
The copied traces and Linux comparison receipts are under
`reports/phase4-wmt14-metal-resume-linux-v1.N51iAT/`.

| Receipt | SHA-256 |
| --- | --- |
| Fixed-history comparison | `155d4c19d6364e8d300d665d0735e0f96a1f019e59c2d15894e6326542816c70` |
| Row 632 head comparison | `10ae11e48fed63e5a02aa78e51426e7ceab24a773f0efeeaa51634299edc03f2` |
| Targeted base range 632 | `a3e31064c33f3ee44c00321f02e06314567d8a9c07cbc3432c802dbe757eb2d2` |
| Targeted base range 1376 | `cd95362d80228dfcfee8fe52682ac3888c2cd2f862dd9cd85d43ed4254297699` |
| Targeted guided range 2656 | `2aaf48eb51648a2afe966753bfda5dc77e4343b0bf410b31b957934677950c50` |

### Repeated timing with the checkpoint and tiles

The paired Metal runner uses three fresh processes per runtime and
cell, with five warmups and 30 synchronized generation-and-readback
trials per process. It excludes model load from latency and includes
it in RSS and allocator peaks. The Rust processes use the reusable
loader, deferred head validation, the level-five checkpoint, and
two-row decoder tiles. Python uses the independent MLX adaptation.
Both runtimes are checked against the frozen CUDA output. Each cell
has a separate manifest so a failed cell cannot suppress later runs.

| CFG | Batch | Median ms Rust / Python | p95 ms Rust / Python | Peak RSS MiB Rust / Python | MLX peak MiB Rust / Python |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 97.10 / 107.52 | 97.39 / 107.99 | 1561.05 / 1656.48 | 1162.73 / 1174.55 |
| 0 | 4 | 220.14 / 235.14 | 226.07 / 239.49 | 1561.11 / 1657.19 | 1621.24 / 1665.64 |
| 0 | 16 | 456.45 / 593.39 | 457.04 / 614.82 | 1561.23 / 1659.47 | 2599.13 / 3274.29 |
| 0 | 32 | 819.57 / 1043.99 | 820.12 / 8768.93 | 1561.25 / 1657.48 | 2835.02 / 3925.68 |
| 1 | 1 | 138.35 / 151.69 | 138.77 / 154.38 | 1561.11 / 1657.34 | 1315.58 / 1319.22 |
| 1 | 4 | 351.11 / 364.25 | 353.52 / 368.14 | 1561.09 / 1657.64 | 2153.81 / 2142.62 |
| 1 | 16 | 804.67 / 1025.07 | 805.16 / 1029.70 | 1561.09 / 1655.88 | 2864.28 / 3318.69 |

All 42 retained process outputs match both codes and raw IDs. The
seven summaries replay byte for byte on Linux. The four priority cells
run at `0e99ce4`. Base batches 1 and 4 and guided batch 1 run at
`f65fb6f` with identical runtime settings, leaving fusion at its
ordinary threshold. All seven cells have lower median latency,
p95, and process RSS. Guided batch four has a native allocator peak
11.19 MiB higher than Python. Six cells pass all four measures.

The three batch-32 Python process medians were 1038.55, 1784.55, and
1043.22 ms. In the second process, 19 trials exceeded 1.5 seconds.
All three batch-32 Rust process medians rounded to 820 ms. The pooled
p95 retains the Python tail.

| Summary | SHA-256 |
| --- | --- |
| Base batch 1 | `d8cc642b0ff4fe520c58ebd487766c98231db5154a4a8a4f2a3e0b29c1494ea5` |
| Base batch 4 | `1bd418bb901743144dc649037cf81c235fa6e212d253a2406b696534466bdab9` |
| Base batch 16 | `22599f1c3feae59fca032b274c9f7030db0e4f6b720a6e0388758feb46fd6e6f` |
| Base batch 32 | `33e9806f9ff95780787436d6da561283e94b12893706bdf06cd6b9d6965b7193` |
| Guided batch 1 | `5f49bc8176f3f2341469157fa5e2ea84a84f88d9d8824494d1fbff8417ab85d3` |
| Guided batch 4 | `a9327711f45f94d855eaa0156a09d0da07c7e5f0cf38ae25a342eb6276f572f9` |
| Guided batch 16 | `07fab97389afcec02fbebc0e09b5e0ae8732baab7dc2ade38bb35c78f42f9b22` |

Run a cell from the Mac checkout after building `wmt14_perf` and
preparing the inputs from the [timing procedure](#repeating-the-metal-timing-comparison).

```sh
.reference/mlx-venv/bin/python tools/bench/run_phase4_wmt14_mlx_bench.py \
  --cell 0:16 --materialize-after 5 --decoder-rows 2 \
  --output reports/wmt14-metal-combined-base16
```

### Full corpus and regression checks

The base and guided Metal captures each cover all 3,003 input rows
with no rejected batches. All 384,384 raw IDs match the frozen CUDA
reference. The only padded retries are base batch 1376 and guided
batch 2656. The guided capture matches every intermediate code.
The 13 intermediate differences in base row 632 are the only
code differences across both modes. Both offline environments
pass the full and held-out checks in each mode and reproduce the
prior Metal scorer receipts byte for byte. The decoded hypotheses
are identical to the reference, so all paired bootstrap differences
are zero.

| CFG | Scope | Cased BLEU | Lowercase BLEU | chrF | Length ratio |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | All 3,003 rows | 21.72022 | 22.61211 | 49.56486 | 0.93740 |
| 0 | Held-out 2,875 rows | 21.91867 | 22.82002 | 49.74774 | 0.93864 |
| 1 | All 3,003 rows | 22.94641 | 23.83582 | 51.51598 | 0.95867 |
| 1 | Held-out 2,875 rows | 23.14703 | 24.03098 | 51.69116 | 0.95972 |

| Evidence | SHA-256 |
| --- | --- |
| Base JSONL capture | `a2cc46d562b9939c036b6b4e9ca5d6826f650857aebba1f809aa4a8f77383cf7` |
| Base batch comparison | `ce5f7c78e0e999c051c5ce4f216ed6d2353ef1452838aba294603f581d6e8f95` |
| Base frozen score | `5598f2d2d1a484d077c875874b8105f004ad9af68149e801bdc58246f81ee62b` |
| Guided JSONL capture | `2c602391ff2d42ddf8214380d8f1a6fb7650a42e500215fe9deaa84895744040` |
| Guided batch comparison | `c7af66020d8916240bcef38667a25148762acfcef7a7eff8f698371eaf3894cc` |
| Guided frozen score | `915cae3bf2351db1f90bd290ebbe79353a4b9bdb91a67cb494d30791403f89bc` |
| Full Metal regression log | `87a8358ab4cdbfc20041fe64ca9e59f9c100e407108b624fc2ec765a84bc7d98` |

The full `tools/check.sh --mlx` run passes on Mac Studio, including
421 portable report tests and 18 real-device operator cases. It
started after the four timing cells completed.

On Mac, run the following for each `cfg` of 0 and 1, changing the output
filename for each capture. Run the [frozen scorer replay](#scoring-a-complete-batchwise-capture)
on both captures with `--runtime metal`.

```sh
target/release/examples/wmt14_corpus_capture artifacts/canonical/wmt14-inference-v1-final \
  mlx-retry-padded reports/phase4-wmt14-tokenizer-reference-v1.json \
  "reports/wmt14-metal-combined-$cfg.jsonl" "$cfg" 0 3003 materialize-after=5 decoder-rows=2
```

### Guided batch-four allocator screen

With deferred validation and two-row decoder tiles, the eleven short
guided batch-four runs covered no reconstruction checkpoint and each
of the ten supported checkpoint positions. Each fresh process used two
warmups and three measured generations. Each matched the pinned codes
and raw IDs and reported the same Metal allocator peak of 2,153.81 MiB.
The full guided corpus also used the GPU during this allocation
screen. Comparative latency requires its own paired run.

The raw results are in `guided-checkpoint-screen.k02N6I/` under the
Mac run directory. The copied summary SHA-256 is
`74627a1cc7cf55970085294310b34b5b86c53a77a9d7b89f70ef05cb123f5d05`.

At commit `f65fb6f`, the Rust timing probe accepts
`fused-rope-batch8`. The existing width-64 fused RMS normalization
and rotary kernel normally applies from 16 internal rows. The option
also uses it at eight internal rows, including guided batch four.

```sh
target/release/examples/wmt14_perf artifacts/canonical/wmt14-inference-v1-final \
  reports/phase4-wmt14-tokenizer-reference-v1.json 1 4 2 3 \
  reusable deferred-head drop-logits materialize-after=5 decoder-rows=2 fused-rope-batch8
```

The focused Metal build and Clippy passed. Each arm ran in a fresh
process on three four-row guided ranges. On ranges 0 and 128,
both arms match the frozen codes and raw IDs. The control rejects
range 2656 through the existing head guard. The fused arm returns
two code differences and one raw-ID difference there. Its allocator
peak is 2,329.41 MiB, up 175.60 MiB from the control. The probe
failed the output and memory checks, so the option was removed.
The short screen remains reproducible at `f65fb6f`.

The Mac receipts are in
`reports/phase4-wmt14-metal-fusion-v1.kfpWb6/`, including the
three completed small-cell timing runs. The copied receipts are
in `reports/phase4-wmt14-metal-fusion-linux-v1.CFxqOb/`.
The fusion screen summary SHA-256 is
`feabe7542aa8fa935216b8f995968201bac77c029bc6fd4f4f9caa5512f84795`.

The nine process-local allocator and output-lifetime controls
all matched pinned codes and IDs. The screen covered one-row,
two-row and full vocabulary projection, the release barrier,
retained logits, and four cache caps between 0 and 2,048 MiB.
Every control reported a 2,153.81 MiB Metal peak. End-of-trial
active memory varied with output lifetime. The short screen's
copied summary hashes to
`4f4cf675373d64af225555819a7d809613fdb672ad753fc563dc3a09520b344f`
under `reports/phase4-wmt14-allocator-policy-linux-v1.HpFRR4/`.

A separate source experiment reconstructed the shared guided
codes once, then duplicated the resized latent before the two
transformer branches. In CUDA controls it matched ranges 0, 128
and the three-row tail. It changed two codes and one raw ID in
row 2658. The ordinary CUDA path matches that range exactly.
This ended the candidate before the full corpus and Metal replay.
The release build and focused Clippy passed. The rejected diff and
the paired comparison receipts are retained under
`reports/phase4-wmt14-shared-reconstruction-cuda-v1.rlF4Da/`.

| Evidence | SHA-256 |
| --- | --- |
| Rejected experiment diff | `b57fab72927e498e104d4fb0faf46321c8a1f78ad69cb87c9e49219c9a47904d` |
| Ordinary CUDA range 2656 | `15f85ccadc8cf8d70e808c83c71498dc6a140cfe7f2f96f4eb9a4f953bbdc3d7` |
| Shared-reconstruction range 2656 | `f82b86f7b0de844791e5c84e4ab643ecb7b0c848b78b890fdd61bdeb349e5c2a` |

The next diagnostic uses `last-transformer-checkpoint=N` to
evaluate the hidden state after block N of the final generator level.
Values 1 through 11 are valid. The probe and recorder can combine
this with the earlier level checkpoint and tiled decoder.

```sh
target/release/examples/wmt14_perf artifacts/canonical/wmt14-inference-v1-final \
  reports/phase4-wmt14-tokenizer-reference-v1.json 1 4 2 3 \
  reusable deferred-head drop-logits materialize-after=5 decoder-rows=2 last-transformer-checkpoint=6
```

In Linux CUDA controls, checkpoints after blocks 1, 6 and 11
all match pinned codes and raw IDs on ranges 0 and 2656.
Feature-enabled release build and Clippy passed, as did the WMT14
model tests. The CUDA receipts are under
`reports/phase4-wmt14-transformer-checkpoint-cuda-v1.X4e7px/`.

All eleven Metal block positions match pinned codes and raw IDs in
the four-row timing fixture. Blocks 3 through 9 report the same
2,096.74 MiB peak, compared with 2,153.81 MiB in the control and
2,142.62 MiB in the repeated adapted Python MLX cell. Block 3 had
the lowest short-run median among the exact settings that beat the
Python allocator peak. With block 3, the padded retry and the
three-row tail also match the pinned codes and raw IDs.

The focused real-device Metal build and Clippy passed. Linux
host checks passed all 531 report tests under the benchmark's
unprivileged identity. The Metal sweep summary hashes to
`0402318c827a28af60c72f32151dd4145185f26987da5d9a05e9d4223b341e3f`
under `reports/phase4-wmt14-transformer-checkpoint-linux-v1.tkw2aJ/`.
The Linux host log SHA-256 is
`35c06ae9baf89a9f8711272c859c4a1eb98a16b62eb3f7b1bddd91ddb559d172`.

Repeat the candidate using the tracked paired runner.

```sh
.reference/mlx-venv/bin/python tools/bench/run_phase4_wmt14_mlx_bench.py \
  --cell 1:4 --materialize-after 5 --decoder-rows 2 --last-transformer-checkpoint 3 \
  --output reports/wmt14-metal-guided-last-transformer-block3
```

### Repeated block-three comparison

Three fresh processes per runtime matched the pinned codes and
raw IDs. Each process ran five warmups and 30 synchronized measured
trials. The repeated guided batch-four block-three configuration
passes median latency, p95, process RSS, and native allocator peak.

| Runtime | Median (ms) | p95 (ms) | Peak RSS (MiB) | MLX peak (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Rust Metal | 353.82 | 356.54 | 1561.19 | 2096.74 |
| Adapted Python MLX | 364.55 | 368.53 | 1657.72 | 2142.62 |

Linux reproduced the summary byte for byte. Its SHA-256 is
`4ba93df48451c2598eb49583a59c57d55138bcc3c260ae0767b9171ceea5aa81`.
The six other cells in the [earlier paired comparison](#repeated-timing-with-the-checkpoint-and-tiles)
pass with the level-five checkpoint and two-row tiles alone. The
block-three setting is used specifically for guided batch four.

The full guided CUDA block-three capture matched every code and raw ID
across 3,003 rows. Both frozen scorers pass the full and held-out
checks. The batch comparison and scorer receipt match the earlier
guided CUDA run byte for byte. They are under
`reports/phase4-wmt14-transformer-checkpoint-cuda-v1.X4e7px/block3-quality/`
and hash to `6343ea547f36d047e5df3c4f5a141ca2b2a3c2c9c1c9af46b2a60cd22aaba585`
and `915cae3bf2351db1f90bd290ebbe79353a4b9bdb91a67cb494d30791403f89bc`.

The full guided Metal block-three capture is byte-identical to the
earlier combined Metal capture. All 3,003 rows match the frozen
codes and raw IDs. Batch 2656 uses the single padded retry. Both
frozen scorers pass the full and held-out checks, with identical
receipts. The raw Metal run is under
`reports/phase4-wmt14-block3-metal-v1.i7tImw/`. The copied capture,
comparison, timing and score receipts are in
`reports/phase4-wmt14-transformer-checkpoint-linux-v1.tkw2aJ/`.

### CLI and profile qualification

At `bbc0e9e`, the experimental Metal CLI uses the level-five
checkpoint and two-row decoder projections for scales zero and one.
At scale one with four prefixes it also uses the block-three
checkpoint. The bounded retry remains an explicit device choice.

The release CLI and focused Clippy passed on both devices. Linux
CUDA confirms all codes and raw IDs for 62 control rows. Metal
confirms all raw IDs for 92 control rows across all seven measured
shapes, both recovered ranges, the tails, and base row 632.
The base row retains its 13 recorded intermediate differences.
Both native Metal controls still report the head rejection.
The explicit retry recovers them and rejects a five-row request.
The omitted CFG scale matches explicit zero. The scale-two output
is byte-identical to the earlier Metal CLI.

The final regression suite passed 531 Linux report tests, 421
portable Mac report tests and 18 operator cases on each GPU.
The Mac service stayed resident. Performance comparisons exclude
startup. The final seven cells use the original timing and memory
settings with the [block-three result](#repeated-block-three-comparison)
for guided batch four. Rust has lower median and p95 generation
latency, peak process RSS, and native allocator peak in every
measured Metal cell. CUDA has its own
[paired performance evidence](cuda-fast-argmax-plan-v1.md).

The paired Metal processes also recorded immediate active and cached
allocator counters after their final trial. Values below are medians
of process observations and reflect the backend's output lifecycle
at that boundary. Rust has a higher immediate active counter and
a lower retained cache in every measured cell. In the short guided
batch-four control the release barrier brought immediate Rust
active memory to 1,008.14 MiB without changing the allocator peak.
That barrier control is separate from the paired measurements.

| CFG | Batch | Final active MiB Rust / Python | Final cache MiB Rust / Python |
| ---: | ---: | ---: | ---: |
| 0 | 1 | 1045.18 / 1009.16 | 1267.33 / 1314.51 |
| 0 | 4 | 1082.21 / 1009.16 | 5284.43 / 5387.12 |
| 0 | 16 | 1082.21 / 1009.16 | 13391.09 / 18818.95 |
| 0 | 32 | 1082.21 / 1009.16 | 18884.18 / 22918.68 |
| 1 | 1 | 1045.18 / 1009.16 | 2604.86 / 2659.37 |
| 1 | 4 | 1082.21 / 1009.16 | 9720.89 / 9793.60 |
| 1 | 16 | 1082.21 / 1009.16 | 19001.80 / 22405.21 |

| Receipt | SHA-256 |
| --- | --- |
| Metal CLI summary | `bd627a9720b50412f0efce1864f6b84a3624868c4b179d177f58a3d08c3dc799` |
| Final Metal regression log | `76830297f0ee05c2c9d4592d45875936b0a000f59133a13b5f282d7b6edebae7` |
| Final CUDA regression log | `c19b51b0d60d029b1cbbe7f93c5a43c849cc1b695e8c4ccc51f95863942f0073` |

The CLI harness and Metal receipts are under
`reports/phase4-wmt14-transformer-checkpoint-linux-v1.tkw2aJ/`.
The CUDA CLI controls and regression log are under
`reports/phase4-wmt14-cli-schedule-cuda-v1.WiRZGa/`.
The frozen top-1 WMT14 results support advancement to LM1B-small.
