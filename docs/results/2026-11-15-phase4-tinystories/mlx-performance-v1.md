# TinyStories Metal performance and memory comparison

On an Apple M1 Ultra with 64 GiB unified memory, three alternating fresh processes per runtime and batch each ran five warmups and 30 generation trials. Timing includes generated code and final-ID host readback and excludes model loading. Python uses a separately implemented MLX adapter; its 16-row fixed-code development trace and the native Metal trace are compared in the [development observations](mlx-development-v1.md), with cross-runtime floating differences still open. Rust and Python use different categorical random streams. A later independent quality retest is required; this report does not assert quality or complete Apple acceptance.

| Batch | Python median ms | Rust median ms | Python pooled p95 ms | Rust pooled p95 ms | Python peak RSS MiB | Rust peak RSS MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 107.638 | 93.799 | 108.008 | 94.190 | 1046.42 | 1024.73 |
| 4 | 170.909 | 152.014 | 172.255 | 153.604 | 1046.36 | 1025.45 |
| 32 | 658.031 | 574.259 | 661.817 | 577.068 | 1046.69 | 1025.48 |
| 128 | 2181.020 | 2008.945 | 6704.634 | 2055.359 | 1045.95 | 1025.69 |

Median latency is the median of three per-process medians; p95 is nearest rank across 90 pooled trials. RSS is the median process high-water mark including startup and model loading. Rust was faster and had lower process peak RSS in all four measured cells. The RSS advantage is only about 20–22 MiB, so it needs the activity caveat below. Nine trials in Python's third batch-128 process exceeded four seconds, while the other two processes had no trials that slow; the pooled p95 therefore reflects a substantial outlier cluster. The capture does not establish its cause.

The following MLX counters are medians of the three immediate post-trial process observations, in MiB. Peak is MLX's process-global active-allocation high water, active is live MLX allocation, and cache is allocator-held reusable memory; these are not process RSS or total physical memory and need not add to the peak. Immediate cache/active snapshots do not establish later callback or cleanup behavior.

| Batch | Python peak / active / cache | Rust peak / active / cache |
| ---: | ---: | ---: |
| 1 | 608.70 / 479.29 / 580.48 | 606.98 / 490.08 / 583.81 |
| 4 | 975.69 / 479.29 / 2401.61 | 980.63 / 520.95 / 2435.30 |
| 32 | 2528.94 / 479.29 / 15809.14 | 2438.42 / 477.63 / 15047.67 |
| 128 | 2539.28 / 479.29 / 24596.45 | 2427.42 / 477.63 / 22148.48 |

Rust's batch-4 MLX peak exceeds Python's by 4.94 MiB, and Rust's immediate active and cache are also higher at batches 1 and 4. These are **visible memory-gate failures**, not a pass inferred from RSS. Both runtimes retained very large caches at the larger batches. A separate process-local cache-limit experiment is pending.

The run recorded 348 CPU activity samples at approximately two-second intervals, including background processes such as `herdr`, `WallpaperAerialsExtension` and iTerm2; GPU activity was not measured and the machine was not exclusively reserved. No clock, power or system settings were changed. The manifest records source, executable, artifact and native MLX library identities before and after; they remained stable. The ignored local capture is `reports/phase4-tinystories-mlx-bench-v1/`: `manifest.json` SHA-256 `6b408fb2f860b9819e7239d140001fea876779829614351956cef65a569e075a`, `summary.json` SHA-256 `14efbd89fe2bc1cbf739c7010b8a82a2654a3972d306285b1e0b8be9e4a37acf`. The trace qualification identity, full commands, per-process data and activity are retained with the capture. Reproduce using `tools/bench/run_phase4_tinystories_mlx_bench.py` with the recorded qualified trace and its SHA-256 on the same configured Mac.

## Exploratory cache limits

A separate process-local experiment set the native MLX cache limit before loading the model; it changed no system-wide setting. One fresh Rust process per cell ran the same five warmups and 30 generation/readback trials, with CPU snapshots before and after rather than continuous monitoring. At batch 4, 512 MiB and 2048 MiB limits produced 143.336 and 146.714 ms medians, with immediate cache 500.40 and 2018.21 MiB. The corresponding immediate peaks were 980.32 and 980.66 MiB, both still above Python's 975.69 MiB baseline. These are not paired or repeated controls.

At batch 32, the same limits reduced immediate Rust cache from the default-run median of 15047.67 MiB to 516.05 and 2050.61 MiB, but increased latency from 574.259 ms to 665.498 and 663.260 ms respectively. Both limited results were slower than the default Python MLX median of 658.031 ms. Neither cap is adopted as a general runtime default. Raw process output, GNU-style `/usr/bin/time -l` records and activity snapshots are retained under ignored `reports/phase4-tinystories-mlx-cache-dev-v1/`; the cap results do not constitute a new acceptance run.

Further single-process diagnostics used the same timing scope, with before/after CPU activity snapshots. At batch 32, 4 GiB and 8 GiB cache limits gave medians of 612.913 and 596.394 ms, with immediate cache of 4098.81 and 8196.77 MiB respectively. At batch 128, an 8 GiB limit gave a 2101.184 ms median and 8214.05 MiB immediate cache, compared with the uncapped three-process Rust medians of 2008.945 ms and 22148.48 MiB. The limited results were faster than the prior default Python medians, but the Rust-versus-Rust latency regression and single-process/activity limitations remain. No process-global default is changed by these diagnostic examples.

In a separate default-cache run, dropping unused decoder logits before readback changed the batch-1 immediate Rust active snapshot from the prior three-process median of 490.08 MiB to 477.63 MiB, while its peak was 607.37 MiB. At batch 4, peak and active remained 980.63 and 520.95 MiB, so this did not resolve the batch-4 peak-memory failure. The cache-limit and output-lifetime results remain exploratory and do not replace the paired results. Raw results and before/after activity snapshots are in `reports/phase4-tinystories-mlx-memory-dev-v2/`; the 8 GiB batch-32 run is in `reports/phase4-tinystories-mlx-cache-dev-v1/`.

A later safety review found that an invalid sampling sentinel could reach a subsequent MLX embedding before the final error check. The above measurements predate the resulting sentinel-safe-embedding candidate and must be revalidated on that candidate before they support a release decision.

## Safe-embedding and Q/K/V-view candidate

With sentinel-safe embeddings and separate Metal Q/K/V views, a single default-cache batch-4 run measured 152.751 ms, 971.52 MiB MLX peak, 477.63 MiB active and 2439.94 MiB cache. The earlier Python medians were 170.909 ms, 975.69 MiB peak, 479.29 MiB active and 2401.61 MiB cache. The peak and active directions improved in this exploratory run; cache was still higher. This does not establish repeated acceptance.

One process per cell then tested fixed process-local caps of 512 and 2304 MiB at all batches. The 512 MiB cap gave Rust medians of 94.21, 143.01, 659.07 and 2061.84 ms at batches 1, 4, 32 and 128; corresponding MLX peaks were 607.4, 972.6, 2584.6 and 2421.5 MiB. At batch 32, both latency and peak exceeded the earlier Python medians of 658.031 ms and 2528.94 MiB. The 2304 MiB cap gave 94.05, 153.50, 651.96 and 2054.97 ms, but batch-32 peak remained 2584.4 MiB and immediate batch-1 cache was 583.5 MiB versus Python's 580.48 MiB. Immediate active snapshots also varied; no cap or process-global policy was adopted. Before/after CPU activity snapshots and raw files are in ignored `reports/phase4-tinystories-mlx-safe-cache-v1/`.

The previously measured packed path had lower batch-32 peak than the separate-view candidate, while separate views improved batch 4. A batch-sensitive Metal layout dispatch is prepared as an unqualified candidate: views below batch 16 and the original packed path at larger batches. Its threshold is a measured exploratory choice, not a change to the model algorithm. It still needs real-Metal correctness, quality-equivalence and repeated performance/memory checks.
