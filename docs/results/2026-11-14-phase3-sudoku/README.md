# Phase 3 Sudoku — Linux and Apple evidence

**The scoped FP32 greedy Sudoku hardware checks passed on CUDA and MLX; the final publication check remains pending.** The implementation follows the pinned Python model semantics, uses eager level-by-level selection and no prefix cache. Both real-device operator checks and the pinned synthetic generator golden passed, followed by all three 2,000-board evaluations and fresh-process performance/memory comparisons. The Linux intermediate-code near tie, background activity and other limits are disclosed below. This is not a general text-model inference release.

The two-board synthetic trace matches Python in all 644 codes and 178 decoded IDs. Across its 12 level-logit tensors the maximum observed absolute difference is 4.244e-5. Original and canonical Python weights produced identical traces, and the code/ID hashes match the already published [Phase 1 oracle](../2026-11-10-sudoku-artifacts/raw/oracle.json).

The CLI's separate `reconstruct` path was checked on both devices against the two input rows, codes and decoded IDs from that pinned codec trace: both output JSON files match the Python-derived expected file byte-for-byte (SHA-256 `747b103358042e9b12c860f4f30f1be3350c890aa93ba4b8fc23f6dbf92ea9e6`). `inspect` returned identical checked manifests on both, and requesting an unavailable device on each build exited with an explicit no-fallback error. The input, expected values and device outputs are retained locally under `reports/phase3-cli-reconstruct-v1/`; this is a two-board CLI check in addition to the broader codec qualification, not a full reconstruction corpus measurement.

The initial batch-128, 2,000-board-per-checkpoint [quality report](linux-exploratory-quality-v1.json) compares the original Python reference with the Rust CUDA CLI. It uses newly pinned dataset revision `6580ba41a4b0a976dde876fbe84d25327617b0e6`, with file hashes and safe conversion in `tools/reference/sudoku_phase3_data.py`. Counts are reproduced using `tools/bench/phase3_sudoku_score.py`:

| Difficulty | Solved / 2,000 (both) | Cell accuracy (both) | Valid grids (both) | Clues preserved (both) | Malformed (both) | Code-disagreement boards | Output-ID disagreement boards |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Easy | 1,988 | 99.974% | 1,988 | 1,998 | 0 | 0 | 0 |
| Medium | 1,928 | 99.794% | 1,928 | 1,993 | 1 | 1 | 0 |
| Hard | 1,726 | 98.352% | 1,726 | 1,997 | 1 | 0 | 0 |

The subsequently optimized CUDA candidate completed the same 6,000 boards with **zero code or output-ID differences from the initial Rust run**, and thus has the same single Python code near tie and zero Python decoded-ID disagreements. Its [quality report](linux-optimized-quality-v1.json) retains the per-difficulty metrics and source-output hashes. This earlier candidate CLI binary SHA-256 was `c1326de7e2e9be4e11bae2e4970e20242989033ec53935b922a04a26171cf004`; its source-to-binary binding was not established. Raw optimized per-board outputs remain locally under `reports/phase3-linux-optimized-v1/`.

The later candidate, including shared-generator request serialization, was measured in [three fresh processes per runtime and batch](linux-final-timings-v1.json), each with five warmups and 30 synchronized trials. All corresponding code/ID hashes match. Medians below pool the 90 trials; whole-process RSS and sampled GPU process memory show the median of the three process peaks. The GPU samples were approximately one second apart and can miss short peaks.

| Batch | Rust / Python median latency (ms) | Rust / Python peak RSS (MiB) | Rust / Python sampled GPU process peak (MiB) |
| ---: | ---: | ---: | ---: |
| 1 | 25.72 / 33.89 | 613 / 1,716 | 356 / 456 |
| 4 | 63.23 / 68.04 | 600 / 1,706 | 388 / 540 |
| 32 | 419.45 / 447.28 | 598 / 1,760 | 676 / 1,470 |
| 128 | 1,644.83 / 1,744.12 | 597 / 1,725 | 1,700 / 5,096 |

At those median latencies, throughput in boards per second is Rust / Python:
38.88 / 29.50 (batch 1), 63.26 / 58.79 (batch 4), 76.29 / 71.54 (batch 32),
and 77.82 / 73.39 (batch 128). These are batch size divided by the median
latency, not a separate concurrent-throughput measurement.

The CUDA default-pool and PyTorch allocator peaks in the raw summary have different scope and reset points and are not directly comparable. Whole-process RSS includes startup and loading; latency includes generation and code/ID readback but excludes loading. Desktop GPU processes were present, and the 481 activity samples are retained locally under `reports/phase3-linux-final-timings-v1/`; this is a single-machine result, not an idle-machine guarantee. The exact measured files are preserved locally under `.reference/phase3-final-timings-source/`; `python3 tools/bench/phase3_bench_summary.py reports/phase3-linux-final-timings-v1 --source-root .reference/phase3-final-timings-source` reproduces the summary.

The later [source-bound full-quality run](linux-qualified-quality-v1.json) repeated all 6,000 boards at commit `cb0162c`: every final ID matched Python, and the same single medium code near tie remained. All watched source, checkpoint and dataset hashes were unchanged across the six runs. The CLI SHA-256 was `52e696598b4f882d34ab0be18b120e9131860530c7f828ccb2dbb058f935353b`; the local manifest SHA-256 is `3fa4abca7c28c92085c3a3ac57656f4283433499187d4f5a58545ff4bb1ba82f`. Raw results and exact commands remain under `reports/phase3-linux-qualified-v1/`; run `python3 tools/bench/run_phase3_linux_quality.py --output NEW_DIRECTORY` to repeat with the specified local assets.

## Apple Silicon / MLX

On an Apple M1 Ultra with 64 GiB unified memory and macOS 26.7.3, the real-Metal operator check passed all 18 cases and the pinned two-board generator golden passed. The [independent audit](mlx-quality-performance-audit-v1.json) recomputes the quality and performance summaries from retained process outputs. Across the same 6,000 boards, Metal matched the original CUDA Python reference in **all intermediate codes and final IDs**. Both produced the solved, cell, valid-grid, clue and malformed counts in the table above, except that Metal has zero code-disagreement boards on medium. The original CUDA Python output is the quality oracle; the separate Python MLX adaptation used for timing matched both codes and IDs on the first 128 easy boards before the benchmark.

| Batch | Rust / Python median latency (ms) | Rust / Python p95 latency (ms) | Rust / Python peak RSS (MiB) | Rust / Python MLX peak (MiB) | Rust / Python boards/s at median |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 54.15 / 82.16 | 54.55 / 82.50 | 380 / 403 | 262 / 270 | 18.47 / 12.17 |
| 4 | 94.99 / 127.36 | 96.00 / 127.89 | 380 / 404 | 549 / 577 | 42.11 / 31.41 |
| 32 | 398.74 / 494.57 | 404.69 / 500.98 | 383 / 404 | 2,188 / 2,227 | 80.25 / 64.70 |
| 128 | 1,542.03 / 1,780.06 | 1,612.40 / 4,853.58 | 382 / 401 | 2,729 / 2,910 | 83.01 / 71.91 |

Each runtime used three fresh processes, five warmups and 30 synchronized generation-plus-code/ID-readback trials per batch. Medians and p95 pool 90 trials; RSS and MLX peak columns show the median of three process peaks. Model loading is outside the timed scope but included in process peak RSS. The Python MLX implementation is an offline adaptation, not production Python or the original PyTorch implementation; its 128-board qualification does not prove identical behavior on every possible input. The batch-128 Python p95 includes a large observed tail; raw trials remain available.

MLX active and cache counters are separate observations, not measures of total physical unified memory. Post-run batch-1 median cache was 0.653 GiB Rust versus 0.632 GiB Python, and batch-4 active memory was 0.164 versus 0.153 GiB: not every memory counter favors Rust. Batch-128 post-run median cache was 21.971 versus 24.638 GiB. Both processes retain substantial allocator cache; the peak column should not be interpreted as a total-memory bound. CPU activity was sampled 58 times during quality and 283 during timing; system processes including `mediaanalysisd` were observed. GPU utilization was not sampled and there was no exclusive reservation. The result is a single-machine measurement, not an idle-machine guarantee.

The captured source revision was `82e07ea`. The audit checked 84 quality and 87 benchmark tracked-source fingerprints against that revision, rechecked quality file hashes, benchmark-reported output hashes and derived metrics, and rejected a deliberately altered timing summary. Seven Mac binary/runtime-library fingerprints also matched after capture. Raw per-board outputs, process data and activity are retained locally on both machines; the local byte-verified copy is under `reports/phase3-mlx-captured-v1/`. Reproduce the local audit with `python3 tools/bench/audit_phase3_mlx.py reports/phase3-mlx-captured-v1`; original assets and the Linux reference capture are required.

The one medium disagreement is board 85, code position 181 (level 10, position 15): Python selected 8 and Rust 5. Python's top logits for 8 and 5 were 6.3651566505 and 6.3651471138 (margin 9.5367e-6); Rust's were 6.3651552200 and 6.3651561737. The maximum level-10 absolute logit difference across this batch was 1.5402e-4. This is a measured near tie, not an exact-code match; the decoded output is identical. No tolerance or golden has been changed.

Source and converted checkpoint hashes and dataset pins are recorded by `tools/reference/sudoku_artifact.py` and `tools/reference/sudoku_phase3_data.py`. Canonical artifact manifest SHA-256s: easy `ad6a99ad7d5ccc069e0fedf26a85f755e06cd76590463e1e04e17363e35d9832`, medium `facff42955d42294dc94f33d54688dc53046c1f9cc029e903f1ffe787ce45219`, hard `9ed3e86fdc3d193c88e1f4aae8f58a54019f985bdcf2acb9c851772c51dad86b`. The initial CUDA CLI binary was `cae4a61fbb83a8212141f26a05c6eb0bcd2dfe60cfe0b497bd7968e5e873b72f`; this alone is not a source-to-binary attestation.

The raw per-board codes and IDs, reports, CPU/GPU diagnostic traces and the failed `/usr/bin/time` invocation are retained locally under `reports/phase3-linux-exploratory-v1/` and `.reference/phase3-initial-oracle/`; their hashes are in the quality report. The user installed `/usr/bin/time` after that failure. They are not committed here. Those local outputs contain evaluation-derived data, and their redistribution has not been assumed.

Python used CUDA FP32, TF32 disabled, PyTorch math SDPA, no KV cache and batch 128. Rust used the explicit CUDA backend and batch 128. Python's self-reported execution timings exclude load; the Rust process records include load, so they are not comparable performance results. These exploratory runs did not control concurrent workload, thermal state, or process order. Do not treat them as the required controlled performance comparison.
