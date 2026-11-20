# WMT14 CUDA/Python repeated model performance comparison

2026-11-20. At clean commit `86bd0b5`, the existing fast-argmax Rust CUDA model probe and pinned upstream Python were compared on an NVIDIA RTX 3060 Ti, CFG=0/top-1, with the first 1/4/16/32 pinned corpus prefixes. Each cell used three alternating fresh processes per runtime, five warmups and 30 synchronized measured requests per process. Timing covers prefix upload, generation, codec decode and code/ID readback, excluding model load and Python's separately measured garbage collection. RSS includes loading. The collector checked output codes and IDs across runtimes and repetitions for each batch; this is not a full-corpus quality test. The separate [nonblind quality retest](cuda-wmt14-quality-retest-v1.md) covers all 3,003 base rows; the prior CFG=1 diagnostic failure remains disclosed there.

| Batch | Runtime | Median of process medians (ms) | Pooled p95 (ms) | Median peak RSS (MiB) | Sampled GPU process peaks (MiB) |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Rust | 57.753 | 58.223 | 1794.191 | 1222 / 1222 / 1222 |
| 1 | Python | 64.236 | 64.583 | 2093.254 | 1360 / 1360 / 1360 |
| 4 | Rust | 165.591 | 166.210 | 1789.051 | 1254 / 1254 / 1254 |
| 4 | Python | 177.943 | 179.455 | 2093.160 | 1492 / 1492 / 1492 |
| 16 | Rust | 617.598 | 619.304 | 1784.203 | 1414 / 1574 / 1734 |
| 16 | Python | 650.454 | 653.923 | 2093.070 | 2300 / 2300 / 2300 |
| 32 | Rust | 1205.846 | 1207.330 | 1819.145 | 1640 / 2792 / 1608 |
| 32 | Python | 1276.103 | 1277.579 | 2092.914 | 3726 / 3726 / 3726 |

Rust had lower observed median latency and process RSS in these cells. The [compact aggregate](wmt14-cuda-perf-summary-v1.json) contains per-process medians and RSS, sampled memory, sample counts and Python allocator counters; its SHA-256 is `25516fe1c6c5195c16a0960f87144308e7d95795c2859445ad4ba984fd992229`. Rust native allocator peaks were not collected. The driver GPU numbers are roughly 1 Hz sampled process memory, not native allocation peaks; batch-32 Rust samples visibly vary, so a lower sampled value is not proof of a lower true peak. All 522 activity samples had successful queries and four other GPU processes were observed throughout. The GPU was not exclusively reserved.

The raw bundle remains ignored locally at `reports/phase5-wmt14-cuda-perf-v1/`, not durably published. The manifest, measurements, process events and activity hashes are in the aggregate. The probe binary SHA-256 is `8f630ee1d40ce6a632dafc3f6449d4f810b5a7cfa438edc6571e95ba24ed9774`. The collector rechecked its watched files and clean tracked source after capture; its watch list does not prove all mapped libraries. The aggregate replayed byte-for-byte and its offline tests passed. The first full `tools/check.sh` failed in the previously observed lifecycle `capture_error:ObservationMismatch`; a focused lifecycle retry and a subsequent full `setpriv --inh-caps=-all tools/check.sh` passed. The failure's cause remains unestablished.

Reproduce after preparing the pinned corpus prefixes and artifact as in the [Phase 4 comparison](../2026-11-16-phase4-wmt14/cuda-fast-argmax-plan-v1.md#source-bound-corpus-and-repeated-timing), with clean tracked source and a new output directory:

```sh
setpriv --inh-caps=-all env LIBRARY_PATH="$PWD/.reference/phase4-cudnn-probe-lib" \
  cargo build --release --locked -p ncm-backend-cuda --features wmt14-experimental,cuda-cudnn --example wmt14_fast_argmax_perf
setpriv --inh-caps=-all env -u LD_LIBRARY_PATH -u PYTHONPATH OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  .reference/venv/bin/python tools/bench/run_wmt14_fast_argmax_bench.py \
  --output reports/NEW-DIRECTORY --repetitions 3 --batches 1 4 16 32 --modes 0
setpriv --inh-caps=-all python3 tools/bench/summarize_wmt14_phase5.py reports/NEW-DIRECTORY
```

This probe does not invoke the changed CLI, measure cold load, compare previous Rust with candidate, prove GPU isolation or provide Metal evidence. It is a partial current Rust/Python model comparison, not the Phase 5 release performance gate.
