# XSum CUDA repeated performance, reference-order profile

The `xsum-inference-v2` CUDA/cuDNN path was compared against pinned Python
CUDA at temperature 1, top-k 1, top-p 0 and CFG 0. Each batch has three fresh
processes per runtime, five warmups and 30 measured synchronized generations
per process. Runtime order alternated. All codes and final IDs matched within
each batch across runtimes and processes.

| Batch | Runtime | Median ms | Pooled p95 ms | Median peak process RSS MiB |
| ---: | --- | ---: | ---: | ---: |
| 1 | Rust | 155.85 | 157.28 | 1058.71 |
| 1 | Python | 188.51 | 190.26 | 1860.44 |
| 4 | Rust | 565.72 | 568.17 | 1060.54 |
| 4 | Python | 689.77 | 690.94 | 1860.20 |
| 32 | Rust | 4401.06 | 4409.08 | 1060.32 |
| 32 | Python | 5332.49 | 5338.15 | 1863.88 |

Median latency is the median of the three per-process medians; p95 is pooled
across 90 trials. `/usr/bin/time` reports whole-process peak RSS, including
load; timings exclude load. Python garbage collection and synchronization are
run after each trial, measured separately and excluded from the table; the
summary also records cleanup-inclusive timings. Native default-pool and Torch
allocator numbers cover different allocations and cannot be directly compared.
These measurements satisfy the requested speed and process-memory direction in
the tested cells; they do not measure positive-CFG performance.

The activity collector recorded 1,207 samples and no query failures. Desktop
GPU processes were present; device-wide telemetry cannot exclude interference.
The raw output and recorded source, binary, artifact, driver and library hashes
are under `reports/phase4-xsum-native-performance-cfg-source-v1/`. The manifest
hash is `15d04726967104a144ea3dac41c0e563076a570bba20602268c210b18cf32295`;
the summary hash is `e70f0a26f77cf72616e17d0efa1968d4ea1dcb0cd8a3dc96464006a74f029e35`.
The collector began before commit `64f8a2a`, so its `git_head` records
`756df68` with the guidance change in the working tree. All 42 tracked source
files in the manifest match their hashes at `64f8a2a`; binary and library
hashes were checked before and after the run.

```sh
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/bench/run_phase4_xsum_bench.py --profile v2 --repetitions 3 --batches 1 4 32 --cudnn-dir .reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib --output NEW_REPORT
.reference/venv/bin/python tools/bench/summarize_phase4_xsum_b32.py --directory NEW_REPORT --batches 1 4 32
```

This is not the complete XSum gate: updated-source full quality, broader
guidance and tie coverage, and MLX remain open.
