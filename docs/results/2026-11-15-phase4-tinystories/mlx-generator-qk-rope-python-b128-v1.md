# Fused Q/K RMS and rotary: batch-128 Python MLX comparison

The isolated [candidate](mlx-generator-qk-rope-screen-v1.md) was measured against the qualified same-device Python MLX adaptation in three alternating fresh processes per runtime, with five warmups and 30 measured requests per process. Both measured generation and readback; Rust additionally included output drop and a GPU barrier, using a process-local 4,096 MiB cache limit. Python used its default cache. The random streams differ.

| Metric | Python | Rust |
| --- | ---: | ---: |
| Median of process medians | 2196.05 ms | 1616.38 ms |
| Pooled p95 | 10770.83 ms | 1931.59 ms |
| Maximum process peak RSS | 1046.00 MiB | 1021.92 MiB |
| MLX peak / active / maximum cache | 2539.28 / 479.29 / 24597.20 MiB | 2475.29 / 477.63 / 4086.29 MiB |

Rust was faster and used less memory for these measured counters in this batch-128 run. Its process medians were 1616.38, 1615.81 and 1834.56 ms; none of its 90 trials exceeded four seconds (maximum 1970.33 ms). Python's were 2196.05, 2766.02 and 2163.77 ms; ten Python trials exceeded four seconds in the second Python process, with a maximum of 17827.33 ms. This is a reversal of the earlier [compiled-GELU comparison](mlx-generator-compiled-gelu-python-performance-v1.md), where eight Rust trials exceeded four seconds. Neither run establishes the cause of the tail or proves it fixed.

The run used the candidate's own 16-row fixed-code qualification, and watched inputs and source were stable. CPU activity was sampled without errors, but no GPU activity or exclusive reservation was measured. RSS includes startup and loading; MLX peak, active and cache are different counters and the cache limit is not a hard instantaneous bound. The other batch comparisons and safe production integration remain separate. Full isolated CUDA and MLX checks subsequently passed, as recorded in the screen report.

Raw ignored evidence: `reports/phase4-qk-rope-mlx-python-b128-v1/`. The copied summary SHA-256 is `02b23dc701c98da363ef0108795d4ac2e8c31a21a9b10f468720d50c777b8bdb`, manifest `bb4474e01eb9713e067a38fbf25f73e1fc8075bc170735939de93249e58d0852`, and activity file `6d464512caafcf4b8628c4e1a09c8c46d791a1668b76ca9f549b3ffd8b298ecb`; each matched the Mac original.
