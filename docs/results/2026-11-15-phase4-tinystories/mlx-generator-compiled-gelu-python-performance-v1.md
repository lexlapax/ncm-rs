# Compiled generator GELU: Python MLX comparison

After the compiled candidate passed the full CUDA and Metal checks, the [five-seed quality-file identity check and fixed-code comparisons](mlx-generator-compiled-gelu-screen-v1.md), it was compared with the qualified same-device Python MLX adaptation. Three alternating fresh processes per runtime and batch each ran five warmups and 30 measured generation/readback requests. Rust used output drop and a GPU barrier inside timing, with explicit process-local cache limits of 512 MiB at batches 1/4 and 4,096 MiB at batches 32/128. Python used its default cache; their random streams differ.

| Batch | Python / Rust median ms | Python / Rust pooled p95 ms | Python / Rust peak RSS MiB | Python / Rust MLX peak / active / cache MiB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 107.49 / 94.38 | 107.94 / 94.67 | 1046.39 / 1024.69 | 608.70 / 479.29 / 581.50; 607.39 / 477.63 / 512.50 |
| 4 | 170.44 / 143.22 | 171.96 / 144.00 | 1046.56 / 1026.08 | 975.69 / 479.29 / 2399.95; 971.17 / 477.63 / 512.53 |
| 32 | 656.91 / 501.91 | 660.99 / 502.79 | 1046.75 / 1020.75 | 2528.94 / 479.29 / 15809.14; 2425.56 / 477.63 / 4095.30 |
| 128 | 2168.02 / 2186.48 | 2539.21 / 12214.94 | 1045.73 / 1019.92 | 2539.28 / 479.29 / 24596.45; 2475.50 / 477.63 / 4093.44 |

Rust was faster and lower in the listed memory counters at batches 1, 4 and 32 in this run; the small-batch MLX peak margins were only 1.31 and 4.52 MiB. **Batch 128 failed the required latency direction:** eight Rust trials exceeded four seconds, all in its second process, with maximum 15.70 seconds; no Python trial exceeded four seconds. Rust process medians were 1745.82, 2452.67 and 2186.48 ms; Python process medians were 2168.02, 2160.98 and 2413.04 ms. The candidate is not integrated, and the [earlier hybrid failure](generator-hybrid-norm-mlx-large-performance-v1.md) remains relevant.

The runner recorded 352 approximately two-second CPU activity samples without errors, but no direct GPU activity or exclusive reservation. During the 77 snapshots containing the slow Rust process, WindowServer did not appear among the top 12 processes at 15% CPU or above. This narrows a CPU-activity hypothesis from the earlier run but does not measure WindowServer GPU work or identify the cause. RSS is a whole-process high-water mark; MLX counters have different scopes, and the cache limit is not a hard instantaneous bound. Startup and compile-cache resource behavior remain separate, unresolved concerns.

Watched inputs and sources were stable. Raw ignored evidence is `reports/phase4-gelu-compiled-mlx-python-bench-v1/`; summary, manifest and activity SHA-256 values are `22d566aa188a54d2578c1c09fc550ef08f9fbab17a9de37df78459d749a10e37`, `0c78902c5297278f9980baceed9651de735101fd0b3e981ffb5e6b89d83725fb` and `5e697f75c2bc58c2813f4a365b9d6cfd94d99e8e57a9bbdbae2cf44a75437c43`, respectively, verified against Mac originals. This is not a Phase 4 performance pass.
