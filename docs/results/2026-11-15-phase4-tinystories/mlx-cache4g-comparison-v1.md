# TinyStories Metal 4 GiB cache comparison

Three alternating fresh processes per runtime and batch each ran five warmups and 30 timed generation/code/ID-readback trials. Rust used the explicit process-local 4,096 MiB cache setting and included output drop and a GPU barrier in the timing; Python used its default MLX cache. This tests batches 32 and 128 only, using the separately qualified Python MLX adaptation and the same fixed-code qualification as the previous comparison. Model load is excluded from timing; whole-process peak RSS includes it. Inputs, sources, binaries and libraries remained hash-stable; telemetry reported no errors. The setting does not modify the library default or apply to the CLI automatically.

| Batch | Python / Rust median ms | Python / Rust pooled p95 ms | Python / Rust peak RSS MiB |
| ---: | ---: | ---: | ---: |
| 32 | 657.05 / 617.64 | 661.67 / 629.13 | 1046.33 / 1025.44 |
| 128 | 2166.95 / 2343.16 | 2537.69 / 3992.94 | 1045.52 / 1024.58 |

| Batch | Python MLX peak / active / cache MiB | Rust MLX peak / active / cache MiB |
| ---: | ---: | ---: |
| 32 | 2528.94 / 479.29 / 15809.14 | 2439.94 / 477.63 / 4094.29 |
| 128 | 2539.28 / 479.29 / 24596.45 | 2427.45 / 477.63 / 3974.44 |

Latency is the median of three process medians; p95 is nearest rank across the 90 pooled measured trials. Memory cells are medians of three immediate post-trial observations, except RSS, which is the median process high-water mark. MLX counters differ in scope from each other and from physical memory; the cache setting is not an instantaneous hard bound, as the [single-process exploration](mlx-cache4g-probe-v1.md) shows.

Batch 32 favored Rust on the reported latency and memory measures. Batch 128 **failed the requested latency direction**: Rust process medians were 2063.60, 2343.16 and 2467.69 ms, versus Python 2166.95, 2161.69 and 2472.07 ms. Four Rust trials in the second process exceeded four seconds, with a maximum of 16.86 seconds; no Python trial exceeded four seconds. The new cap substantially reduced Rust's retained cache but did not reliably fix the tail and cannot be declared a performance gate pass. The original [default-large-batch comparison](mlx-lifecycle-performance-v3.md) and its outlier remain preserved.

There were 340 approximately two-second CPU activity samples; WindowServer and other background processes were present and `corespotlightd` reached 166.7% CPU in seven samples. GPU activity, exact per-trial timestamps and exclusive reservation were unavailable, so these observations do not identify the cause of a slow trial. No GPU clock or power policy changed. Different Rust and Python random streams are used. This is a dedicated-process benchmark, not concurrent-service or CLI performance evidence.

Raw reports are retained at `reports/phase4-tinystories-mlx-cache4g-bench-v1/`. SHA-256 values for `summary.json`, `manifest.json` and `activity.jsonl` are `dbbaf4b97c4161a676ea98b0f326f352355e0cdbf4763ba87c98daf3f6b7294e`, `97aec87e2ae999ff92577599bf581f8fb57240f1d74c57a966816db35369047e` and `3d75f6f59e93c897884f1968d25b7217803a25de220916dcaf7a3e1d1edc7a8c`, respectively, verified against the Mac originals. The checked runner was commit `1791347`.
