# Fused generator normalization: Metal performance comparison

Three alternating fresh processes per runtime and batch each ran five warmups and 30 synchronized generation/code/ID-readback requests. Rust used the isolated fused-normalization candidate, dropped unused outputs and evaluated a GPU barrier within each timed request. Its dedicated process selected an explicit 512 MiB MLX cache limit at batches 1 and 4 and 4,096 MiB at batches 32 and 128. Python used the separately qualified same-device MLX adaptation with its default cache. The policy does not change a library default and does not apply automatically to the CLI. Random streams differ. The [nonblind quality retest](generator-fused-norm-quality-v1.md) passed its frozen criteria, while sampled sequences differed.

| Batch | Python / Rust median ms | Python / Rust pooled p95 ms | Python / Rust peak RSS MiB |
| ---: | ---: | ---: | ---: |
| 1 | 107.45 / 80.71 | 108.12 / 81.01 | 1046.41 / 1020.25 |
| 4 | 170.17 / 127.02 | 172.10 / 127.71 | 1046.39 / 1021.05 |
| 32 | 657.05 / 549.42 | 660.76 / 559.06 | 1046.56 / 1020.69 |
| 128 | 2179.43 / 1996.83 | 2374.95 / 12649.79 | 1045.59 / 1019.89 |

| Batch | Python MLX peak / active / cache MiB | Rust MLX peak / active / cache MiB |
| ---: | ---: | ---: |
| 1 | 608.31 / 479.29 / 579.30 | 629.90 / 477.63 / 512.10 |
| 4 | 975.69 / 479.29 / 2400.90 | 1059.86 / 477.63 / 513.00 |
| 32 | 2528.94 / 479.29 / 15809.16 | 2437.98 / 477.63 / 4094.83 |
| 128 | 2539.28 / 479.29 / 24596.82 | 2490.48 / 477.63 / 4014.10 |

Latency is the median of three process medians; p95 is nearest rank across 90 pooled trials. RSS is the median whole-process high-water mark, including startup and load. Each MLX counter is the median of three immediate post-trial observations and has a different scope from RSS and physical memory. The cache limit is not an instantaneous hard bound.

Rust's median latency and RSS were lower in all four cells, but **MLX peak was higher at batches 1 and 4**, and **batch-128 p95 was worse**. Rust's batch-128 process medians were 1973.46, 1996.83 and 2278.10 ms; process `r1` had nine trials above four seconds, with a maximum of 16.58 seconds. No Python trial exceeded four seconds. The candidate therefore does not pass the requested performance/memory direction and was not integrated. The earlier failures are retained; this run does not identify their cause.

The capture recorded 356 approximately two-second CPU activity samples, including background processes. GPU activity was not directly measured, exact per-trial timestamps were not captured and the machine was not exclusively reserved. No clock or power policy changed. Input, source and binary hashes remained stable, and telemetry reported no errors. These limits also apply to the comparison with earlier runs; no background process is asserted as the cause.

Raw reports are `reports/phase4-fused-norm-mlx-bench-v1/`. SHA-256 values for `summary.json`, `manifest.json` and `activity.jsonl` are `0181bf51291416997d9a20fead266e4a79ea4c46e5b2fafe05d358658d62a8a9`, `93412209b84650f835ad75d7e5e1aef5d057259a5f82f9271fbac6630936adea` and `f01c98b5614da3ef7380ddd4cf933028f69aa179836a80a953da08a7d22e2d02`, respectively, verified against the Mac originals. The runner recorded commit `17544e8` and fingerprinted the isolated candidate tree and binary.
