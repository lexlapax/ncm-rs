# Batch-sensitive normalization: large-batch Metal comparison

After the full CUDA and Metal checks and the five-seed sample identity comparison in the [hybrid screen](generator-hybrid-norm-screen-v1.md), the isolated hybrid binary was compared with the qualified Python MLX adaptation at batches 32 and 128. Three alternating fresh processes per runtime each ran five warmups and 30 measured generation/readback requests. Rust used output cleanup and a GPU barrier inside timing and an explicit 4,096 MiB process-local cache limit. Python used its default cache; their random streams differ. The cache limit is not an instantaneous hard bound.

| Batch | Python / Rust median ms | Python / Rust pooled p95 ms | Python / Rust peak RSS MiB | Python / Rust MLX peak / active / cache MiB |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 658.05 / 547.51 | 694.35 / 578.57 | 1046.34 / 1020.86 | 2528.94 / 479.29 / 15809.21; 2439.24 / 477.63 / 4095.52 |
| 128 | 2224.69 / 2553.30 | 2453.61 / 16353.81 | 1045.78 / 1020.03 | 2539.28 / 479.29 / 24596.63; 2490.48 / 477.63 / 3977.35 |

Batch 32 favored Rust in the reported latency and memory counters. **Batch 128 failed the latency direction:** nine Rust trials exceeded four seconds, all in the second Rust process, with a maximum 16.51 seconds; none of Python's trials exceeded four seconds. Rust's three process medians were 2003.11, 2588.12 and 2553.30 ms; Python's were 2207.38, 2224.69 and 2227.25 ms. The prior [fused-candidate tail failure](generator-fused-norm-mlx-performance-v1.md) remains relevant. This run establishes no cause or fix for the tail, and the candidate is not integrated.

RSS is a whole-process high-water mark, while the MLX counters have different scopes. The runner recorded 342 approximately two-second CPU activity snapshots, no direct GPU activity or exclusive reservation, and no telemetry errors; watched inputs and sources were stable. A separate read-only SSH process inspection was made once during the long second Rust process and found it active. No clock or power settings were changed. These observations do not establish the cause of the delay.

Raw ignored evidence is in `reports/phase4-hybrid-norm-mlx-large-bench-v1/`. The summary, manifest and activity SHA-256 values are `d9eb06ebfeb349fb283da23fcb429084f8275c04f00e6c08e275b4d4cdc278af`, `b83b0ed493ed6a36b0d9b54556c2f6d1dc3277fce55f0a3074ee25470398bfa2` and `9991c634013da7d368e9cfb41b46335500fe92cb777d0d402958a5401afca141`, respectively, verified against the Mac originals. Do not treat this as a four-batch performance pass or a Phase 4 completion.
