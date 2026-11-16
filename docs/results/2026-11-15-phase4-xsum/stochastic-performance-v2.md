# XSum stochastic CUDA performance and memory

The current strict-cuDNN Rust and pinned Python CUDA implementations ran temperature 1, top-k 0, top-p 1, CFG 0 with seed 42 and independent runtime RNGs. For each batch and runtime, three alternating fresh processes ran five warmups and 30 timed requests, synchronized through code/ID host readback; model loading is excluded. Python garbage collection and synchronization after each request are measured separately, so the timings below exclude those cleanup costs. Each runtime's outputs were checked for stable shape, range and repeatability, without requiring different RNG streams to match. Rust is faster and lower-memory in all measured cells.

| Batch | Runtime | Median ms | Pooled p95 ms | Median peak RSS MiB | Median sampled driver peak MiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Rust | 165.69 | 177.79 | 1061.81 | 838 |
| 1 | Python | 202.42 | 217.16 | 1847.84 | 990 |
| 4 | Rust | 595.59 | 616.51 | 1064.07 | 1062 |
| 4 | Python | 728.23 | 752.14 | 1846.75 | 1624 |
| 32 | Rust | 4651.84 | 4857.99 | 1061.18 | 3080 |
| 32 | Python | 5639.17 | 5676.07 | 1849.89 | 6214 |

Medians are the median of the three process medians; p95 is pooled across 90 measured requests. RSS includes startup and model load. NVIDIA driver memory is sampled approximately once per second and can miss peaks. CUDA pool and Python allocator counters have different scopes and are not directly interchangeable; individual values and cleanup timing are in the [summary](stochastic-performance-v2.json), and [driver memory sampling](stochastic-performance-v2-gpu.json) records per-process peaks and sample counts. Across 1,274 samples, no activity queries failed. Three other GPU processes were present and their reported aggregate memory ranged from 151 to 681 MiB; GPU utilization ranged from 0 to 100%. This is not a fully isolated desktop or concurrent-request throughput measurement.

Raw inputs, command records, activity and output evidence are retained under `reports/phase4-xsum-stochastic-repeated-v2/`. An earlier `reports/phase4-xsum-stochastic-repeated-v1/` run was interrupted because a CPU bootstrap analysis overlapped its first timing measurements; those partial results are not used here. The complete v2 run was performed without that analysis. This performance result does not override the [failed stochastic quality gate](stochastic-heldout-v1.md), and XSum Linux and MLX qualification remain open.
