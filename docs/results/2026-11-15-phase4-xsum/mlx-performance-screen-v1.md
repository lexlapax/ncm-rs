# XSum Metal top-1: short performance screen

The initial same-device screen used the [32-passage-qualified Python MLX adaptation](python-mlx-reference-v1.md) and Rust at commit `1039e2c`, with two warmups and five timed requests in one fresh process per runtime/batch. Timed scope includes model generation, code and final-ID host readback, and explicit CPU FP64 final penalties; model loading is excluded. At all three batches the Rust and Python outputs matched exactly.

| Batch | Runtime | Median ms | Peak process RSS MiB | MLX peak MiB |
| ---: | --- | ---: | ---: | ---: |
| 1 | Rust | 209.20 | 1303.39 | 1117.88 |
| 1 | Python MLX | 239.92 | 1245.89 | 1206.26 |
| 4 | Rust | 542.71 | 1304.03 | 2318.73 |
| 4 | Python MLX | 597.74 | 1290.38 | 2492.07 |
| 32 | Rust | 2956.62 | 1302.56 | 2583.15 |
| 32 | Python MLX | 4450.06 | 1648.08 | 2856.92 |

The Rust median is lower in all three cells. Rust process RSS is higher by 57.50 MiB and 13.65 MiB at batches 1 and 4, respectively, so this screen does not meet the user's time-and-space direction there; batch 32 favors Rust on both. MLX peak is a separate native allocator measure and cannot be substituted for process RSS. RSS and MLX peak include model loading; the separate overall process wall times and allocator active/cache measurements are retained in the raw files. The reported MLX cache was substantial and is not total physical unified memory.

This is a short, one-process exploratory screen with process snapshots before/after, not continuous activity observation, a throughput test or a formal performance gate. Raw data are on Mac under `reports/phase4-xsum-mlx-performance-screen-v1/` and copied locally to `reports/phase4-xsum-mlx-performance-screen-captured-v1/`. Investigate the small-batch RSS gap and rerun controlled comparisons; do not average away failing cells. The Python MLX implementation is an adaptation qualified on 32 top-1 passages, not original PyTorch on Metal.
