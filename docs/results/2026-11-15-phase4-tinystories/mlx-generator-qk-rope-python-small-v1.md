# Fused Q/K RMS and rotary: batch-1/4/32 Python MLX comparison

The same isolated candidate and protocol as the [batch-128 comparison](mlx-generator-qk-rope-python-b128-v1.md) were run in three alternating fresh processes per runtime at batches 1, 4 and 32, each with five warmups and 30 measured requests. Rust includes output drop and a GPU barrier, with a 512 MiB process-local cache limit at batches 1/4 and 4,096 MiB at batch 32; Python uses its default cache. Random streams differ.

| Batch | Python / Rust median ms | Python / Rust pooled p95 ms | Python / Rust maximum process RSS MiB | Python / Rust maximum MLX peak MiB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 107.65 / 94.39 | 108.10 / 94.74 | 1046.77 / 1024.92 | 609.47 / 607.43 |
| 4 | 170.61 / 142.98 | 171.96 / 143.93 | 1046.88 / 1026.11 | 975.69 / 971.24 |
| 32 | 655.93 / 478.45 | 661.78 / 479.24 | 1046.73 / 1022.59 | 2528.94 / 2420.17 |

Rust was faster and lower in the measured memory counters at all three batches in these runs. The small-batch MLX peak margins were only 2.04 and 4.45 MiB. The CPU activity monitor reported no errors; there was no GPU activity measurement or exclusive reservation. Counters have different scopes, as described in the batch-128 report. This is not a claim about other profiles, future tail behavior, or an integrated production build.

Inputs and candidate sources remained hash-stable. Copied ignored evidence is `reports/phase4-qk-rope-mlx-python-small-v1/`: summary SHA-256 `0497ee4035f816422e47a1aaa6dfe6cc8850e4f299cdfd975d08a495c602b5ba`, manifest `7350d1e6dff1c599a5844eec3c16b8c36bd2d5e7b271cd19ea6a27d599fb240a`, and CPU activity `883f1fd79e7b1f64c8bd698cd64687530352c9feb207b99e6e6bb9206f7a7cf8`, matching the Mac.
