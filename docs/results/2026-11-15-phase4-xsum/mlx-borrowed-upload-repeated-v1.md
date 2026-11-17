# XSum borrowed upload: repeated Metal comparison

Following the [predeclared plan](mlx-borrowed-upload-repeated-plan-v1.md), the
Linux CUDA harness check passed before Mac testing. On clean commit `6d72013`,
full `tools/check.sh --mlx` passed, including 18 real-Metal operator cases
(captured log SHA-256
`a980b2a82320ebd8c4088019f3e4bd151e98da4aa4944e481a044ab9e9f8fb50`).
Focused Metal Clippy and release build passed. On real Metal, the default 2/5
and explicit 1/1 warmup/trial modes of both Rust and adapted Python produced
matching codes and IDs. The candidate CLI had already passed its ordered
32-passage comparison with pinned Python CUDA after the corrected Linux CLI
check; its qualification SHA-256 is
`30415fa65a9346989ea8b4924a27d565bfd448ece82c310cb09adc148d7f3d12`.

The Mac command was `.reference/mlx-venv/bin/python
tools/bench/run_phase4_xsum_mlx_bench.py --qualification
reports/phase4-xsum-borrowed-upload-mlx-cli-v3/result.json
--qualification-sha256 30415fa65a9346989ea8b4924a27d565bfd448ece82c310cb09adc148d7f3d12
--gpu-activity --output reports/phase4-xsum-borrowed-upload-mlx-repeated-v1`.
There were three fresh processes per runtime and batch, five warmups and 30
synchronized trials per process. Every run's codes and IDs matched the other
runtime and all repetitions for its batch. Load is excluded from timing;
process peak RSS includes startup and load. The Python adapter's explicit
garbage collection is outside recorded request latency.

| Batch | Runtime | Median of process medians ms | Pooled p95 ms | Median peak RSS MiB | Median MLX peak MiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Rust | 211.75 | 214.95 | 1205.25 | 1117.88 |
| 1 | Python MLX | 238.42 | 241.55 | 1246.53 | 1206.26 |
| 4 | Rust | 544.08 | 546.60 | 1205.52 | 2318.73 |
| 4 | Python MLX | 599.74 | 637.39 | 1296.95 | 2492.07 |
| 32 | Rust | 3289.41 | 3796.99 | 1203.97 | 2583.15 |
| 32 | Python MLX | 4453.64 | 7422.67 | 1646.97 | 2828.37 |

Rust meets the plan's measured median, pooled-p95 and process-RSS conditions in
all three batches. The Rust batch-32 process medians were 2992.57, 3690.13 and
3289.41 ms; Python's were 4453.64, 4420.38 and 4819.82 ms. The causes of this
variation and Python's high pooled tail are not established. The 532 CPU and
532 GPU registry observations had no collector errors; device utilization
ranged from 38% to 100%. The registry snapshots are system-wide, can perturb
measurements and cannot attribute GPU time or establish exclusive use. MLX
memory counters have a different scope from RSS. This is a comparison against
an adapted Python MLX implementation, not original PyTorch on Metal, and it is
not a paired measurement of the previous loader.

All watched sources and inputs were stable at postflight. Manifest and summary
SHA-256 values are respectively
`86c7febc22bcba4700d7f7010081632f397bb1d8859439a3eda1f89a3b96e120` and
`863b43414b02a30eae900466dd3a16ab99643d3536691dfdb60e34b2a42b1fed`.
The raw files were copied to
`reports/phase4-xsum-borrowed-upload-mlx-repeated-captured-v1/` and all copied
file hashes matched the Mac. This 32-passage performance comparison does not
establish full candidate quality or resolve the [full-corpus raw-ID failure](mlx-full-top1-v1.md).
