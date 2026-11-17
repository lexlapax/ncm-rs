# WMT14 CUDA stage profile

The restored immediate-validation path was profiled with explicit device
synchronization at each trace checkpoint. This is an instrumented diagnostic,
not headline latency or a Python/Rust stage comparison. It includes CPU work
between checkpoints, changes scheduling, and excludes model loading. Three
warmups and five measured runs were made for CFG=0 batch 32 and CFG=1 batch
16. All codes and raw IDs matched the retained uninstrumented Rust benchmark
outputs at the same workload. The first level interval includes prefix upload
and preparation. Each reconstruction interval includes selection and its
validation, embedding and reconstruction; each subsequent generator interval
includes target preparation and transformer/head execution. The final interval
includes codec decode and greedy selection. Medians are per checkpoint over
five runs and must not be added as a latency estimate.

| Level | CFG=0 batch 32 generator / reconstruction ms | CFG=1 batch 16 generator / reconstruction ms |
|---:|---:|---:|
| 0 | 75.60 / 1.30 | 76.85 / 1.26 |
| 1 | 84.99 / 0.82 | 82.96 / 0.81 |
| 2 | 88.30 / 0.83 | 87.40 / 0.83 |
| 3 | 93.47 / 0.84 | 91.84 / 0.83 |
| 4 | 97.75 / 0.86 | 98.31 / 0.84 |
| 5 | 103.84 / 0.87 | 103.98 / 0.86 |
| 6 | 107.26 / 0.89 | 108.23 / 0.86 |
| 7 | 121.72 / 0.91 | 120.14 / 0.87 |
| 8 | 128.15 / 0.92 | 129.03 / 0.90 |
| 9 | 134.17 / 0.94 | 133.99 / 0.90 |
| 10 | 142.38 / 0.74 | 141.49 / 0.69 |

The final decoder/selection checkpoint was 125.37 ms for base and 63.50 ms
for CFG. In this diagnostic, the generator intervals dominate; the tiny
reconstruction/validation intervals help explain why deferred validation did
not materially improve throughput. A finer transformer breakdown and a
same-scope Python profile are still required before attributing a bottleneck
to a particular operation. Desktop GPU processes were present.

The locally built probe passed feature-specific Clippy with `-D warnings` and
release build; the restored generator had already passed the full CUDA check.
The runs were exploratory before committing the new example. Binary SHA-256
was `0886c33457e22b5e10f187a0de39479afa84136aa3bd7193affe63be34b9324b`;
the base and CFG result SHA-256 values are
`625f90aceb00477c941d568165a0eb638f8c98e90816e7d73a9e5dc6bce34a9e` and
`fb0802c296cf751e96ca6f8e692a3d0248267d4bf668b2dba878bb4a5239bd41`.
Raw outputs remain in ignored `reports/phase4-wmt14-stage-{b32,cfg16}-v1.*`.
The WMT CUDA performance gate remains open.
