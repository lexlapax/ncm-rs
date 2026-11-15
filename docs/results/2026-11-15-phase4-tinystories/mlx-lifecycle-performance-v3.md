# TinyStories Metal lifecycle-policy comparison

Three alternating fresh processes per runtime and batch each ran five warmups and 30 generation/code/ID-readback trials on the Apple M1 Ultra. For Rust, the timed scope additionally drops unused logits and output tensors and evaluates a small GPU barrier. The dedicated Rust process explicitly sets a 512 MiB MLX cache limit for batches 1 and 4, and keeps the default cache limit for batches 32 and 128. This process-global choice does not alter system settings or the library default, but it is **not** a result for an unconfigured long-lived or multi-model application. Python uses the separately qualified same-device MLX adaptation and its default cache. The runtimes use different categorical random streams. Previous quality files remain byte-identical after the safety and layout changes; [quality evidence](mlx-quality-v1.md).

| Batch | Python median ms | Rust median ms | Python pooled p95 ms | Rust pooled p95 ms | Python peak RSS MiB | Rust peak RSS MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 107.684 | 94.194 | 109.686 | 95.020 | 1046.17 | 1024.75 |
| 4 | 169.843 | 143.710 | 172.320 | 144.377 | 1046.36 | 1025.59 |
| 32 | 691.182 | 607.749 | 698.785 | 611.620 | 1046.42 | 1025.25 |
| 128 | 2271.868 | 2119.253 | 2658.448 | 7016.875 | 1045.59 | 1025.75 |

Latency is the median of three process medians; p95 is nearest rank over 90 pooled trials. RSS is the median whole-process high-water mark including startup and load. Rust's median latency and RSS are lower in all cells, but its **batch-128 pooled p95 is much worse**. Rust's batch-128 process medians were 2013, 2119 and 3047 ms; eight trials in the second Rust process exceeded four seconds and its slowest took 14.8 seconds. Python's process medians were 2248, 2272 and 2429 ms. This tail regression is an unresolved performance result, not a pass inferred from the median.

| Batch | Python MLX peak / active / cache MiB | Rust MLX peak / active / cache MiB |
| ---: | ---: | ---: |
| 1 | 609.08 / 479.29 / 582.80 | 607.04 / 477.63 / 512.20 |
| 4 | 975.69 / 479.29 / 2401.78 | 971.20 / 477.63 / 512.73 |
| 32 | 2528.94 / 479.29 / 15809.25 | 2438.42 / 477.63 / 15038.30 |
| 128 | 2539.28 / 479.29 / 24789.95 | 2427.42 / 477.63 / 22148.49 |

Each counter is the median of three immediate post-trial process observations. Under this explicit policy Rust's reported values are lower in every cell, although active-memory margins are only about 1.66 MiB. Peak, active and cache have different scopes from process RSS and total physical memory. The earlier [default-policy comparison](mlx-hybrid-performance-v2.md) remains visible and had higher Rust active or cache in several cells.

The capture recorded 368 approximately two-second CPU activity samples. Approximate alignment using process duration and monotonic clock offset places sustained `triald` CPU activity during the start of the slow Rust cluster, with other background activity also present. This is correlation, not evidence of causation: GPU activity was not directly measured, exact per-trial event timestamps were not captured, and the machine was not exclusively reserved. No GPU clock or power setting changed. Input, source, binary and library hashes remained stable. The ignored local capture is `reports/phase4-tinystories-mlx-lifecycle-bench-v3/`; `manifest.json` SHA-256 is `c4366ce2dfe2691fc517ace00f391df20ec8240d7e648cbd2038cd858646d2bf` and `summary.json` SHA-256 is `0159ef21aa6ca1842564eabaa12e6869376eb2886e5a2dae90a512184972ef82`.
