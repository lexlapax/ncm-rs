# TinyStories CUDA safety and Q/K/V candidate performance

After adding sentinel-safe sampled embeddings and a shared Q/K/V split dispatch, three alternating fresh processes per runtime and batch each performed five warmups and 30 synchronized generation/readback trials on the RTX 3060 Ti. The CUDA implementation retains its existing packed Q/K/V behavior; the separate Metal view implementation is evaluated independently. Python uses the pinned upstream runtime. Rust and Python use different categorical random streams. All five previously scored strict-CUDA sample files were byte-identical after this change, as documented in the [quality report](heldout-quality-v1.md).

| Batch | Python median ms | Rust median ms | Python pooled p95 ms | Rust pooled p95 ms | Python peak RSS MiB | Rust peak RSS MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 47.048 | 41.181 | 48.235 | 42.320 | 2335.9 | 971.7 |
| 4 | 101.321 | 96.782 | 102.526 | 98.091 | 2319.0 | 971.9 |
| 32 | 628.626 | 604.879 | 633.998 | 609.144 | 2331.9 | 966.2 |
| 128 | 2448.972 | 2356.914 | 2489.236 | 2361.817 | 2280.8 | 970.1 |

Latency is the median of process medians; p95 is nearest rank across 90 trials. RSS is the median whole-process GNU `time` high-water mark, including loading. Rust was faster and had lower RSS in all four cells, with a smallest latency margin of about 3.8%. Python's explicit `gc.collect()` after each request was outside the timed generation scope; the reason and separately measured cost are disclosed in the [earlier strict-path comparison](cuda-strict-performance-v1.md).

GPU-process sampled peaks, Python/Rust in MiB, were 844/678 at batch 1, 928/712 at batch 4, 2386/1096–1160 at batch 32, and 5532–5964/2760 at batch 128. One-second process samples can miss transient peaks and are not equivalent to native allocator counters. Chrome, 1Password, walker and t3code were present in all 711 GPU-activity samples. No clock policy was changed. The small timing margins and device-memory coverage warrant these limitations.

The source and executable identities were checked before and after the run. Ignored `reports/phase4-tinystories-linux-bench-safe-split-v1/manifest.json` has SHA-256 `dd4ac2e46d719cf39a36d27c4f3b40b06d79bb6aa07aca081a9042421e1055c7`; the independent `summary.json` has SHA-256 `d3092d12abf09d91dbd5f64f737c87b1a54409d76fc9afff10f780f98a190c4b`. Per-process trials, RSS, Python collection costs and activity are retained alongside them. The earlier separate native allocator measurements have different coverage; updated native pool observations are below.


## Native allocator observations

A separate short capture ran one warmup and one measured request in each of three fresh Rust processes per batch. The CUDA default async-pool high water includes loading; current values were read after request outputs were dropped. Python's peak counters come from the three-process comparison above; its current counters are from the separate short Python observation in the [earlier report](cuda-strict-performance-v1.md). All values are MiB.

| Batch | Rust pool used peak / current | Rust pool reserved peak / current | Python allocated peak / current | Python reserved peak / current |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 492.9 / 482.6 | 512 / 512 | 648.9 / 629.5 | 676 / 676 |
| 4 | 524.0 / 482.6 | 544 / 512 | 700.9 / 629.5 | 760 / 760 |
| 32 | 814.0 / 482.6 | 992 / 512 | 1183.6 / 629.5 | 2218 / 2218 |
| 128 | 1808.3 / 482.6 | 2592 / 576 | 2836.9 / 629.5 | 5796 / 4054 |

The counters measure different allocators: CUDA pool values omit context and non-pool allocations; PyTorch values omit allocations outside its allocator. These observations therefore do not prove an equal-coverage instantaneous device peak. The native-capture binary SHA-256 was `a6b34b3f1cfa92de662ccb6a34ead25721a21220ac579966eee91eb8ef3c2115`; raw process reports and GPU activity snapshots are in ignored `reports/phase4-safe-split-cuda-pool-v1/`.
