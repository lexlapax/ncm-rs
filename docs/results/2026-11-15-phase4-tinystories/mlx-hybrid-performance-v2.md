# TinyStories Metal safety and hybrid-layout comparison

This compares the sentinel-safe, batch-sensitive Q/K/V candidate with the same-device Python MLX adaptation in three alternating fresh processes per runtime and batch, each with five warmups and 30 generation/code/ID-readback trials. The candidate uses separate Q/K/V views below batch 16 and the original packed path at larger batches. Fixed-code comparisons cover both branches, and all five generated 256-sample Metal files were byte-identical to the previously scored outputs; see the [development](mlx-development-v1.md) and [quality](mlx-quality-v1.md) reports. The original Python CUDA float-band exceptions remain disclosed there.

| Batch | Python median ms | Rust median ms | Python pooled p95 ms | Rust pooled p95 ms | Python peak RSS MiB | Rust peak RSS MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 107.081 | 93.940 | 108.387 | 95.319 | 1046.5 | 1024.6 |
| 4 | 168.879 | 151.530 | 171.307 | 153.389 | 1046.7 | 1025.7 |
| 32 | 692.160 | 604.091 | 698.161 | 611.395 | 1046.5 | 1025.6 |
| 128 | 2741.907 | 2231.680 | 8354.041 | 2499.199 | 1045.8 | 1026.0 |

Latency is the median of three per-process medians; p95 is nearest rank over 90 trials. RSS is the median process high-water mark, including load. Rust had lower median latency and RSS in every cell. Batch-128 timing varied materially: Python process medians were approximately 2328–3066 ms and Rust's 2034–2298 ms; the capture does not establish the cause, and the Python pooled p95 is substantially higher than its median.

| Batch | Python MLX peak / active / cache MiB | Rust MLX peak / active / cache MiB |
| ---: | ---: | ---: |
| 1 | 609.1 / 479.3 / 581.9 | 606.6 / 490.1 / 583.5 |
| 4 | 975.7 / 479.3 / 2401.8 | 971.5 / 520.9 / 2396.5 |
| 32 | 2528.9 / 479.3 / 15809.3 | 2438.4 / 611.6 / 14904.3 |
| 128 | 2539.3 / 479.3 / 24596.4 | 2427.4 / 477.6 / 22148.5 |

These are medians of three immediate post-trial observations; peak is MLX's active-allocation high water, not physical unified memory or the sum of active and cache. Rust's peak is lower in all cells, but **immediate active remains higher at batches 1, 4 and 32, and cache is higher at batch 1**. Those are unresolved memory cells. At batch 32, Rust active varied between 477.6 and 611.6 MiB across processes; delayed lifecycle cleanup is a hypothesis, not an established attribution. An output-lifetime diagnostic is pending.

The 384 approximately two-second CPU activity samples included WindowServer, herdr and other desktop processes; GPU activity was not directly measured and the machine was not exclusively reserved. No GPU clock or power setting changed. Sources, artifacts, executable and native-library hashes remained stable during the run. The ignored local capture is `reports/phase4-tinystories-mlx-hybrid-bench-v2/`; `manifest.json` SHA-256 is `e5717b665214238385001c1b72ea48079799495fe130f07e73fc9da42a85e564`, and `summary.json` SHA-256 is `46c562df38e5c30fed5b1c8a1865c03e7052c288beb736722eb1b6253695ecb3`. Full commands, source identities, per-process results and activity are retained alongside them.

## Output-lifetime development test

A separate single-process diagnostic dropped unused decoder logits and output tensors after host readback, then evaluated a small GPU barrier before stopping the request timer. This explicitly includes the cleanup barrier in Rust's timing. With the default cache, batches 1, 4 and 32 measured 94.13, 152.25 and 606.16 ms, and each immediate active snapshot was 477.6 MiB. Cache was 595.2, 2439.5 and 15038.3 MiB respectively. With an explicitly process-local 512 MiB cache limit, the same batches measured 94.29, 143.69 and 680.26 ms, with active 477.6 MiB and cache 512.1, 512.7 and 486.5 MiB. The 512 MiB batch-32 timing was slower than the earlier Python 658.031 ms median, although faster than the noisier 692.160 ms median in this run. These unpaired, one-process results do not establish a robust gate; raw data and before/after activity snapshots are in ignored `reports/phase4-tinystories-mlx-lifecycle-dev-v1/`.

A proposed dedicated-process comparison uses the measured cleanup sequence for every Rust request, a 512 MiB process-local cache cap at batches 1 and 4, and the default cache limit at batches 32 and 128. The policy is selected explicitly in the benchmark, recorded per process and does not modify the backend's library default or system settings. It must be repeated and evaluated as that scoped workload; it is not evidence for a long-lived multi-model application or concurrent requests.
