# XSum tie-enabled CUDA performance

The current `xsum-inference-v2` tie-enabled path was measured against pinned
Python CUDA with the official top-1/CFG-zero settings, three fresh processes
per runtime and batch, five warmups and 30 synchronized measured generations
per process. Every checked code and output ID matched within each batch and
across processes.

| Batch | Runtime | Median ms | Pooled p95 ms | Peak RSS MiB | Sampled driver GPU peak MiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Rust | 184.42 | 188.79 | 1060.65 | 838 |
| 1 | Python | 207.82 | 221.62 | 1860.62 | 990 |
| 4 | Rust | 607.44 | 639.75 | 1058.30 | 1062 |
| 4 | Python | 726.85 | 767.58 | 1860.25 | 1624 |
| 32 | Rust | 4727.40 | 5110.36 | 1060.84 | 3080 |
| 32 | Python | 5801.74 | 6221.67 | 1863.39 | 5966 |

Latency is the median of process medians, p95 pools 90 trials, and RSS is the
median whole-process peak including load; generation timing excludes load.
Python garbage collection is measured separately. GPU memory is the median
of one-second sampled NVIDIA driver per-process peaks; brief peaks can be
missed. There were 1,309 activity samples and no memory-query failures, with
7–11, 21–28 and 161–207 attributed samples per process for batches 1, 4 and
32. Chrome and other desktop GPU processes were present. Point-in-time CPU
checks also observed high Chrome CPU use; those observations do not establish
causation or prove GPU exclusivity.

Rust remains faster and lower in both listed memory measures than Python in
these cells. However, compared with the [previous three-process run](cuda-final-repeated-v1.md),
Rust median latency increased 18.2%, 7.3% and 7.3%, while Python increased
9.6%, 5.4% and 8.7%. The new run also had wider timing ranges. This is a
visible performance regression requiring investigation; the concurrent
activity is a possible contributor, not a proven explanation.

An alternating same-process selector screen used the old unique-max kernel
and the new tie-enabled kernel on identical finite unique inputs, ten warmups
and 100 measured calls each, including readback. Old/new medians were
15.60/15.59 microseconds for shape `[1,4,4096]`, 25.00/24.84 microseconds for
`[4,48,4096]`, and 78.35/77.87 microseconds for `[32,48,4096]`. Outputs matched.
This does not reproduce the full-model slowdown in the selector itself, but
does not exclude changes elsewhere or prove system activity caused it. The
screen did not capture separate activity telemetry.

The raw repeated bundle is `reports/phase4-xsum-reference-tie-repeated-v1/`;
its manifest hashes to
`e15774fc144777ac2400adc9b8ec633a955b283e513ab898d45b1d97874e1757`.
The timing summary hashes to
`8cc0888c31c28d3b7f0b048e4e69d88295c94629e6130122a4ea7e71eeca1a1d`, and the
sampled GPU-memory summary to
`b9c4e8258c837809fc78ed21c6d5e3d9b59a46e06e27ac359a672e16e8c0b7a3`.
Source, binary and library hashes matched after collection. The selector
screen is `reports/phase4-xsum-reference-tie-selector-screen-v1.json`, SHA-256
`b241b094ef337edec71b45fd8daf3b31e6e1c1c1ecf835a686653987dfb7a3cd`.

Use the [prior repeated collection command](cuda-final-repeated-v1.md) with a
new output path. Run `xsum_tie_probe --benchmark` with the strict-cuDNN build
and library path to repeat the selector screen. Full-model performance and
stochastic qualification, and XSum MLX, remain open.
