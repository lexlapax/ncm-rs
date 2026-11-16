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
32. Chrome and other desktop GPU processes were present. Point-in-time `ps`
checks reported high Chrome CPU percentages, but those percentages are
process-lifetime averages and do not establish contemporaneous CPU usage,
causation or GPU exclusivity.

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

## Paired full-model follow-up

The previous binary was rebuilt from commit `531ea96` and matched its prior
benchmark binary byte-for-byte (SHA-256
`95ad92a52ae7f1d1353293517404d50eecbe9b1283eb6afe5d8e68f1034b7e86`). The
tie-enabled binary matched the measured binary (SHA-256
`e8d68bb0619f0f0041130e14528cc9ac695272644db2924ef88a298bd90e63d5`).
Under the same current workstation conditions, three fresh-process pairs per
batch alternated their order and matched all codes and IDs. Batches 1/4 used
five warmups and 30 measured trials per process; batch 32 used two warmups and
five measured trials per process as a shorter diagnostic.

| Batch | Previous process medians ms | Tie-enabled process medians ms | Median paired change |
| ---: | --- | --- | ---: |
| 1 | 157.49, 159.33, 159.43 | 158.32, 158.57, 159.40 | −0.02% |
| 4 | 575.22, 577.05, 579.02 | 572.85, 578.53, 578.13 | −0.15% |
| 32 | 4479.03, 4486.44, 4491.30 | 4482.11, 4487.91, 4491.33 | +0.03% |

The earlier 7–18% difference did not reproduce as a tie-change slowdown in
this direct control. This narrows the investigation but does not identify the
cause of the earlier occupied-workstation run; the shorter batch-32 control is
not a replacement for the full Python comparison. Input, binary, artifact
manifest and linked-library hashes matched before and after. The raw bundle is
`reports/phase4-xsum-tie-paired-control-v1/`; its manifest hashes to
`755f6afdee386183f12c4dcffa40e2384f8b065b4a4c5fdb6eeeb041a763c78`.
There were 357 GPU activity samples and 184 CPU/load samples, with no query
failures. CPU `ps` percentages are process-lifetime averages, not interval CPU
utilization; neither observer proves causation or GPU exclusivity. The temporary
source archive was removed after the byte-identical rebuild.

Reproduce with the two binaries and pinned inputs recorded in that manifest
using `tools/bench/run_xsum_tie_paired.py --output NEW_REPORT`. The baseline
and candidate binaries, and baseline build log, remain under
`reports/phase4-xsum-tie-paired-regression-v1/`.

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
