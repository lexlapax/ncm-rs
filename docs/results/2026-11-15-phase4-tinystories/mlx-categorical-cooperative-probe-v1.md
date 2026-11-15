# Cooperative Metal categorical experiment

The isolated [experiment](mlx-categorical-cooperative-plan-v1.md) compared the current one-thread-per-row inverse-CDF sampler with a 32-thread-per-row candidate. The candidate loads a row and computes exponentials cooperatively, then retains the existing index-order total and prefix accumulation and full-word selection comparison. This was test-only code in an isolated source copy; no production sampler, RNG or model algorithm changed.

On real Metal, all outputs matched the existing sampler exactly across 256 and 16,384 controlled rows. These inputs included NaN, positive infinity, an all-negative-infinity row, a single finite final entry, and selected edge random words. This limited comparison does not prove equivalence on every input.

| Shape | Rows | Existing median / p95 ms | Candidate median / p95 ms |
| --- | ---: | ---: | ---: |
| 4 × 64 × 4096 | 256 | 1.650 / 1.685 | 1.807 / 1.852 |
| 128 × 128 × 4096 | 16,384 | 2.461 / 2.542 | 42.559 / 42.755 |

Each shape ran in one process, alternating the same inputs between kernels, with five warmups and 15 measured evaluations per kernel. These are isolated kernel-call timings including array evaluation, not end-to-end generation, a Python comparison or a performance gate. The candidate was approximately 17 times slower at the large shape in this diagnostic and was **not integrated**. The measurement does not establish why it was slower; no GPU kernel-level timing or continuous activity telemetry was collected. Before/after CPU process snapshots are retained; `mediaanalysisd` appeared in the after snapshot. No relationship to the kernel timings can be inferred.

The isolated source was archived from commit `cdd1a11` and built against native MLX 0.32.2. The patched sampler, probe source, test binary, MLX library and metallib hashes matched before and after the test. `reports/phase4-categorical-cooperative-probe-v1/test.log` has SHA-256 `bba5529d4ff286ee6d2bd9ba3d1291e428cc585ff5e66761e9c1c32dc8d0a819`; the source fingerprint file has SHA-256 `9ebfc8ea0feb9112e6ba48aabda75b1a5af8cceda01eef49297867a7b253bde6`. Both were checked against the Mac originals. The Mac fetched the reviewed pushed commit through a verified local Git bundle after its GitHub access timed out; the temporary transfer copies were removed.
