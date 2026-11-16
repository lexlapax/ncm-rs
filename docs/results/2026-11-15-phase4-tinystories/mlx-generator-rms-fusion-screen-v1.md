# Generator RMS weight-fusion screen

The isolated [RMS fusion candidate](mlx-generator-rms-fusion-plan-v1.md) applies the previously studied MLX weighted-RMS primitive on top of hybrid generator normalization. Focused CUDA Sudoku and strict-cuDNN TinyStories output checks passed first. On real Metal, the weighted-versus-broadcast RMS operator test passed, and the seed-42 batch-128 sample was byte-identical to the retained hybrid output in codes, raw IDs, text and re-encoded tokens. This is one seed and does not establish exhaustive parity or quality.

One fresh process per binary and batch ran five warmups and 30 timed generation/readback requests with the same explicit 4,096 MiB process-local cache, output drop and GPU barrier. Baseline ran first at batch 32; candidate ran first at batch 128. This is an exploratory old/new comparison, not a Python performance gate.

| Batch | Baseline / candidate median ms | Baseline / candidate p95 ms | Baseline / candidate RSS MiB | Baseline / candidate MLX peak / active / cache MiB |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 566.36 / 546.99 | 578.44 / 568.72 | 1020.80 / 1020.47 | 2438.68 / 477.63 / 4220.15; 2438.96 / 477.63 / 4095.30 |
| 128 | 1965.19 / 1967.38 | 2019.35 / 2012.56 | 1019.58 / 1019.22 | 2490.48 / 477.63 / 4009.61; 2490.48 / 477.63 / 4093.98 |

The candidate was about 3.4% faster at batch 32 in this screen, with a 0.28 MiB higher MLX peak; batch-128 median and peak were effectively unchanged, and its cache snapshot was higher. No request exceeded four seconds. This does not establish a general speedup, explain or fix the previously observed tail, or establish a four-batch performance pass. The candidate is not integrated. RSS and MLX memory counters have different scopes, and the cache limit is not a hard instantaneous bound.

The runner recorded 89 approximately two-second CPU activity snapshots without errors; direct GPU activity and exclusive reservation were unavailable. Watched sources, inputs and binaries were stable. Copied reports matched the Mac originals: weighted-RMS test log SHA-256 `6bc7d17049cff0c46cf7ca7cceb9156d98765be1502fd8aeb022e365cfb2f9e8`, build log `15628fb2972270b5327105fdbfb31d1c3a2ec13818759f446b3360afc0523c6a`, seed-42 output `0cc0f03bedd1927c2fc4bc4817768c3c6f3232ab50a3224da4c49d93bdce8f67`, preflight fingerprints `fec26697a6ae5e0d41ae0095dea5b4c157b3e44100ad0a5a2785c3f8d464c9f7`, comparison manifest `5e9cd913884f127ae8729f4938db4ad893e4ea104f6f602a8638d2feb03414aa`, and activity `df1cf121f416ebc414e0d0ee2940980455cc1bd983e47b01c2dbcf309a9dd570`. The corresponding ignored files are under `reports/phase4-fused-rms-mlx-{test-v1.log,build-v1.log,identity-v1/,ab-v1/}`.
