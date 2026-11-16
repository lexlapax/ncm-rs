# Split-batch Metal screen

The isolated [two-by-64 prototype](mlx-chunk64-plan-v1.md) preserves each row's level-major random-word assignment and concatenates results in input order. The focused CUDA golden and strict-cuDNN TinyStories output checks passed before the Mac experiment. The Metal candidate built successfully and was compared with the existing unsplit hybrid binary at seed 42.

At batches 64, 127 and 128, all codes, raw IDs, decoded texts and re-encoded tokens matched. At batch 65, row 64 differed in eight code positions and two raw-ID positions, with changed decoded text and tokens. In that case the one-row second chunk also takes the hybrid's composed-normalization branch, while the unsplit batch takes the fused branch; this is a real implementation difference, not a claim that random words differ or that the exact source of the output difference has been proven. One seed does not establish general output identity or quality.

One fresh process per binary then ran batch 128, candidate first, with five warmups and 30 measured generation/readback requests under the same 4,096 MiB cache, output drop and GPU barrier policy. The candidate measurement includes both chunks and output concatenation. This is an old/new comparison, not a Python gate.

| Binary | Median / p95 / max ms | Peak RSS MiB | MLX peak / active / cache MiB |
| --- | ---: | ---: | ---: |
| Unsplit hybrid | 2002.10 / 2026.38 / 2030.16 | 1021.56 | 2490.48 / 477.63 / 4087.98 |
| Split at 64 | 2111.14 / 2170.45 / 2177.90 | 1020.70 | 2471.13 / 477.63 / 4067.24 |

The split was approximately 5.4% slower at the median and reduced MLX peak by 19.35 MiB in this screen. Neither process had a request over four seconds; this does not demonstrate a tail remedy. The broad split changes a batch-65 output, and the candidate is **not integrated**. One process per binary, with fixed ordering, cannot establish a performance gate or exclude a later tail. The earlier slow batch-128 results remain part of the record. RSS and MLX counters have different scopes, and the cache limit is not a hard instantaneous bound.

The runner captured 72 approximately two-second CPU activity snapshots without errors, with no direct GPU activity or exclusive reservation. Watched sources, inputs and binaries were stable. Reports copied from the Mac matched the originals: build log `reports/phase4-chunk64-mlx-build-v1.log` SHA-256 `d25de486f9a224705e8033b84666053f54c27651da3662eb7291abb7a758cd36`; boundary manifest `reports/phase4-chunk64-mlx-identity-v1/manifest.json` SHA-256 `28af8d3280b5e2a859f100420136d51fa596cf343bef39e57673dc32bef90f12`; comparison manifest `reports/phase4-chunk64-mlx-ab-v1/manifest.json` SHA-256 `8aa0f757d4393f4225c52e0e37bb442f2ce02722126155df797838d9a3c90bef`; activity SHA-256 `e32eb41bed1ffea169e97f2ae3a5919e968582e6b23b0353c25cb678b217c238`.
