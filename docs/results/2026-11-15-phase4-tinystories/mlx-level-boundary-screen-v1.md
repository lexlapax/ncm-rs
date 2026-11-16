# Generator level-boundary Metal screen

The isolated [level-boundary experiment](mlx-level-boundary-plan-v1.md) synchronously materializes cumulative state during each generator level on MLX; the CUDA hook remains a no-op. The focused CUDA Sudoku and strict-cuDNN TinyStories checks passed first. On the Mac, the isolated release binary built successfully, and a seeded batch of 128 matched the retained hybrid candidate byte for byte in codes, raw IDs, text and re-encoded tokens. The output SHA-256 was `0cc0f03bedd1927c2fc4bc4817768c3c6f3232ab50a3224da4c49d93bdce8f67`; watched source, binary and reference hashes were stable. This is a single seed and does not establish complete numerical parity or quality.

An exploratory old/new comparison then ran one fresh process per binary at batch 128, candidate first, each with five warmups and 30 timed requests. Both used the same 4,096 MiB process-local cache limit, output drop and GPU barrier within timing. This compares the experiment to the prior hybrid candidate, not Python.

| Binary | Median / p95 / max ms | Peak RSS MiB | MLX peak / active / cache MiB |
| --- | ---: | ---: | ---: |
| Hybrid baseline | 1979.93 / 2035.55 / 2042.12 | 1019.62 | 2490.48 / 477.63 / 3966.29 |
| Level boundaries | 2165.57 / 2192.35 / 2195.27 | 1010.28 | 2490.15 / 477.63 / 4093.87 |

The experiment was approximately 9.4% slower at the median and did not materially reduce MLX peak memory in this screen. Neither process had a request over four seconds, so the screen does not explain or fix the previously observed tail. The candidate is **rejected and not integrated**. One process per binary cannot establish an acceptance gate, and the order was not repeated. RSS and MLX counters have different scopes; the cache limit is not a hard instantaneous bound.

The runner recorded 73 approximately two-second CPU activity snapshots without errors; GPU activity and exclusive reservation were unavailable. Watched sources, inputs and binaries remained stable. The locally copied reports matched the Mac originals: build log `reports/phase4-level-boundary-mlx-build-v1.log` SHA-256 `d96af00bddc229dd088d136634e6cd5c0b2caaffb698755be623962bf17a5b56`; comparison manifest `reports/phase4-level-boundary-mlx-ab-v1/manifest.json` SHA-256 `52a111ba9b90c7d3e94452c3f1f5b7bbc359b82b5ab1b99dae10751066995615`; activity `reports/phase4-level-boundary-mlx-ab-v1/activity.jsonl` SHA-256 `8d6b8f4f2b77dbff7c11061cd3382e34905a4a1cfa950a2b1759435e9bafe2b2`; identity preflight `reports/phase4-level-boundary-mlx-identity-v1/before.sha256` SHA-256 `ba95dae1487ffee04446b2ba83233d84e4b5fc6702404730c13729d23e3e3dea`.
