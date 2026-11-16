# Independent MLX reconstruction comparison

For each of five retained CUDA traces, the Python MLX adapter independently recomputed all 12 reconstruction levels using the same 16 rows of forced code indices. The comparison used the existing accumulated diagnostic band and authenticated TinyStories artifact; it did not use the CUDA-copied reconstruction tensors in the merged adapted trace as MLX outputs.

| Seed | Outside-band reconstruction values, all levels | Outside-band at level 0 | Level-0 maximum absolute difference |
| ---: | ---: | ---: | ---: |
| 100 | 19 | 0 | 0.001464844 |
| 101 | 22 | 1 | 0.001464844 |
| 102 | 5 | 0 | 0.001464844 |
| 103 | 17 | 0 | 0.000976562 |
| 104 | 0 | 0 | 0.000976562 |

Across 47,185,920 compared reconstruction values, 63 were outside the band. Their per-level outside-band counts match the earlier Rust Metal-versus-CUDA summaries for every seed and every level. Matching counts do not establish identical error positions or a numerical pass. Level 0 already differs before the generator transformer; the first-level cascade and residual must be separated with same-input comparisons before attributing the cause. No tolerance changed.

Each diagnostic checks forced-code range, shape and finiteness, validates the artifact via the pinned adapter, and fingerprints source, artifact and reference before/after; all remained stable. The comparator itself is `tools/reference/phase4_mlx_cascade_diagnostic.py`. Retained ignored results `reports/phase4-mlx-cascade-independent-v1/seed-{100,101,102,103,104}.json` have respective SHA-256 values `a0bc5396788c91a9be1fec65fb1259c14e7ce53d7d088280a4575f67aa58afa8`, `a49d50e4c4a9c39ef989f03c958d12627868485146e4f7ce0a3c62d5ead0a536`, `5a1f05949477ede31a72a752fd63b5b4bda5da7401de964e1654175029ec430b`, `7cc3751395e8415d166ec8d835cc6565dcd3d4eb449fa6d9597c7d6747a29182`, and `4ae1998e3f954c3dfda6b6f59a2bcd546e6d90a7b6a3f8adf21fe56bc1e16eb6`, verified against the Mac originals.
