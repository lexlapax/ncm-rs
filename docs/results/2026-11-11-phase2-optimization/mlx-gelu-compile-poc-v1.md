# Exact GELU compilation: isolated result

The [predeclared diagnostic](mlx-gelu-compile-plan-v1.md) ran three fresh
processes against pinned Metal MLX. It compares the backend's exact-erf GELU
operation order, with all arrays and scalar constants passed explicitly to
compilation. The reported maximum absolute difference was zero; the original
check did not explicitly reject non-finite output, so it does not establish
elementwise equality. Inputs are synthetic finite values, not trained-model
traces. A corrected finite-check diagnostic is prepared separately.

| Batch × 89 × 1024 | Pooled plain median | Pooled compiled median | Plain / compiled p95 |
| --- | ---: | ---: | ---: |
| 1 | 0.3406 ms | 0.3021 ms | 0.3619 / 0.3376 ms |
| 2 | 0.3337 ms | 0.2864 ms | 0.3624 / 0.3174 ms |
| 4 | 0.3611 ms | 0.3096 ms | 0.4065 / 0.3311 ms |

For B4, compiled/plain median ratios within the three processes were
**0.8406, 0.8531 and 0.9033**. These are small paired microbenchmarks, not
full-model results or a demonstrated improvement in block tail latency. No
desktop activity samples were collected; no thermal warning was recorded.
Constants were pre-created in both paths, so these timings do not include the
current backend's per-call scalar-array construction.

The first compiled B1 evaluation took **276.612, 19.285 and 19.196 ms** in
the three processes; later B2/B4 first calls took 0.396–0.453 ms. Persistent
framework/system compilation caches were not cleared, so this is not a
controlled cold-start comparison. First sampled MLX peaks were 0.75, 1.47 and
2.92 MB by shape; these counters are not host RSS or a cache-lifetime bound.
Compilation's first-use cost, bounded shape specialization, memory lifetime,
trained parity and full-model timing remain unresolved. No production change
or acceptance claim follows.

The executable SHA-256 is
`344bf9b9dc9ebe504de6b73a67385e7e7fb4cc7c39a98011203603f5749c48c8`;
the [source](mlx-gelu-compile-poc-v1.rs) and isolated worktree copy both hash to
`b8e1fc768fb77bce3ca0242426b4fdc213d9cac3749d6dde19bab86014124ce0`.
The [three raw reports, identities and thermal observations](raw/mlx-gelu-compile-poc-v1.tar.xz)
have SHA-256 `4b76d5473b384cd1edddf12740ca993a84e673ce27110e4aeddd2d57057d9a41`.
