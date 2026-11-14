# Exact GELU compilation with finite-output checks

The corrected [diagnostic source](mlx-gelu-compile-poc-v2.rs) explicitly rejects
non-finite values before comparing every output element. All three processes
completed for B1/B2/B4 × 89 × 1024, with a reported maximum difference of zero
against the uncompiled same-order expression. These synthetic inputs are not
trained-model parity.

| Batch | Pooled plain / compiled median | Pooled plain / compiled p95 |
| --- | ---: | ---: |
| 1 | 0.3435 / 0.3026 ms | 0.3680 / 0.3303 ms |
| 2 | 0.3489 / 0.2947 ms | 0.3700 / 0.3170 ms |
| 4 | 0.3665 / 0.3134 ms | 0.3885 / 0.3503 ms |

B4 compiled/plain process median ratios were **0.8569, 0.8432 and 0.8935**.
First B1 compiled evaluations took **22.496, 19.229 and 19.459 ms**; system
and framework compilation caches were not cleared after the prior experiment.
The earlier **276.612 ms** first-use observation is retained in the
[v1 result](mlx-gelu-compile-poc-v1.md). No desktop activity samples were
collected and no thermal warning was recorded. Both paths pre-created scalar
arrays; this is an isolated microbenchmark, not a full-codec or gate result.

The same-formula result justifies evaluating an isolated full-codec prototype,
but does not establish its benefit, model parity, bounded specialization or
acceptable first-use and memory costs. No production code or numerical policy
has changed. The executable SHA-256 is
`1a3e54f16bdabba3496852ace32811cfdcc0d26a7c30e7c7e6c136f672767ea7`;
both the source and isolated worktree copy hash to
`1ebce59256368f679e4629930ae1a52e622c60091e66bbe6a4fbbf0f38e60fad`.
The [raw reports, identities and thermal observations](raw/mlx-gelu-compile-poc-v2.tar.xz)
have SHA-256 `0985f4ddb1b0d95e36257467a6b2e1e17aa998d3ced0edd84b2942a5223c1775`.
