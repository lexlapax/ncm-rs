# Exact GELU scalar reuse diagnostic: deferred

Production source matches the Linux-qualified `5200924` candidate, and the
qualified CUDA benchmark, trace and lifecycle executable hashes were rechecked.
This documentation-only experiment has no production change. After the Mac's
GitHub fetch timed out, the clean checkout fast-forwarded from a verified bundle
of already-pushed `86ecc62`; the isolated diagnostic built in its own worktree
and target. The [plan](mlx-gelu-scalars-plan-v1.md) was fixed before running.

Three fresh Metal processes each compared the existing exact-erf expression
with fresh versus reused immutable scalar arrays in alternating order. All
finite outputs were bit-identical before timing. Each timing has five warmups
and 30 trials; p95 is nearest rank. Ratios are reused/fresh; lower is faster.

| Process | Batch | Fresh median ms | Reused median ms | Median ratio | p95 ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.33842 | 0.33921 | 1.0023 | 1.0181 |
| 1 | 2 | 0.34154 | 0.34152 | 0.9999 | 1.0155 |
| 1 | 4 | 0.36492 | 0.36469 | 0.9994 | 1.0140 |
| 2 | 1 | 0.33925 | 0.33642 | 0.9916 | 1.0103 |
| 2 | 2 | 0.34629 | 0.34348 | 0.9919 | 0.9867 |
| 2 | 4 | 0.37096 | 0.37073 | 0.9994 | 1.0129 |
| 3 | 1 | 0.33469 | 0.33879 | 1.0123 | 1.0192 |
| 3 | 2 | 0.34506 | 0.34383 | 0.9964 | 0.9816 |
| 3 | 4 | 0.37302 | 0.36350 | 0.9745 | 0.9504 |

There is no consistent material gain in this microbenchmark: B4 median ratios
are 0.9994, 0.9994 and 0.9745, and p95 ratios are 1.0140, 1.0129 and 0.9504.
Do not add backend state or integrate a runtime change on this evidence. The
experiment did not measure the complete model, test nonfinite values, establish
trained parity, or assess lifetime/memory tradeoffs. Two activity samples were
collected; they are too sparse to characterize competing GPU work. No thermal
warning was recorded. This is not an acceptance gate and changes no prior result.

The diagnostic binary SHA-256 is `1101b8aeecb345f7ea9f6a3a931402b39066adf804d0ee6378a56a6eb201e921` and source
SHA-256 is `8c29cbace3a82fe4a768be05dc9d84414f401e2a64774455a47d51e732d3ece7`. The [reports, commands, activity and runner](raw/mlx-gelu-scalars-poc-v1.tar.xz)
have SHA-256 `ffa87fc53b49fd43dedeebb50c87c164696bc17dc0ceda5c0f309665adcdd4f3`.
