# TinyStories Metal 4 GiB cache exploration

A process-local 4,096 MiB MLX cache limit was tried separately at batch 32 and 128 using the existing lifecycle benchmark. Each Rust process performed five warmups and 30 timed generation/code/ID-readback trials, including output drop and a GPU barrier. The same Rust binary and native MLX library hashes were checked before and after each run; no library default or system setting was changed. This was one process per batch, not paired or interleaved with Python, and is not an acceptance run.

| Batch | Median ms | p95 ms | Max ms | MLX peak / active / cache MiB |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 616.86 | 628.30 | 630.25 | 2438.99 / 477.63 / 4221.75 |
| 128 | 2076.91 | 2088.63 | 2092.58 | 2427.45 / 477.63 / 3964.15 |

The earlier [three-process comparison](mlx-lifecycle-performance-v3.md), using a default Rust cache at these batches, had median Rust cache snapshots of 15,038.30 and 22,148.49 MiB and median latency 607.75 and 2,119.25 ms respectively. The new results suggest a substantially smaller retained cache may be feasible, but runs were not paired and do not establish a causal speed or p95 improvement. They do not invalidate the earlier slow batch-128 process. Only before/after CPU process snapshots were collected here; GPU activity and intervening CPU activity were not observed. Full paired runs are needed before drawing a performance conclusion.

The batch-32 snapshot exceeded the requested 4,096 MiB. The [MLX cache documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_cache_limit.html) describes reclamation on a subsequent allocation; the tested [MLX 0.32.2 allocator source](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/allocator.cpp) admits a freed buffer when the existing cache is below the limit without including that buffer's size in the check. This is consistent with [MLX issue 4841](https://github.com/ml-explore/mlx/issues/4841). It explains why the setting is not a hard instantaneous cache bound, but does not establish the cause of the latency spike or rule out other memory consumers.

Raw local and Mac reports are `reports/phase4-tinystories-mlx-cache4g-b32-v1/` and `reports/phase4-tinystories-mlx-cache4g-probe-v1/`. The output JSON SHA-256 values are `ce336940b642d07583b3299bb2a9d8c893519f40d50cf25f478a91bde987b6d7` and `3089ce2be91ba53d2ce794bfea4f06a2b624c8b28cf99ded37bc17a59553a55b` respectively, verified against the Mac originals. Both used source commit `45b5252`; the benchmark binary SHA-256 was `4dd30377fdab4e8c819aee0b343bedb711da480febb667779886240938dbf09f`. These exploratory observations are not evidence for the CLI or a multi-model service.
