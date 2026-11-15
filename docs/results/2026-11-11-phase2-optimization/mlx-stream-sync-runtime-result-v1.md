# MLX stream synchronization runtime diagnostic

A source-isolated candidate added an explicit same-stream `mlx_synchronize` after each nonempty backend evaluation. Three alternating fresh process pairs ran the unchanged full-codec timing workload and the retained-root memory probe. Immediate B4 resident-block active bytes were 59,970,828 in all synchronized processes, versus 67,097,880 / 69,702,948 / 69,702,948 in the current-runtime processes. All models returned active bytes to zero after drop. Available token/code/output ID and latent hashes match; block floating-point parity was not qualified.

The cost is material. Batch-four resident-block median-time ratios (synchronized/current) for the three pairs were **1.1488, 1.1616 and 1.1033**. Batch-two block ratios were 0.9507, 1.2888 and 1.3210; batch-four reconstruction ratios were 1.2524, 1.2851 and 1.0488. Median of the three process medians, synchronized/current, for every workload:

| Batch | Scope | Ratio |
|---:|---|---:|
| 1 | encode | 1.0215 |
| 1 | decode | 1.0013 |
| 1 | reconstruct | 0.9852 |
| 1 | resident_decoder_block | 1.0320 |
| 2 | encode | 0.9284 |
| 2 | decode | 0.8649 |
| 2 | reconstruct | 1.0155 |
| 2 | resident_decoder_block | 1.3070 |
| 4 | encode | 1.0415 |
| 4 | decode | 0.9821 |
| 4 | reconstruct | 1.2588 |
| 4 | resident_decoder_block | 1.1566 |

These are exploratory timings, not a 78-cell audit; the Mac was not exclusive and 50 process-activity samples are retained. No thermal warning was recorded. First full-execution times ranged 27.084–29.058 ms current and 26.786–27.637 ms synchronized, with warm filesystem and system caches; these do not bound cold startup. First checked-load ranges were 173.754–178.281 and 173.931–177.382 ms respectively.

Because the wait adds a consistent 10–16% to the target B4 block in this comparison, the unconditional synchronization is deferred. This does not establish exactly which operation releases the allocations or whether a narrower mechanism can preserve speed. No production code, pinned dependency, tolerances, or existing gate changed; a trained float qualification was not run for the deferred variant.

[Plan](mlx-stream-sync-runtime-plan-v1.md), [patch](mlx-stream-sync-runtime-v1.patch), [review script](mlx-stream-sync-runtime-review-v1.py), and [raw reports, runner, hashes and activity](raw/mlx-stream-sync-runtime-v1.tar.xz), SHA-256 `9bdf504384b397c5bcbc07a2eef6e380c3d213dcb89ca2c902918686648b0d08`.
