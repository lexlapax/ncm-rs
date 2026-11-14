# Bounded MLX inference and model-lifetime observation

On Mac Studio, diagnostic branch `3ff2f00` loaded and dropped three models and
ran 128 batch-4 reconstructions per model. Its production source matches
`2850331`; the branch only adds the diagnostic example and its Cargo entry.
All 384 requests matched the previously qualified input and output-ID hashes.
The example was built in release mode with the pinned MLX native libraries.

| Snapshot | Cycle 1 | Cycle 2 | Cycle 3 |
| --- | ---: | ---: | ---: |
| Active MLX bytes after load | 59,593,216 | 59,593,216 | 59,593,216 |
| Active MLX bytes after request 128 | 59,594,016 | 59,594,016 | 59,594,016 |
| Cached MLX bytes after request 128 | 104,407,712 | 104,407,712 | 104,407,712 |
| Active MLX bytes after model drop | 0 | 0 | 0 |
| Cached MLX bytes after model drop | 164,001,728 | 164,001,728 | 164,001,728 |
| Host RSS bytes after model drop | 275,759,104 | 307,822,592 | 308,805,632 |

The global MLX active-allocation peak was 155,557,200 bytes. Samples after
requests 1, 8, 32, 64 and 128 show a stable allocator range across this finite
workload, with no cache clear between requests or cycles. Host RSS rose across
cycles; these observations do not establish a general bound or explain host
allocator behavior. MLX active/cache counters and host RSS are distinct and
must not be added as separate VRAM. The observation is not a timing benchmark.

[Report, build log, compiler dependency file and binary/library hashes](raw/mlx-long-lived-memory-v1.tar.xz)
have SHA-256 `5639ee608fbdc83dcfaccdb3e828879ffe81b7022d9dab4cf2d9f0cf026c82b6`.
The report's source hashes match the clean branch, and its compiler dependency
file contains the model, backend, core and diagnostic sources. The executable
SHA-256 is `488a7c29ad9233f033a22513dd7c8c2449c6b85c60d202be1324e5b1487f6c28`.
The existing 75/78 Apple gate remains open; re-evaluate resource evidence if
later execution-affecting changes alter the tested runtime.
