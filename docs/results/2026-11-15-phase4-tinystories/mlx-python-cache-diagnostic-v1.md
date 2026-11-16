# Python MLX cache-cap slowdown diagnostic

The [diagnostic](mlx-python-cache-diagnostic-plan-v1.md) repeated three alternating batch-128 processes per runtime with Python's process-local free-cache limit set to 4,096 MiB and the same request markers and system-wide GPU observer. Rust retained its existing 4,096 MiB lifecycle policy. A separate small smoke check found capped/uncapped codes and IDs identical; it is not a full quality test. This run is diagnostic and does not replace the default-policy Python comparison.

| Runtime / process | Median ms | p95 ms | Maximum ms | Requests over 4 s | Final MLX cache MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Python 0 | 2437.7 | 2458.8 | 2463.6 | 0 | 4320.0 |
| Python 1 | 2886.2 | 17914.3 | 17915.5 | 10 | 4253.1 |
| Python 2 | 2437.4 | 2482.9 | 2486.1 | 0 | 4227.2 |
| Rust 0 | 1692.6 | 1709.3 | 1714.1 | 0 | 4091.0 |
| Rust 1 | 1695.8 | 1710.6 | 1710.8 | 0 | 4087.4 |
| Rust 2 | 1921.7 | 2130.7 | 2313.5 | 0 | 4016.2 |

Python process 1 reproduced the severe plateau despite the cap: ten requests exceeded four seconds and five consecutive requests took 17.4–17.9 seconds. The cap is a free-cache reclamation threshold, not a hard instantaneous limit; the reported final cache counters can exceed it. For plateau trials 9–12, all 35 aligned GPU snapshots reported 39% device utilization; renderer utilization was 12–22%. Their last-submitter snapshot identified the Python process 25 times, WindowServer eight times and another process twice. This remains a system-wide snapshot, not attributed GPU time.

During those four requests, global `Alloc system memory` was 21.44–23.10 GiB and `In use system memory` 2.95–5.51 GiB. In ten later normal requests, those fields were 21.53–23.03 GiB and 5.96–7.55 GiB. Approximately aligned `ps` snapshots reported median Python CPU of 8.7% during the plateau and 52.0% during the later normal requests. Sampling windows, cache accounting and CPU time are not calibrated to prove where work waited. Compared with the [uncapped diagnostic](mlx-python-tail-gpu-v1.md), this shows a large uncapped Python free cache is not necessary for this observed plateau; it does not prove the cache has no effect on frequency, identify a shared cause with the earlier Rust plateau, or establish a runtime fix. The runs were not paired under identical external activity.

All marked Python requests had at least one GPU snapshot and passed the existing clock-alignment check. Each observer collected 268 snapshots; source and inputs remained stable and no observer errors were reported. Polling and markers may perturb timing, GPU counters are global, and no exclusive reservation was established.

Copied ignored evidence `reports/phase4-pycache-gpu-v1/`: summary SHA-256 `59f227a91c553c5208421bd5232f61f0df3d88646b2541e8a51f8f40e8a1a7ac`, manifest `3728b4e3ad6deb94c47c7b7be1e1a810e7ae02b6ca7b610dea2be6057571be7e`, CPU activity `1b5b2b3dea96407ecb2b90d9b0571232e2f9d5ef1f0afe1416daabf16bc67517`, GPU activity `30b0b976f000d7869a8e3a1b47de7d34d674062ccdea19f7e957e6e4bab43ec5`. Locally generated Python alignment hashes in process order are `a107bbb36f664fb1d9cb613e872013f950ef40e8b7a1ab1a1db1b72d8e882aa5`, `b58374c426c141328cc099ad896c2f10810fee9aa20dbb52fe7b8b2e374e8d5d`, `22eaafe27deea51f78cda6335ac343e60639896aac5946f8ae6e6bb9206f7a7cf8`.
