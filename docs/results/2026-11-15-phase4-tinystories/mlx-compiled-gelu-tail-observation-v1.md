# Alternating-runtime GPU activity diagnostic

The [planned diagnostic](mlx-compiled-gelu-tail-observation-plan-v1.md) ran batch 128 in the same Python/Rust, Rust/Python, Python/Rust order as the earlier uninstrumented comparison. Three fresh processes per runtime each ran five warmups and 30 generation/readback requests. Rust used the timestamped compiled-GELU binary with a process-local 4,096 MiB cache, output drop and GPU barrier. CPU and system-wide GPU polling and request timestamps can perturb timing; this is not a performance gate.

| Runtime / process | Median ms | p95 ms | Maximum ms | Requests over 4 s |
| --- | ---: | ---: | ---: | ---: |
| Python 0 | 2185.5 | 2245.4 | 2254.6 | 0 |
| Rust 0 | 1743.3 | 1768.2 | 1769.3 | 0 |
| Rust 1 | 1744.1 | 1760.3 | 1762.2 | 0 |
| Python 1 | 2166.9 | 2743.7 | 4255.0 | 1 |
| Python 2 | 2323.4 | 4867.1 | 6008.0 | 3 |
| Rust 2 | 1745.3 | 1759.3 | 1765.4 | 0 |

The marked Rust processes did **not** reproduce the earlier long tail. Four Python requests exceeded four seconds in this separate instrumented run. This shows that a slowdown can also occur in the Python workload, but does not establish that it has the same cause as the earlier Rust cluster. The [uninstrumented compiled-candidate comparison](mlx-generator-compiled-gelu-python-performance-v1.md) remains a batch-128 failure.

All 90 measured Rust request timestamp triples passed the monotonic ordering and 20 ms wall-duration consistency checks; the largest measured disagreement was 0.0072 ms. Of 30 requests per Rust process, 25, 26 and 25 had at least one GPU snapshot respectively. The associated reported system-wide device-utilization values were 96–98%. CPU snapshots identified the third Python process in 74 samples; within that approximate process window, 72 GPU snapshots reported device-utilization values from 38 to 99, with the test process as last submitter. Python requests were not individually timestamped. GPU registry counters have an uncalibrated averaging window, do not measure per-process GPU time and cannot establish cause or exclusive use.

The runner collected 241 CPU and 241 GPU snapshots without observer errors; watched sources and inputs were stable. Evidence copied from the Mac matched the originals: build log `reports/phase4-gelu-wall-mlx-build-v1.log` SHA-256 `b9400e610579c02682fe1cfb53402fe745883ebe2deef7a452d826dc780af6b1`; summary `reports/phase4-gelu-wall-python-diagnostic-v1/summary.json` `a9dc0a2869284ef3e977704678307aef7d4897be5864f6f6715efac1714b6bd8`; manifest `d7016c6039cdac9343f550d2efca6ad9f535f1d59f5ce184783104cd34834b46`; CPU activity `de773de41880febc59cc18f638f2bb4632c2b22eb96a79bcbdda9d2e93a0afb2`; GPU activity `b2c577a90187ae095bea8f18153566b25ddb410dad1a98ccb55865abf49dd8e3`; alignment files for Rust processes 0/1/2 `179be7ee7427ba990f8ef87094818f6fdb559a715c73bf9602fa3bdf6cba888b`, `a61919acd1bedc8b40a2af9e4973904e9381cb63f5bae930121fc8d3477a0938` and `ddb9d583afd90fd1b78b7f8b48df84fc7586d7a4682df109add7f31743364b05`. The latter files are in the diagnostic directory.
