# XSum Metal top-1 corpus capture setup

The user resumed Mac work on 2026-11-16. The clean Mac checkout was fast-forwarded
from `1039e2c` to pushed `e14ae02`; the previously screened Metal CLI binary is
SHA-256 `498b70050edc69f7cc93255b9c681d3bc34ff628b0eb07b35f6e40223654cac2`.
The candidate owned loader is not present on the Mac. A new capture script at
`tools/bench/capture_xsum_mlx_quality.py` was copied to a new ignored Mac reports
path after its hash was checked; it records the dataset, artifact, source revision,
binary and script identities and launches the existing CLI for 32-row top-1
batches. Its first smoke attempt failed before inference because system Python
lacked `hashlib.file_digest`. The existing `.reference/mlx-venv/bin/python` worked.

The 32-row smoke matched every code and ID in the retained CUDA full-quality
capture. The receipt comparator also passed its focused mutation and disagreement
unit tests. A separate completed 32-row batch starting at row 2,560 also matched;
this is only a partial observation, not a quality result. The full 11,334-row
Metal capture subsequently completed; its [result](mlx-full-top1-v1.md) retains
raw code and ID mismatches and an exact-gate failure.
The subsequent [stochastic follow-up](mlx-stochastic-followup-v1.md) passed its
scoped bounds. Metal performance, memory and the owned-loader candidate checks
remain open.
