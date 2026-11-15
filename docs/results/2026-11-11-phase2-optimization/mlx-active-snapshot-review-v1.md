# Retained MLX active-allocation snapshots

The existing 78-cell performance audit checks timing and process RSS/HWM. Its
20 memory cells do not adjudicate the separate 1.05 comparable steady-state
live-memory limit in [acceptance-v1.json](acceptance-v1.json). The retained full
codec reports also record MLX active-allocation bytes immediately after each
scope's 35 calls. An [offline review](mlx-active-snapshot-review-v1.py) checks
all report hashes against both original control audits and retains every scope
and process; it performs no new GPU work.

For the B4 resident block, candidate active snapshots ranged 59,970,824–
69,702,944 bytes in the earlier nine-process control and 67,097,876–69,702,944
in the quiet control. In both, the maximum candidate/previous-Rust snapshot
ratio was **1.058744** and candidate/Python was **1.162296**. The median ratios
were the same in these samples. For B2, candidate/previous maximum ratio was
1.038120 and candidate/Python 1.089894 and 1.083068. The script prints B1,
all remaining scopes, all ranges and both maximum and median ratios; none are
excluded. The current 18-process gate must be analyzed independently.

These are **not a declared steady-state memory gate failure**. Both Rust
benchmarks retain a model and input latent and drop each output before sampling;
the Python harness has corresponding operations, and all describe the same
MLX active-memory counter. However, the reports do not prove equivalent live
object graphs or that temporary buffers and asynchronous cleanup have settled.
Candidate and baseline implementations deliberately differ in prepared state and
execution graph, and Python lifetime semantics differ. Active allocator bytes
are not host RSS or separate VRAM. A matched post-drop/synchronization comparison
with explicit retained roots is needed to establish whether the samples represent
comparable live memory and to evaluate the 1.05 policy. Do not infer a pass or a
failure from these snapshots alone, and do not conflate them with the 20 RSS/HWM
cells that passed in the original audits.

Reproduce from the root with the pinned reference environment:

```sh
.reference/venv/bin/python docs/results/2026-11-11-phase2-optimization/mlx-active-snapshot-review-v1.py
```

The [candidate-only 384-request lifetime observation](mlx-memory-addmm-v1.md)
shows active MLX bytes return to zero after each model drop, but it is not a
matched live-memory comparison. The earlier gate failures and all raw data are
unchanged.
