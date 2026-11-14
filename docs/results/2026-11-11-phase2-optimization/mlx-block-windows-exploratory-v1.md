# Exploratory B4 p95 across both Apple control windows

The earlier and quiet controls each contain nine counterbalanced process
triplets with 30 measured B4 resident-block trials per process. Their native
binary hashes, MLX library hashes, harness, Python environment, OS, compiler,
workload declarations and codec source inventories match. Earlier control
activity included wallpaper; the quiet control's sampled top processes did not.
Both outcomes and their original analyses remain authoritative for their own
predeclared gates.

As an explicitly **post-hoc diagnostic**, the adjacent script pools all 18
processes per runtime and resamples paired process indices independently within
each window. It uses 10,000 bootstrap draws and a fixed seed. No process or
trial is removed. For candidate p95 over previous Rust, the pooled point ratio
is **1.051825** and the descriptive 95% percentile interval is
**[0.923895, 1.098714]**. Against Python the corresponding values are
**1.005722** and **[0.893834, 1.085129]**. The earlier and quiet controls'
separate Rust-comparison interval upper endpoints were **1.13695** and
**1.10442**.

This illustrates how process sampling and the choice of window can change the
interval for the same binaries. The pooled analysis was not predeclared and
cannot close or replace either gate. Sequential processes do not necessarily
experience the same machine state; the bootstrap assumes the paired process
units are suitable resampling units within each window. Two observed windows
cannot establish general stationarity, isolate background activity, or give a
simultaneous guarantee across the full 78-cell matrix. The pooled upper bound
is also close to the unchanged 1.10 limit. No claim of a reliable p95 pass or
causal explanation follows.

For any later measurement redesign, specify the process count, window sampling,
pairing, stopping rule and treatment of all 78 cells *before* collection, and
retain all prior outcomes. Do not change the 1.05/1.10 limits or substitute
this exploratory interval for the fixed controls. Meanwhile the
[marked trace](mlx-block-signposts-v1.md) motivates investigating the delay
between encoding and GPU start rather than assuming a slow model algorithm.

Reproduce from the repository root with
`python3 docs/results/2026-11-11-phase2-optimization/mlx-block-windows-exploratory-v1.py`.
