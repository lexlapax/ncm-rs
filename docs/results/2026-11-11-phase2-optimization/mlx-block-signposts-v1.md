# Marked Metal block trace

The [diagnostic plan](mlx-block-signposts-plan-v1.md) ran one isolated,
signpost-enabled process under Metal System Trace plus Points of Interest. The
shim compiled with warnings denied and the release example built against the
pinned MLX libraries. The executable SHA-256 is
`6e3c68c9beda39daab01d43f81b6b923319c52ba3377bd9b4e5605fbdb2796da`.
Its manifest, workload and all scope input/output hashes match the previous
block-segments diagnostic; among its source inventories only the intentionally
patched diagnostic example differs. The process exited successfully. This
instrumented diagnostic is not an acceptance measurement.

The exported trace contains two identical signpost event streams from the
instrument configuration; analysis uses one stream, not both. It verifies
108 nonoverlapping block sequences, batch labels and ordered begin/graph/end
markers. All 30 measured B4 marker durations agree with the report's enclosing
wall times within 0.15 ms and graph-plus-evaluation durations within 0.1 ms.
GPU overlap is the union of exported target-process GPU intervals clipped to
each marker, not a claim about every source of GPU or wall-clock time.

One B4 measured trial exceeded 1.2 times its process median. Its wall time was
**1.1964 ms** against **0.8914 ms** median. Its graph construction was
**0.0244 ms**, essentially the same as the five nearest-median trials
(0.0239–0.0248 ms); synchronous evaluation was **1.1604 ms** versus
0.8636–0.8691 ms. The slow trial had one overlapping GPU interval of
**0.4818 ms** versus **0.4103–0.4169 ms** for those comparison trials. Its
exported CPU-to-GPU latency was **0.4178 ms**, versus **0.1960–0.2010 ms**
for two near-median examples. These are descriptive measurements from one
instrumented process, not an explanation for all gate failures. In particular,
the trace does not establish whether the additional delay came from CPU
submission, scheduling, queueing, contention or another cause.

The [report, target-process marker and GPU tables, analysis and output](raw/mlx-block-signposts-v1.tar.xz)
have SHA-256 `8aab10b509a4212c786dea50b766f38bd1c1051ff9e4e3487c454a9db2166641`.
Extract the archive in a temporary directory and run `python3 analyze.py` to
reproduce the checks and figures. The full system trace, which includes other
process metadata, remains on the Mac at `.reference/phase2-mlx-signposts-trace-v1.trace`.
The exported GPU table contains only the target process; the retained marker
table contains the target's diagnostic markers. No clocks or power settings
were changed.

The next useful investigation should distinguish host submission and scheduling
from GPU queueing with a trace that includes those boundaries, and check whether
the same pattern recurs before selecting an optimization. The Apple gate remains
77/78 after the quiet-window control; no tolerance or prior failure is changed.
