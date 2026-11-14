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

An additional offline inspection joined the overlapping GPU intervals to the
Metal submission table by command-buffer ID. The matching submission records
begin **before** the respective block markers: 0.6904 ms before the slow marker
and 0.7406/0.7314 ms before the two near-median markers. Another submission
record overlaps each marker, so simply attributing all overlap to the current
block would be misleading. Two nested target-process driver intervals also
overlap each marker; their approximately 0.06–0.07 ms durations are similar
across these three trials, but their identity was not bound to the GPU command
buffer. These observations narrow the question; they do not prove where the
additional wait originated.

The matched Metal application encoding interval ended approximately 0.144 ms
after the block marker in all three examples. GPU activity began at 0.562 ms
in the slow trial versus 0.346/0.341 ms in the two near-median trials. The
elapsed time between these recorded boundaries therefore accounts for much of
the observed difference in this one trace. This is an interval between trace
events, not proof of a particular queue, scheduler or competing workload. The
matched command-buffer completion records occur after the block markers end;
their timestamps must not be used to redefine the measured wall time.

The [report, target-process marker, GPU, application, submission and driver tables, selected completion events, analyses and outputs](raw/mlx-block-signposts-v1.tar.xz)
have SHA-256 `d741b06579e3f5d35097bbc619b1f5b6a333429690800207ae5bcda38c2c5246`.
Extract the archive in a temporary directory and run `python3 analyze.py` and
`python3 analyze-submissions.py` and `python3 analyze-app.py` to reproduce the
figures. The full system trace, which includes other
process metadata, remains on the Mac at `.reference/phase2-mlx-signposts-trace-v1.trace`.
The exported GPU table contains only the target process; the retained marker
table contains the target's diagnostic markers. No clocks or power settings
were changed.

The next useful investigation should distinguish host submission and scheduling
from GPU queueing with a trace that includes those boundaries, and check whether
the same pattern recurs before selecting an optimization. The Apple gate remains
77/78 after the quiet-window control; no tolerance or prior failure is changed.
