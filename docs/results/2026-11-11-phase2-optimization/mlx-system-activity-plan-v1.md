# One system-wide Metal activity diagnostic

The marked trace found one slow block with a longer interval between target
encoding completion and GPU start, but the target-only trace cannot show other
GPU work or host thread activity. Run one bounded system-wide Metal System Trace
with Points of Interest and Thread Activity around the same marked diagnostic.
The recorder must report readiness before launch, both owned processes must exit
successfully, and all expected target markers and output identities must be
verified. If the trace lacks a slow trial or the relevant GPU data, report that
limitation; no repeat or acceptance gate is authorized by this plan.

The trace contains system process metadata and remains private on the Mac.
Only narrowly reviewed target timings and aggregate overlap facts may be
published. No applications, clocks, power settings, pins or model code change.
The instrumented run is not acceptance evidence and cannot supersede previous
gate failures. Compare intervals, not assumptions about cause, and do not
attribute unrelated GPU work to the delay without temporal overlap evidence.
