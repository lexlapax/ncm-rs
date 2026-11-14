# System-wide Metal activity diagnostic: incomplete

The [single predeclared diagnostic](mlx-system-activity-plan-v1.md) started its
system-wide recorder and then the marked target. The target exited successfully;
its source, manifest, workload, and scope input/output hashes match the prior
marked diagnostic. Its B4 block had a 0.9307 ms median, 0.9708 ms maximum and
no measured trial above 1.2 times its process median. It is an instrumented
diagnostic, not an acceptance result.

The recorder reported reaching its 35-second recording limit, but did not exit
within the runner's 40-second wait after target completion. The runner then
terminated it. The resulting trace cannot be exported (`Document Missing
Template Error`), and no claim about GPU overlap or host scheduling can be made.
The exact reason the recorder had not exited is unknown; it may have been
finalizing the system-wide trace. No recorder or target process remained at the
subsequent process check. The incomplete trace and system log remain private on
the Mac. No repeat was run.

The runner now records a `saving` state and waits for recorder completion
without an arbitrary post-target deadline; the recording duration itself stays
fixed at 35 seconds. A focused mocked regression checks that finalization is
entered without the former timeout. This change has not been validated in a new
system-wide recording and does not establish that any future trace will succeed.
The [target-only report](raw/mlx-system-activity-failed-v1.tar.xz) has SHA-256
`3ceb6e8908e6c10d907001898073d64f092219157d3c4839e774a1e99521d98f`.
