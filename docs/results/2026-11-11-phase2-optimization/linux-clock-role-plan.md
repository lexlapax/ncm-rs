# Clock diagnostic role plan

`clock_role_plan.role_plan` constructs an explicit set of six finalizer role
receipt declarations, the sibling observer-supervisor command, and the GPU
telemetry command from a supplied non-launch-ready snapshot, GPU UUID and
coordinator PID. It is pure command construction: it runs no command, inspects no
process, reads no files and does not authorize clock changes. The actual entry
and live role handlers have not been implemented.

The plan fixes coordinator-related roles on CPU0; the CPU sampler starts on CPUs
0 and 2 and performs its existing transition to CPU0. The restorer has the tested
660-second outer limit and 60-second termination grace; set/reset each have the
existing 10-second limit, the worker 600 seconds with its separate 590-second
absolute work deadline enforced by restoration, the observer worker 950 seconds,
and CPU telemetry 930 seconds. All capture/report limits match existing helpers.
The plan includes one requested 1710 MHz lock and one reset command as data;
only the existing restoration supervisor may later perform the allowed cleanup-only
retry. It does not infer supported clocks, permissions or successful restoration.
Each role carries the GPU UUID and the fixed private `snapshot.json` path under
the launch directory, enabling future handlers to recompute the same plan from
the snapshot digest and explicit arguments. This convention is not proof that
the private copy is authentic.

The file snapshot must include the future entry in its tool inventory; existing
historical snapshot v1 predates that entry and cannot satisfy this requirement.
A future entry must independently authenticate the snapshot digest and origin,
validate role commands, parents/images/environments and safe sibling topology,
repeat current preflight, and establish GPU identity, permissions, restoration
and a fresh quiet window. The plan's own flags deliberately remain false. It does
not prove safe live ancestry, actual reaping, telemetry coverage, hardware state,
complete preparation or performance acceptance.

```sh
umask 077; PYTHONPATH=tools/bench python -m unittest test_clock_role_plan test_clock_control_evidence test_clock_diagnostic_coordinator -v
```

Nineteen focused tests pass, including command budgets, CPU affinity declarations,
isolation, scope/path/environment/UUID failures, existing receipt binding and
synthetic coordinator cleanup behavior. No clock, GPU, workload or telemetry
collection was run for this component. Focused-test log:
`/tmp/ncm-phase2-clock-role-plan-focused.log`.
`umask 077; CARGO_BUILD_JOBS=4 ./tools/check.sh` passed all default checks,
including 420 reporting tests; its local log is
`/tmp/ncm-phase2-clock-role-plan-check.log`. That run began before the private
snapshot path and UUID argument refinement, which then passed the two focused
`test_clock_role_plan` tests separately.
