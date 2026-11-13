# Read-only GPU preparation component

`prepare_clock_gpu.inspect` uses two bounded metadata probes to verify one GPU's
UUID, PCI address and name against supplied reviewed expectations and to parse its
reported supported graphics-clock rows. It rejects an absent 1710 MHz row,
malformed units/rows, mismatched identity, multiple visible devices, contaminated
metadata environment and a changed NVIDIA utility identity. It checks the utility
before and after the probes. It does not invoke a lock or reset, import Torch, or
start a workload. Capture and receipt files remain in a fresh private output folder.

The reported list is application-clock data. The result explicitly leaves GPU
lock support/permission, enforcement, policy, availability and launch readiness
false. It is not a quiet-window or restoration check. Utility pathname/hash checks
are not live process authentication or proof of loaded library bytes. The future
entry must generate and bind this record itself, recheck the device at launch, and
separately establish safe privileges and restoration.

```sh
umask 077; PYTHONPATH=tools/bench python -m unittest test_prepare_clock_gpu test_prepare_readonly_launch test_clock_role_plan -v
```

Sixteen focused tests pass, including three new metadata-component tests with
stubbed command results; no real GPU query occurs in those tests. The component
itself has not yet been exercised against the actual GPU. The separately executed
read-only identity and supported-clock queries are documented in the
[role plan](linux-clock-role-plan.md#read-only-gpu-identity-and-supported-clocks).
`umask 077; CARGO_BUILD_JOBS=4 ./tools/check.sh` passed all default checks,
including 429 reporting tests; local log: `/tmp/ncm-phase2-gpu-preparation-check.log`.
