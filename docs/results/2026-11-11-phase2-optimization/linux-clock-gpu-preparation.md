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
stubbed command results; no real GPU query occurs in those tests. Earlier separate
read-only identity and supported-clock queries are documented in the
[role plan](linux-clock-role-plan.md#read-only-gpu-identity-and-supported-clocks).
`umask 077; CARGO_BUILD_JOBS=4 ./tools/check.sh` passed all default checks,
including 429 reporting tests; local log: `/tmp/ncm-phase2-gpu-preparation-check.log`.

## Actual read-only component preparation

The committed component ran outside the sandbox and matched the historical GPU
UUID, PCI address and name. It parsed 114 supported-graphics rows including
1710 MHz and retained `gpu_lock_permission_verified=false` and
`launch_ready=false`. This does not verify current lock policy, lock support,
permission, enforcement, restoration, quietness or future device identity.

```sh
umask 077; PYTHONPATH=tools/bench python3 -S -c 'import json; from pathlib import Path; from prepare_clock_gpu import inspect; from readonly_trace_preflight import isolated_environment; root=Path.cwd(); out=root/"reports/phase2-clock-gpu-preparation-v1"; r=inspect(root,out,nvidia_smi=Path("/usr/bin/nvidia-smi").resolve(strict=True),environment=isolated_environment(Path.home(),{}),expected_uuid="GPU-180283f7-eafd-6501-7e06-eef40d304c41",expected_pci="00000000:01:00.0",expected_name="NVIDIA GeForce RTX 3060 Ti"); (out/"result.json").write_text(json.dumps(r,indent=2,sort_keys=True)+"\n"); print(json.dumps({k:r[k] for k in ("supported_graphics_row_count","requested_graphics_mhz","requested_graphics_in_application_list","gpu_lock_permission_verified","launch_ready")}))' > /tmp/ncm-phase2-clock-gpu-preparation-v1.log 2>&1
```

Private ignored result `reports/phase2-clock-gpu-preparation-v1/result.json` has
SHA-256 `bc96c5020034f59066fefa3030f0b7636b0ab608532fe073e0eb1d351b38a011`;
its `identity.txt` has SHA-256
`a36ff37986bb24701d3235c7af4a2e69e9121afc2c575061023bdeed392da544`,
and `supported-graphics.txt` has SHA-256
`b90cfe2abbb98c11366c42bf2dfaebfdb140444476a1f00a10805cc99658441f`.
The local command log has SHA-256
`9eaa34a577ff3b8edad6056b13aedb397faad1d80da080ffbed8e7bd1d7dc2db`.

## Comparing supplied GPU records

`compare_clock_gpu.compare` checks the read-only record scope, exact two query
commands and bounded-success receipts, device identity, NVIDIA utility identity,
row count and query-output digests. It compares stable fields while allowing
receipt process IDs/times to differ. It reads no files and runs no commands.
Producer authentication, capture-file re-reading, environment binding, hardware
policy, permission, enforcement and restoration remain entry responsibilities.

```sh
umask 077; PYTHONPATH=tools/bench python -m unittest test_compare_clock_gpu test_prepare_clock_gpu test_clock_role_plan -v
PYTHONPATH=tools/bench python3 -S -c 'import json; from pathlib import Path; from compare_clock_gpu import compare; r=json.loads(Path("reports/phase2-clock-gpu-preparation-v1/result.json").read_bytes()); print(json.dumps(compare(r,r,root=Path.cwd())))'
```

Seven focused tests pass, including ten altered-record/receipt cases. The retained
actual v1 record compares to itself with producer authentication, permission and
readiness false; this is not a fresh device check or an acceptance result.

`bind_snapshot` additionally verifies that the NVIDIA utility identity in a
supplied GPU record exactly matches the utility frozen in a supplied validated file
snapshot. The actual retained file/GPU v1 records bind successfully, with producer
authentication and readiness still false. Eight focused tests pass after this
addition, including utility-inode drift and omission from the frozen recheck
inventory; neither record is refreshed by binding.
`umask 077; CARGO_BUILD_JOBS=4 ./tools/check.sh` passed all default checks,
including 431 reporting tests, before the binding addition; the updated binding
then passed the focused tests. Local log: `/tmp/ncm-phase2-compare-clock-gpu-check.log`.
