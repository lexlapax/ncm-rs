# Clock preparation: standalone loader inventory

**Status: preparation component only; Linux 72/78, Apple pending.** This extends
[Python metadata preparation](linux-clock-python-preparation-audit.md), not the
qualified benchmark or the failed read-only collection. No model execution, Torch
import, GPU query, permission test or clock action occurred. The system Python
`-S` collector ran; no target interpreter/benchmark main ran.

## Implementation and boundaries

`tools/bench/prepare_clock_libraries.py::inspect` accepts six explicitly declared,
already-reviewed target images: previous/candidate native codec, Python interpreter,
Torch `_C` extension, Torch global-dependency library and taskset. Before the first
utility it verifies every target/loader hash, device/inode/path, size and ELF marker,
SONAME declarations, canonical allowed roots and isolated metadata environment.
The ELF marker is not a full ELF verifier or a sandbox for untrusted executables.

Each existing `metadata_probe` call runs the trusted glibc loader with `--list`,
a 10-second timeout and 1 MiB merged capture. It does not run benchmark main.
The output directory is exclusively created with mode 0700; validation/probe failures
retain prior captures/receipts without writing `libraries.json`. The final report is
written mode 0600. An output-write failure can leave partial JSON; file presence
alone is not evidence of success.
No caller environment/control variables are inherited beyond the reviewed five-key
metadata environment. The clock worker's separate six-key environment is unchanged.

Standalone resolution must contain caller-declared SONAMEs; previous/candidate
native listings must agree. Canonical libraries must remain within explicit roots.
The preparation has limits of 128 distinct libraries, 2 GiB per file and 8 GiB
aggregate library bytes, checked before library hashing; six targets and the loader
are separately bounded by the per-file limit. These are preparation limits, not
changes to diagnostic quotas or bounds on wall time spent hashing. All identities
and listed-to-canonical paths are rechecked after listing. Trusted ancestors and
separate full launch preflight/postflight remain necessary; this is not an atomic
filesystem snapshot or a proof of immutable mapped bytes.

Only `standalone_loader_resolution_verified` is true. Runtime library observation,
Python load-order verification, historical library byte equivalence and launch
readiness remain false. Target declarations/hashes establish consistency, not
independent authentication. This helper does not bind a production launch entry.

## Actual preparation

Command, from the repository root under `umask 077`, with exclusive log creation:

```sh
python -S reports/phase2-clock-loader-prepare-v1.py > reports/phase2-clock-loader-prepare-v1.log 2>&1
```

The private script rechecked the preceding Python preparation's historical sources
and current-only files. Native codec/taskset identities were checked against the
published fixed hashes, not newly substituted programs. Inputs were retained in
`reports/phase2-clock-loader-inputs-v1.json`. The exact invocation contract is:

- Six targets as above; Python image from `phase2-clock-python-preparation-v3.json`.
  Exactly one installed `torch/_C*.so`; explicit `torch/lib/libtorch_global_deps.so`.
- `required` always includes `libc.so.6`; native adds `libcuda.so.1`, `libcurand.so.10`,
  `libcublas.so.13`; Python adds `libpython3.14.so.1.0`; Torch `_C` adds
  `libtorch_python.so`, `libtorch_cpu.so`, `libtorch_cuda.so`; global-deps adds
  `libcudart.so.13`. These are standalone-list requirements, not runtime assertions.
- Loader `/usr/lib/ld-linux-x86-64.so.2`, resolved and snapshotted before use.
- Roots `/usr/lib`, `/opt/cuda/targets/x86_64-linux/lib`, and the preserved venv's
  `lib/python3.14/site-packages`; all required canonical directories.
- Environment from `isolated_environment(HOME, os.environ)`; inherited native-control
  variables checked, only PATH/HOME/LANG/LC_ALL/PYTHONNOUSERSITE propagated.
- Output `reports/phase2-clock-loader-preparation-v1`, created exclusively. No
  installation, loader-path edit, preload override or startup-hook suppression.

All six bounded utility receipts are complete, exit zero, EOF observed and direct
children reaped. A separate evidence check verifies capture hashes/lengths, stored
receipt equality, exact declared argv and library byte accounting. These receipts
are utility observations, not codec-process identity or lifetime observations.

| Target | Standalone dependency entries |
| --- | ---: |
| Previous native | 11 |
| Candidate native | 11 |
| Python interpreter | 4 |
| Taskset | 2 |
| Torch `_C` | 30 |
| Torch global-deps | 10 |

Union: **35 current-only canonical libraries, 3,547,116,642 bytes**. Native accepted
and candidate dependencies agree. Native versus Torch `_C` differ in canonical
resolution for `libcublas.so.13` and `libcublasLt.so.13`. This is not a new package
installation, historical-byte comparison, or explanation for the B1 timing clusters.
No library contents, executable files or licensed weights are published.

## Why standalone listings cannot become the runtime contract unchanged

Read-only review of the installed `torch/__init__.py` loader section shows
`_load_global_deps` using `ctypes.CDLL(..., RTLD_GLOBAL)`, then potentially calling
`_preload_cuda_deps(required=False)` before importing `torch._C`. The preload list
covers additional CUDA components, including nvJitLink and cuSOLVER not in the
standalone `_C` listing. Selection scans `sys.path` and can change which SONAME
wins. `_utils_internal.py` declares `USE_GLOBAL_DEPS=True` and
`USE_RTLD_GLOBAL_WITH_LIBTORCH=False`. These observations are source review, not
execution/import-closure proof; no functions in those installed modules ran.

Two source identities captured before and rechecked after this preparation are
**current-only**, not historical source pins:

| Installed Torch file | Bytes | SHA-256 |
| --- | ---: | --- |
| `__init__.py` | 133,486 | `7ed81103f3d256de131b886e67be458b3205784e3b4af9c495041282391e19db` |
| `_utils_internal.py` | 12,938 | `eb77101e63128e55da13b29c7133e42b702a00e0b6eb1d44521ba1944178e794` |

**Do not promote every standalone dependency into mandatory Python mappings.**
Finish explicit preload-candidate inventory/review and a defensible required-map
contract, then bind actual owned-child observations and postflight. Lazy additional
libraries remain post-only unless explicitly inventoried beforehand. Do not disable
preloads, add `-S` to the qualified Python benchmark, replace CUPTI, or force native
and Python to use one library tree merely to make the listings match.

## Tests and retained evidence

Seven new tests cover static/current-only scope, all-target prevalidation before
utilities, missing/unresolved/out-of-root dependencies, native disagreement, failure
retention, same-byte inode replacement, symlink drift and pre-hash bounds. One uses
the actual system glibc loader on the system Python ELF for all six synthetic role
labels; it proves utility transport only, not six real GPU/runtime identities.
The first run's drift fixture reused `image` as an output directory and failed with
`FileExistsError`; only that fixture path was corrected. No production guard was
weakened. The initial log is retained. Final test-only portability review also
allows the system interpreter's canonical base-prefix library directory, rather
than assuming every Python installation lives under `/usr`; actual preparation
roots and production guards are unchanged.

```sh
PYTHONPATH=tools/bench python -m unittest test_prepare_clock_libraries -v
PYTHONPATH=tools/bench python -m unittest test_prepare_clock_libraries test_prepare_clock_python test_readonly_trace_runtime -v
CARGO_BUILD_JOBS=4 ./tools/check.sh
```

Final **22 focused tests and default checks with 325 reporting tests pass**. No
new CUDA/MLX correctness/performance gate is claimed. Test fixtures do not prove
Torch preload selection, permissions, enforcement or restoration on real hardware.

| Private evidence under `reports/` | SHA-256 |
| --- | --- |
| `phase2-clock-loader-tests-v1.log` (fixture failure) | `e33d4e01d49ca6bd30167190c78e360284bfa1676d901b85c55ebec9469fcf11` |
| `phase2-clock-loader-tests-v2.log` (22 pass) | `b2afdedfe5e09be920b81c33788913c50e8081f4e7c8282e77182e8fc0ee579c` |
| `phase2-clock-loader-check-v1.log` (325 pass) | `cabea995b344e94f5fa45eb2b930dafdd9788c3022d3ab036dd1e384c15db44a` |
| `phase2-clock-loader-tests-v3.log` (final 22 pass) | `f4067d0d00d99664f4f4cf714706013bb94a52716ac12c08b3a057287cd083dc` |
| `phase2-clock-loader-check-v2.log` (final 325 pass) | `67a96acc63a4a58261497d1d145e0f12c05621b8cb191aae54f24f68a29e1208` |
| `phase2-clock-loader-prepare-v1.py` | `c7e14230a7de258c81944c3d59045353124e15f1df698e985b9665e22b7cb225` |
| `phase2-clock-loader-prepare-v1.log` | `50a413c0ce584b6e08ea06ea872767207261f52326648899ccf7a94f4222298e` |
| `phase2-clock-loader-inputs-v1.json` | `4fdb3944be1655b097447efd8fce745dc0719cae79d4ff07b1a2a159e2b18a89` |
| `phase2-clock-loader-preparation-v1/libraries.json` | `3d1a6cd6eaad0277eb33a9cf34fec1c4d44fc0f70b1a225fa66a4e18ec137f57` |
| `phase2-clock-loader-evidence-check-v1.log` | `71559ae3d2fdbd8ff8b714c279de928eb677608e9077a755f6adb57fec76ff69` |

The import-only helper/tests and invocation contract are public; raw local inventory
and installed libraries are not a clean-checkout runtime reproduction. Repeating
preparation needs the preserved installed environment and fresh output paths, and
would yield new utility receipts, not historical identities. This is separate from
fixed, non-repeatable diagnostic attempts.

Remaining: preload/import inventory, current upstream worktree verification, frozen
full preparation/entry and role binding, final lifecycle/process/telemetry/output
qualification with frozen rehash last. Default-policy/no-controller confirmation,
permission/enforcement evidence and a later fresh quiet window are still missing.
The new helper and 35-library inventory close no acceptance comparison.
