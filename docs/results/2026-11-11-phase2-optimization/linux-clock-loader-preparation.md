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

## File-only preload candidate follow-up

`prepare_clock_preloads.py::inventory` now inventories the reviewed Torch source's
preload candidates within **one explicitly supplied site-packages directory**.
It imports no Torch code, calls no candidate `dlopen`, starts no utility and makes
no GPU query. Three explicit current-only source digests are required; bounded AST
literal parsing extracts the CUDA version, build tag, loader flags, ordered CUDA
list and final optional NVTX call. The caller must review the whole source/layout
logic; AST extraction is not a general semantic verifier or independent approval.

The three reviewed layouts are `nvidia/<folder>/lib`, `nvidia/cu<major>/lib`, and
`<folder>/lib`, under that one root. Up to 4,096 directory entries across distinct
search directories are scanned, with 128 distinct candidate files, 2 GiB/file and
8 GiB total bounds before candidate hashing. Source reads retain the existing
1 MiB bound. Search-directory symlinks and candidate escapes fail; in-root library
aliases are retained and canonical identities deduplicated. Directory names,
matched path resolutions, sources and file identities are rechecked. Missing or
multiple matches are retained, never installed or silently selected. This is not
an atomic filesystem snapshot; trusted ancestors and later full rechecks remain
necessary. Directory entries outside the matched patterns are not frozen content.

**Source list order is retained; sorted match order is not Python's glob order.**
Every `runtime_selected_path` is null. Site candidate inventory is verified, but
complete `sys.path`, dependency closure, actual load selection, historical byte
equivalence and launch readiness remain false. The helper does not claim that an
optional best-effort preload succeeds just because its candidate file exists.

Actual fresh system `python -S` call, under `umask 077` with exclusive output/log
creation (the qualified benchmark is unchanged):

```python
from pathlib import Path
import json, sys
root = Path.cwd()
sys.path.insert(0, str(root / 'tools/bench'))
from prepare_clock_preloads import inventory
assert 'torch' not in sys.modules
result = inventory(root / '.reference/venv/lib/python3.14/site-packages', reviewed_sources={
    '__init__.py': '7ed81103f3d256de131b886e67be458b3205784e3b4af9c495041282391e19db',
    '_utils_internal.py': 'eb77101e63128e55da13b29c7133e42b702a00e0b6eb1d44521ba1944178e794',
    'version.py': '6feda231420514c053c7116e00e36d3313d6708aa1ab2a93dfa07ad4e5735184',
})
assert 'torch' not in sys.modules and result['launch_ready'] is False
with (root / 'reports/phase2-clock-preload-inventory-v1.json').open('x') as output:
    json.dump(result, output, indent=2, sort_keys=True)
    output.write('\n')
```

Results: **16 source entries, 53 scanned directory entries, 15 distinct ELF
candidates / 1,956,767,857 bytes**. No pattern had multiple matches in this root.
The optional `libnvToolsExt.so.*[0-9]` pattern has no match here; no package or policy
was changed. This is not absence proof across all `sys.path` roots or the system.

Twelve overlapping files agree with the earlier standalone inventory on
path/device/inode/size/SHA-256. Three additional **current-only** candidates are:

| Candidate | Bytes | SHA-256 |
| --- | ---: | --- |
| `libcusolver.so.12` | 156,946,784 | `bdf40a5229a90438ae8ee9fd32bb9fcacc70024e7e74b61ab98a9f5e315fea31` |
| `libnvJitLink.so.13` | 117,247,464 | `0ba4a7e2c0819291ce928513755f4931dd69c77c63fb77677d14dc1360f28601` |
| `libnvrtc-builtins.so.13.2` | 4,442,056 | `6b1c571cc730d5fcfd57f322e1fa7e0e65de7454b2239ff6d552a09b82d47dbe` |

The union is 38 files / 3,825,752,946 bytes **across two preparations**, not a fresh
simultaneous snapshot of all 38 files. No numerical/performance attribution follows.

### Required-mapping integration boundary

Candidate files and mandatory executable mappings are different sets. The next
binding must keep these distinctions rather than requiring every candidate to load:

- Native previous/candidate: retain the matching eleven standalone dependencies.
- Python: bind its interpreter dependencies, `_C` image and stable core Torch code
  libraries (`libtorch_python`, `libtorch_cpu`, `libtorch_cuda`, `libc10`,
  `libc10_cuda`) as a proposed mandatory subset, to be tested in the frozen entry.
  Do not claim this proposal is already implemented or observed on hardware.
- Freeze reviewed candidate identities even when optional or not required to map.
  For observed known candidates, validate the matching prepared identity; do not
  silently treat a changed known file as a new post-only dependency.
- Preserve additional/lazy mappings as post-only with existing bounds and explicit
  limitations. An inventory is not an allowlist claiming all other mappings absent.
- No removal of startup hooks, forced shared native/Python CUDA tree, changed
  Python argv, or widened numerical/output requirements is justified by this work.

Six new tests cover the three layouts, absent/ambiguous/aliased candidates,
no execution, malformed or changed source declarations, path escapes, invalid ELF,
source flags, concurrent directory/inode replacement and bounds before candidate
hashing. They use synthetic source/ELF fixtures, not CUDA. Exact checks:

```sh
PYTHONPATH=tools/bench python -m unittest test_prepare_clock_preloads test_prepare_clock_libraries test_prepare_clock_python -v
CARGO_BUILD_JOBS=4 ./tools/check.sh
```

**20 focused tests and default checks with 331 reporting tests pass.** The following
private evidence remains under `reports/`; the JSON is not published as a portable
or authenticated environment image:

| Evidence | SHA-256 |
| --- | --- |
| `phase2-clock-preload-tests-v1.log` | `8391f828fbbcf95b95786e1cb0e13ed1528fb80bdeed111f6394d445dcd803b7` |
| `phase2-clock-preload-check-v1.log` | `21d31ce2a2b45b7c2aef1d5a8aecbd690568bed389d634171af53fa0f5ec5fdb` |
| `phase2-clock-preload-inventory-v1.log` | `26433721a7cd4d575be3a00929ff854da99aff72a85f76bad4bdf12baced0c91` |
| `phase2-clock-preload-inventory-v1.json` | `63dce3fd03d4af607c50c631bdf7e66703ea95fade7c1770d34fd3672578faa2` |
| `phase2-clock-preload-overlap-v1.log` | `45be09985597e2e92101898b505a2cd39c22d803dc4112f3f8beb79912f732fb` |

Next: implement/test the required-map and known-candidate binding, current upstream
verification, frozen full preparation/entry and final qualification. Complete
import closure is not claimed by these component results. The live policy,
permission/enforcement and fresh-window gates remain unresolved. Linux **72/78**,
Apple pending; no acceptance cell closes and the active objective is incomplete.

## Sample-time known-candidate guard

The next component adds an opt-in `known_libraries` declaration to
`ProcessObservation`. Its default remains empty for existing read-only callers;
no observation schema, historical record, sample limit or timing budget changes.
The clock `process_factory` now **requires** an explicit nonempty known-library
inventory (at most 256 supplied identities), validates full declaration fields,
rejects conflicting duplicate paths, and requires every mandatory library to be
present with matching device/inode/size/SHA-256. Observation-only `path_device` /
`path_inode` fields are rejected in clock file declarations, so sampled-path fields
cannot override a prepared identity when constructing the required set. The handoff integration fixture
now supplies this explicit argument too. This is internal diagnostic tooling,
not a change to production model or public backend APIs.

The factory deep-copies the declarations and builds independent per-job guard
state before any job. Construction/issuance still performs no file reads, stats,
path resolution, hashing or subprocess calls. During each otherwise complete
sample, a mapped path that matches a known declaration must match its prepared
**pathname device/inode**, including the very first observed sample. A mismatch
raises a fixed local error and latches observer failure; the sequence stops after
that child. It cannot be reclassified as an unknown post-only library simply
because it was already changed at first observation. Kernel-map device values
remain separate from pathname device values, as before.

Known does **not** mean mandatory: a candidate absent from all samples is allowed
unless it also belongs to the required set. Unlisted paths remain subject to the
separate post-only rules. No candidate file hashing is added during workloads or
telemetry. Same-inode byte mutation, arbitrary aliases, unsampled/transient maps
and immutable mapped bytes are not proved by this pathname/inode guard; reviewed
path bindings and final frozen-file checks remain necessary. The guard does not
by itself establish that the caller supplied the full inventory or authentic
preparation evidence.

Tests extend the existing actual CPU-child factory suite:

- An unused `true` executable acts as an absent optional candidate; all eighteen
  fake codec processes still complete with independent real `/proc` observations.
- A mismatched optional libm declaration fails the first child, retains failed
  observation evidence/reaping and prevents a second job. Clearing the guard
  after exit cannot erase the already-latched failure.
- Empty, incomplete, conflicting and oversized inventories fail before work.
  Observed-path fields in known or required declarations are rejected too.
- Mutating caller-owned known identities after factory creation cannot mutate the
  eighteen independent guards; the existing no-IO/no-hashing construction test
  now covers this additional state.

```sh
PYTHONPATH=tools/bench python -m unittest test_clock_process_observation test_observe_readonly_process test_clock_worker_handoff -v
CARGO_BUILD_JOBS=4 ./tools/check.sh
```

**22 focused tests and default checks with 333 reporting tests pass.** Private logs:

| Evidence under `reports/` | SHA-256 |
| --- | --- |
| `phase2-clock-known-library-tests-v1.log` | `563117422ee860c3ad55a3763d057d7903f58f382a2f8530d3cfbd6a200457bc` |
| `phase2-clock-known-library-check-v1.log` | `ceb167bd6fdd0cdd97a1cd9ae1396bf84f1b68979032be29191f9308718ed939` |
| `phase2-clock-known-library-tests-v2.log` (final, including observed-field rejection) | `22013ef0e761af821c2d468eebbd1f8413a66e7d3abe663854de1fa84dfa0104` |
| `phase2-clock-known-library-check-v2.log` (final) | `f863346f5379a9cd0623c70e6a305a25bab917aa774175e3a8ea3e5471c48abf` |

Still pending: compile/bind the actual prepared inventories and proposed mandatory
subset into frozen roles; independently cross-check known-candidate identities in
clock postflight; current upstream verification; full entry/lifecycle/telemetry/
output qualification. No new real library preparation, Torch import, GPU query,
clock operation or benchmark ran in this guard checkpoint. Policy confirmation,
permissions/enforcement and later fresh availability are unchanged prerequisites.
Linux **72/78**, Apple pending; no acceptance comparison or goal completion.
