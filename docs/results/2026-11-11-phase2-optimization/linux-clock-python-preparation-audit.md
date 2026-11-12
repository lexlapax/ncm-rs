# Clock preparation: Python identity coverage audit — NOT launch readiness

Offline audit at `fef3b08`, before implementing the frozen clock preparation.
**Linux remains 72/78; Apple pending.** No reference benchmark, GPU command,
permission test, clock request, package installation or source change occurred.
The audit ran system Python with `-S`; it did not invoke the qualified venv as an
additional interpreter process, import Torch or execute venv startup hooks.

## Evidence actually checked

The v2 archive SHA-256 remains
`feb5ec2138e070811614af86f05777d0e0c09616ca2134e1730a2533e95a368c`.
All **110 members** match the published member sizes/digests, including **97 JSON
members**. All six independent Python codec reports agree on runtime, Torch build,
upstream commit, package versions, three source digests, artifact/workload digests,
precision, index dtype, warmups and trials. Copied reports are not extra processes.

| Historical evidence | Current offline observation | Meaning / limit |
| --- | --- | --- |
| Venv interpreter version `Python 3.14.9` in `environment.json` | `pyvenv.cfg` says 3.14.9, system-site-packages false; venv Python resolves to `/usr/bin/python3.14` | Version/config metadata, not historical executable or libpython byte equality |
| `torch` runtime field `2.15.0+cu132` | Static AST literals in installed `torch/version.py` declare the same build and CUDA `13.2` | Did not import Torch; neither native-library identity nor enforcement proof |
| `reference_versions`: 36 distribution metadata versions, including `torch == 2.15.0` | All 36 installed distribution versions match exactly | Keep distribution version and runtime build tag separate; they are not interchangeable |
| Three Python helper source digests | Current files match all three | Historical byte comparison for these files only |
| Upstream `c70274273f489abc73832b2b98584a8483b6b203` | Historical payload preflight records that HEAD and clean=true | Not a new current upstream worktree check |
| Static preflight `python_environments.venv` | Historical `packages_match=true`, `python_matches=true` | Does not contain interpreter/library file hashes |

The three historically pinned source files are:

| File | SHA-256 |
| --- | --- |
| `tools/reference/codec_bench.py` | `b1d5fc2f3b6722fafaee9f4be6708e59f96a869934138ae4ca94b20095e162dd` |
| `tools/reference/codec_trace.py` | `f8703f7fed1327afc6dc5a085eff609dc11c6259d0ce14fceec6c9d68525f411` |
| `tools/reference/codec_stages.py` | `523b4d79fb9e774a08f8fe9ef0560ef2fb14f87384150d4249b6b056aba25a29` |

The metadata scan found **37 installed distributions**. The one not checked by
the benchmark's requirements lock is `pip == 26.2.2`; this is not permission to
change/remove it or silently extend the reference lock. One top-level `.pth` file
was present: `distutils-precedence.pth`. Its inspected code imports
`_distutils_hack` and installs its import shim by default. The helper was read, not
executed. No top-level `sitecustomize.py` or `usercustomize.py` was found in that
site-packages directory; this is not a whole-import-path absence proof.

## Current-only byte observations — not historical pins

These files were hashed while no workloads/telemetry were running. They are
**current audit snapshots**, not claims that v2 used identical bytes, not a full
Python/stdlib/native-library inventory, and not a replacement upstream pin.
The venv config is path-dependent and is not expected to be byte-identical after
relocation. Future preparation must independently recheck and bind its inputs.

| File | Bytes | Current SHA-256 |
| --- | --- | --- |
| `/usr/bin/python3.14` | 14,424 | `122ce07db306cd111fd10e27125460362610452983a64903c470afd66b6403bd` |
| `.reference/venv/pyvenv.cfg` | 194 | `bd38c78597055117fb78343fe47bf0719a81d1c4de80cc640aee4bef9a2c7b58` |
| `site-packages/torch/version.py` | 317 | `6feda231420514c053c7116e00e36d3313d6708aa1ab2a93dfa07ad4e5735184` |
| `site-packages/distutils-precedence.pth` | 151 | `2638ce9e2500e572a5e0de7faed6661eb569d1b696fcba07b0dd223da5f5d224` |
| `site-packages/_distutils_hack/__init__.py` | 6,800 | `ced72e54431ecf2d8d7dbd8a1e27724f1b171af3ca1503c304bf11c59a7fe861` |

Here `site-packages` means `.reference/venv/lib/python3.14/site-packages`.
A structured search across all 97 archived JSON members found zero interpreter/
venv `path`+SHA-256 records and zero Python-named hash keys. The report/environment
and preflight fields above were also inspected directly. The structured search is
not semantic proof of absence in arbitrary free text or every repository artifact.
No historical executable/library byte-equivalence claim is supported by this audit.

## Consequences for frozen preparation

1. Reuse the three explicit historical source pins and six-report metadata agreement;
   preserve the original upstream, artifact/workload, FP32 and 5/30 protocol.
2. Bind both the venv command spelling and resolved interpreter image, config,
   exact distribution versions and separate Torch build declaration. Do not infer
   `2.15.0+cu132` merely from distribution metadata `2.15.0`.
3. Capture current interpreter/loader/libpython and required library identities
   with explicit current-only provenance. Existing `inspect_libraries` targets only
   accepted/candidate/injection/taskset; it is **not** Python loader coverage.
   DT_NEEDED still omits lazy Torch/driver/JIT loads; actual process observation
   and bounded post-only additional-library handling remain required.
4. Account for startup hooks rather than claiming that all Python roles are `-S`:
   preparation/internal roles use `-S`, but the frozen Python benchmark argv does
   not. Adding `-S` to that benchmark would change the qualified environment and
   is not proposed. Freeze/review relevant startup inputs without editing them.
5. Preserve intrinsic benchmark work. The unchanged Python harness checks package
   versions, invokes Git for upstream HEAD/cleanliness, hashes checked-load payloads
   and emits source/manifest/workload digests. The native harness also emits source
   and manifest digests. The collector's no-added-probes/frozen-rehash-last rule
   must not be advertised as a run-wide absence of hashing or subprocesses. Do not
   remove or move intrinsic harness checks to make that inaccurate claim true.
6. A complete preparation/entry still needs hardware identity, policy/controller
   confirmation, permissions/enforcement, live role/observer binding, final output/
   telemetry qualification and a later fresh quiet window. This audit closes none
   of those gates and authorizes no intervention.

## Reproduce the checked metadata surfaces

Run from the repository root with system Python `-S`. The archive portion needs
only published files. Current-state comparisons require the preserved qualified
`.reference/venv`; a clean checkout without it cannot reproduce those observations.
The script performs file reads/AST parsing only: no installed package import,
startup hook, metadata utility subprocess, GPU probe or benchmark.

```python
import ast
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import tarfile

root = Path.cwd()
base = root / 'docs/results/2026-11-11-phase2-optimization'
spec = json.loads((base / 'linux-diagnostic-v2-evidence.json').read_bytes())['archive']
def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()
assert digest(root / spec['path']) == spec['sha256']
pins = {item['path']: item for item in spec['members']}
records = {}
with tarfile.open(root / spec['path'], 'r:xz') as archive:
    members = archive.getmembers()
    assert len(members) == len(pins) == len({m.name for m in members}) == 110
    for member in members:
        assert member.isfile() and member.size == pins[member.name]['bytes'] <= 16 * 1024**2
        raw = archive.extractfile(member).read()
        assert hashlib.sha256(raw).hexdigest() == pins[member.name]['sha256']
        if member.name.endswith('.json'):
            records[member.name] = json.loads(raw)
assert len(records) == 97
counts = {'structured_venv_or_interpreter_identities': 0, 'python_named_hash_keys': 0}
def inspect(value):
    if isinstance(value, dict):
        path = value.get('path')
        if isinstance(path, str) and ('/.reference/venv/' in path or path.startswith('.reference/venv/')
                                     or '/python3' in path or 'libpython' in path):
            if any('sha256' in key for key in value):
                counts['structured_venv_or_interpreter_identities'] += 1
        for key, item in value.items():
            if 'python' in key.lower() and ('sha256' in key.lower() or 'hash' in key.lower()):
                counts['python_named_hash_keys'] += 1
            inspect(item)
    elif isinstance(value, list):
        for item in value:
            inspect(item)
for record in records.values():
    inspect(record)
assert not any(counts.values())
prefix = 'phase2-linux-diagnostic-v2/'
reports = [records[prefix + f'python/codec-python-{i}.json'] for i in range(1, 7)]
first = reports[0]
for field in ('runtime', 'torch', 'upstream_commit', 'reference_versions', 'source_sha256',
              'manifest_sha256', 'workload_sha256', 'precision', 'index_dtype', 'warmups', 'trials'):
    assert all(report[field] == first[field] for report in reports)
sources = {'benchmark': 'codec_bench.py', 'inputs_helper': 'codec_trace.py', 'stages': 'codec_stages.py'}
for key, name in sources.items():
    assert digest(root / 'tools/reference' / name) == first['source_sha256'][key]
site = root / '.reference/venv/lib/python3.14/site-packages'
normalize = lambda name: re.sub(r'[-_.]+', '-', name.lower())
installed = {}
for dist in metadata.distributions(path=[str(site)]):
    name = normalize(dist.metadata['Name'])
    assert name not in installed
    installed[name] = dist.version
for name, version in first['reference_versions'].items():
    assert installed[normalize(name)] == version
extras = {name: version for name, version in installed.items()
          if name not in {normalize(n) for n in first['reference_versions']}}
values = {}
version_file = site / 'torch/version.py'
for node in ast.parse(version_file.read_text()).body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        if node.targets[0].id in ('__version__', 'git_version', 'debug'):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        if node.target.id in ('cuda', 'hip', 'rocm', 'xpu'):
            values[node.target.id] = ast.literal_eval(node.value)
assert values['__version__'] == first['torch']
current_files = [(root / '.reference/venv/bin/python').resolve(), root / '.reference/venv/pyvenv.cfg',
                 version_file, *sorted(site.glob('*.pth')), site / '_distutils_hack/__init__.py']
print(json.dumps({'archive_members_verified': 110, 'json_members': 97, 'python_reports': 6,
                  'structured_identity_search': counts,
                  'historical_python': records[prefix + 'environment.json']['venv']['python'],
                  'matching_distribution_versions': len(first['reference_versions']),
                  'extra_distributions': extras, 'current_torch_source_literals': values,
                  'current_only_files': [{'path': str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                                          'bytes': path.stat().st_size, 'sha256': digest(path)} for path in current_files],
                  'historical_interpreter_bytes_verified': False,
                  'historical_library_bytes_verified': False, 'launch_ready': False}, indent=2))
```

Exact archive/current metadata inspection commands also read
`tools/reference/codec_bench.py`, the native `codec_bench.rs`,
`readonly_trace_runtime.py`, installed `torch/version.py`, the `.pth` file and
`_distutils_hack/__init__.py`. The retained logs below are private, not new acceptance
evidence. Audit-v2's phrase “no current interpreter execution” means no additional
qualified-venv invocation; the audit itself necessarily ran system Python `-S`.

| Private log under `reports/` | Check | SHA-256 |
| --- | --- | --- |
| `phase2-clock-python-inventory-audit-v1.log` | Seven selected members/report fields | `25f64e73031704664254abc646306c54e46d71d6550f2b9b5577961fc5baf486` |
| `phase2-clock-python-inventory-audit-v2.log` | Six-report agreement/current 36 distribution versions/three sources | `8890b53d689975be5f6efcef6321d79ba557af2f98bc92bc3e4bbe3a4110d17c` |
| `phase2-clock-python-inventory-audit-v3.log` | Extra pip, static Torch literals and five current-only file hashes | `a02cf24ca0018452920d1a2f13f01abd9291816c97ec99ebb946076a3fe1b93b` |
| `phase2-clock-python-inventory-audit-v4.log` | All 110 member hashes and structured identity search | `0481b4373953df5f7a38671d8efb307af748dee9535e02593a7ba3b5ad81db6e` |
| `phase2-clock-python-inventory-audit-v5.log` | Historical Python preflight/upstream fields | `c65e6cfabeee8f15b416bcd15480b415f4da7447901153a92731a2cfccd24371` |
| `phase2-clock-python-audit-repro-v1.log` | Initial documented metadata reproduction | `4f1956e1ac882690bfc355600de03ae155f999a95791b970aa58f5c96db0e53e` |
| `phase2-clock-python-audit-repro-v2.log` | Final documented reproduction, including structured search | `ed399af6d944992595f3f8063abfcbf56d66c2ed21512a7c7369daafa8c539a2` |

The documented Python block was executed in a fresh `python -S` process; both
reproduction versions passed. Logs were created exclusively with `umask 077`.
No executable production/tool source changed in this audit. Default checks were
not rerun for this documentation-only checkpoint; the prior `fef3b08` checkpoint's
311 reporting tests remain historical evidence, not new hardware validation.
