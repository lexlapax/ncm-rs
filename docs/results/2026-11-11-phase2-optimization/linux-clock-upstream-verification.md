# Current upstream and native worktree verification

**Metadata-only checkpoint at `f536960`; not complete preparation or launch readiness.
Linux remains 72/78; Apple pending.** The prior Python preparation deliberately left
current upstream verification false. This checkpoint supplies a new scoped snapshot,
not a retroactive change to that historical record.

## Actual results

- Upstream `.reference/upstream-ncm` is at unchanged pin
  `c70274273f489abc73832b2b98584a8483b6b203`. Its raw commit object recomputes that
  Git SHA-1 with replacement objects disabled. HEAD tree, index and current bytes
  agree for **39 tracked regular files / 877,235 bytes**, including executable modes.
  All index flags are ordinary `H` (no assume-unchanged or skip-worktree entries),
  all stages are zero, and there are **zero untracked and zero ignored entries**.
- HEAD/status checks before and after inspection agree. SHA-256, pathname device/
  inode and final visible metadata checks bind current source files. Git utility,
  reviewed local configuration and index identities are rechecked. Ten bounded
  upstream Git probes completed, exit zero, captured EOF and reaped direct children.
  Their recorded interval is 2026-11-12 08:29:14.084125–08:29:14.104232 UTC; this is
  not an interval for source hashing or a performance measurement.
- Existing `readonly_trace_runtime.verify_worktrees` passed its six probes for
  accepted `3307dff82a08f7251b64de0866a01d131528b74f` and candidate
  `f53696043e71fa5cdaa3f72e28b8c37fef5911d5`. Both tracked trees were clean. Accepted
  has zero untracked entries; candidate has four allowed private session entries.
  **Only counts are public; raw candidate status/names remain private.** Production
  sources/Cargo.lock Git diff from `d44beaa` is empty, as is the manifest Git diff
  from measured `fa040ed`. These native Git guards are not fresh byte inventories
  of the ignored binaries/payloads or of all native worktree files.
- No upstream module/Torch import, GPU query, benchmark, fetch/checkout/reset,
  index-refresh request or global configuration/tooling change occurred. Read-only
  Git probes and file inspection do not establish Python import/bytecode/stdlib/
  native closure, historical interpreter/library bytes, immutable files or an atomic
  snapshot. Ignored-entry absence here says nothing about external Python caches.

[Machine-readable evidence](linux-clock-upstream-verification.json) binds the
[25-member compact archive](raw/linux-clock-upstream-v2.tar.xz), 75,147 raw / 10,524
compressed bytes. It contains the exact v2 inspection source, all upstream captures/
receipts/snapshot, and only the native worktree summary/count-only log. No weights,
upstream source contents, installed library/executable bytes, private session files
or raw candidate status are published. Source SHA-256 identities are current-only;
Git blob/commit SHA-1 is Git's existing object format, not a replacement for those
SHA-256 identities or an upstream pin change.

## Method and retained iterations

The archived `reports/phase2-clock-upstream-verify-v2.py` is the exact inspection
source. It imports only existing metadata/file-identity helpers under `python -S`;
it does not execute the upstream tree. The reviewed `.git/config` has core/remote/
branch settings only and is pinned to SHA-256
`e11456e4bc00390f8f42b98eac749b4e829771b7f59283376b33b8b9ca0d630c` before Git probes.
The explicit environment uses `PATH=os.defpath`, current HOME, C locale,
`PYTHONNOUSERSITE=1`, disabled system/global Git configuration, optional locks off and
terminal prompts off. Commands add `--no-replace-objects --no-optional-locks`,
`core.fsmonitor=false`, `core.untrackedCache=false` and the exact upstream `-C` path.
Each uses the existing owned `metadata_probe`: **10 seconds / 1 MiB**, no retry.

Probe suffixes, in order:

```text
rev-parse HEAD
cat-file commit c70274273f489abc73832b2b98584a8483b6b203
rev-parse --show-toplevel
status --porcelain=v1 -z --untracked-files=all
ls-files -v -z
ls-tree -r -z --full-tree c70274273f489abc73832b2b98584a8483b6b203
ls-files --stage -z
ls-files --others --ignored --exclude-standard -z
rev-parse HEAD
status --porcelain=v1 -z --untracked-files=all
```

Source admission is at most 512 paths, 8 MiB/file and 64 MiB total; actual files are
much smaller. Nonregular files, symlink traversal and unsupported Git modes fail.
Current bytes must independently match SHA-256 inventory and pinned-tree Git blob
SHA-1. Descriptor identities and later metadata rechecks detect visible changes;
they do not make this an adversarial-swap-proof or atomic inventory.

The first metadata inspection (v1) passed the same 39-file check with nine probes.
Review then strengthened the check by disabling Git replacement objects and hashing
the raw commit object; v2's ten probes passed without changing upstream bytes or
pins. V1 remains retained, not relabeled as having those additional checks. These
are metadata inspections, **not retries of either fixed GPU diagnostic**.

Actual commands (stdout/stderr retained privately under the matching log names):

```sh
python -S reports/phase2-clock-upstream-verify-v1.py
python -S reports/phase2-clock-upstream-verify-v2.py
PYTHONPATH=tools/bench python -m unittest test_readonly_trace_runtime test_prepare_clock_python -v
CARGO_BUILD_JOBS=4 ./tools/check.sh
```

Native guards were called in a fresh `python -S` interpreter via
`verify_worktrees(root, expected_candidate_head, root/'reports/phase2-clock-worktrees-verification-v1')`.
The bootstrap expanded reviewed short HEAD `f536960` using `git rev-parse HEAD`
before the helper; that direct bootstrap query is not one of its bounded receipts.
No Git configuration was changed. **15 focused / 391 reporting tests pass**; these
are existing regression tests, not new parity, performance or launch evidence.

| Private retained evidence under `reports/` | SHA-256 |
| --- | --- |
| `phase2-clock-upstream-verify-v1.py` | `7e5587398c37d72c867d52f7c840fb56c9817023b2aaa03974dc672f348c6d3d` |
| `phase2-clock-upstream-verify-v2.py` | `b7cc9cd3824d2925887987b9d23f9d518778e04a1b265ddda95695d60c102fff` |
| `phase2-clock-upstream-verify-v1.log`, `phase2-clock-upstream-verify-v2.log` | `8e929fe968fc897c87d0fe441d85984b2353caa20d0d7c9354604eb73edd67c9` |
| `phase2-clock-upstream-verification-v1/upstream.json` | `8be42d0e036ce420ce64444941960f11b5a5b9954e260a89393884fa69889865` |
| `phase2-clock-upstream-verification-v2/upstream.json` | `250c55d07177330441cd85aad6f747a8f1a7ad8e455f3f2bcbdc8eccce3d38ad` |
| `phase2-clock-worktrees-verification-v1/worktrees.json` | `691334d1d5ebaab56c1d7eec4d6a6b680d334d85cfcb938d477a4489b2c268ab` |
| `phase2-clock-worktrees-verification-v1.log` | `14088ff799306d5439c3ebe80426923ace02e9e2d4ff91b47db95297f846da94` |
| `phase2-clock-upstream-tests-v1.log` | `7c925a6ea1a24e461473abb80546d8706b1001da2372e1f050f82a334740cf04` |
| `phase2-clock-upstream-check-v1.log` | `b90d0f9049210ea5559a89d83a7b9bf18ddf6a43fd08e8835c4cd5f4e1c96d87` |
| `phase2-clock-upstream-reproduce-v1.log` (full snapshot/privacy review) | `784f78566a37a5e02c300ca2747f85221bffa36fc2d25ba877a3bea775b029ee` |
| `phase2-clock-upstream-reproduce-v2.log` (exact archive command below) | `9443fddfa23e6a638c2fc603d6dde589d5cd2a7150cb1d01f9efbe7c25a19d6c` |

Offline archive review checked all 25 member sizes/hashes, ten receipt/raw bindings,
raw commit SHA-1, 39 tree/source declarations and source-byte totals, and equality of
the native summary. It also checked privately that none of the four session filenames
appears in archived bytes. This reproduces snapshot consistency, not fresh source
collection or live provenance. To inspect without executing the inspection script:

```sh
python -S - <<'PY'
import hashlib, json, tarfile
from pathlib import Path
p=Path('docs/results/2026-11-11-phase2-optimization/linux-clock-upstream-verification.json')
spec=json.loads(p.read_bytes())['archive']; archive=Path(spec['path'])
raw=archive.read_bytes()
assert len(raw)==spec['bytes'] and hashlib.sha256(raw).hexdigest()==spec['sha256']
expected={row['path']:row for row in spec['members']}
with tarfile.open(archive) as tar:
    assert len(tar.getmembers())==len(expected)==25
    for item in tar.getmembers():
        assert item.isfile() and item.mode==0o600
        row=expected.pop(item.name); raw=tar.extractfile(item).read()
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
assert not expected
print('25 archived members verified; no collection or hardware action')
PY
```

## Next boundary

This closes the **current upstream inspection gap at this checkpoint**, not the
future prelaunch check. Fresh complete preparation must combine/recheck upstream,
native source/payload/tools, Python/loader/preload identities, aliases, commands and
roles in one reviewed manifest. The restoration-aware production entry and all live
policy/permission/enforcement/restoration/availability gates remain missing. Do not
request a new quiet window until readiness is established; continuation is not a
policy answer. The same-evidence finalizer is implemented, but no actual clock
collection has run and no acceptance comparison closes.

## Reusable preparation component

`tools/bench/prepare_clock_upstream.py` adapts the v2 read-only inspection into
an import-only function with an explicit pinned Git configuration hash and fresh
private output path. The caller must bind the checkout, Git executable, reviewed
configuration hash and records to the future frozen entry. It retains ten bounded
probes and pinned commit/blob checks; unlike v2 it rejects any ignored upstream
entries. It never imports upstream code, touches GPU policy or declares readiness.
Three temporary-Git tests cover pinned bytes including an empty file, tracked and
untracked drift, ignored files, wrong pin/configuration and output containment.
A review caught that passing `None` could otherwise bypass the configuration hash;
explicit digest validation now rejects it before creating output.

Two actual metadata-only calls produced private `reports/phase2-clock-upstream-verification-v3/`
and `-v4/` evidence. The second includes the explicit-hash guard; it inspected
39 tracked files / 877,235 bytes at pinned commit `c70274273f489abc73832b2b98584a8483b6b203`,
with zero ignored entries and `launch_ready=false`. These are current snapshots,
not historical Python/library byte equivalence or a launch preflight. The function
still requires an authenticated caller, later fresh recheck and role binding; it is
not an atomic or adversary-proof filesystem snapshot. No benchmark, Torch import,
GPU query, clock action or acceptance result was produced.

```sh
umask 077; PYTHONPATH=tools/bench python -m unittest test_prepare_clock_upstream test_clock_preparation_inventory -v
```

Seven focused tests pass. The earlier v3 actual call preceded the explicit-hash
guard; the later v4 used the final implementation with the reviewed config SHA-256
recorded above. Raw Git captures remain private in the ignored output directories.
`umask 077; CARGO_BUILD_JOBS=4 ./tools/check.sh` passed 406 reporting tests and
the default checks; that run started before the explicit-hash guard was added,
which was then exercised by the focused tests above. Its local log is
`/tmp/ncm-phase2-upstream-preparation-check.log`.
