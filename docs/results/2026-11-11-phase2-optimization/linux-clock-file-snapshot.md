# Preparation-only clock file snapshot

`prepare_clock_snapshot.prepare` composes the existing native pinned-input,
current upstream, Python/startup, preload, static loader, worktree, tool, utility
and timezone metadata components. It uses the reviewed source/config hashes
already recorded in the Python and upstream preparation audits. The existing
pure library compiler and declaration admission are followed by a current-file
and alias recheck before an exclusive private snapshot is published. This is an
import-only preparation component, with no collection or diagnostic launch.

It independently checks all 18 planned codec commands, rotating order, CPU2,
working directories, reports and deferred Python copies against the fixed
protocol and prepared executable declarations. The producer uses the specified
isolated environment; it neither starts telemetry nor runs Torch, GPU or policy
commands. Snapshot output is bounded to the existing 16 MiB metadata limit.

The resulting snapshot leaves launch readiness, GPU identity, hardware policy,
permissions, restoration and quiet-window availability **false**. It does not
bind the future live coordinator/observer/restorer/worker roles, prove immutable
mapped bytes or establish complete Python import closure. A restoration-aware
entry must freshly compare prepared records, authenticate role ancestry and
commands, verify policy/permissions and request a fresh quiet window; complete
collection and postflight remain separate gates. The snapshot is not a Phase 2
acceptance result.

```sh
umask 077; PYTHONPATH=tools/bench python -m unittest test_prepare_clock_snapshot test_clock_preparation_inventory test_verify_clock_preparation_files -v
```

Sixteen focused tests pass, including changed plan order/cwd/copies/argv/output,
failed file recheck without publication, and no collection directory on failure.
Producer calls are stubbed in these tests. The subsequent actual assembled
snapshot is described below; no GPU run occurred.

`umask 077; CARGO_BUILD_JOBS=4 ./tools/check.sh` passed all default checks,
including 418 reporting tests. Local log: `/tmp/ncm-phase2-clock-snapshot-check.log`.

## First actual file-only assembly

At clean pushed candidate `ff8d3707ca9c98de40e6b8f51b673c596502ee96`, one
`python3 -S` call assembled and rechecked **317 declared files / 4,020,208,680
bytes**, 56 aliases and 18 planned jobs. It included 144 tool files, all 39
upstream source files plus three Git identities, and the reviewed Python and
loader inventories. The result explicitly leaves GPU identity, hardware policy,
permissions, restoration, availability and readiness false; it did not create
the collection path or execute Torch, telemetry, benchmark or GPU/policy commands.

```sh
umask 077; PYTHONPATH=tools/bench python3 -S -c 'import json,sys; from pathlib import Path; from prepare_clock_snapshot import prepare; root=Path.cwd(); result=prepare(root,root/"reports/phase2-linux-clock-diagnostic-v3",root/"reports/phase2-clock-snapshot-v1",candidate_head="ff8d3707ca9c98de40e6b8f51b673c596502ee96",home=Path.home().resolve(strict=True),python=Path(sys.executable).resolve(strict=True),git=Path("/usr/bin/git").resolve(strict=True),taskset=Path("/usr/bin/taskset").resolve(strict=True),nvidia_smi=Path("/usr/bin/nvidia-smi").resolve(strict=True)); print(json.dumps(result))' > /tmp/ncm-phase2-clock-snapshot-v1.log 2>&1
```

The private ignored snapshot is `reports/phase2-clock-snapshot-v1/snapshot.json`,
SHA-256 `a64bc2e96abac38ec118df778f3550c175691aa37e5f33eb7f2d254ea2e8340c`;
the private log `/tmp/ncm-phase2-clock-snapshot-v1.log` has SHA-256
`402cd9bdd3894fda244c7577b9acc4d0be3e38f3956b38e2d51c04a395369367`.
The snapshot describes that exact source state, not this later documentation
commit. Refresh and compare it against the reviewed final entry/HEAD before any
collection. It is neither proof of an atomic filesystem snapshot nor full launch
preparation, producer authentication, live role binding or acceptance.
