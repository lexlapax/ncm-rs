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
Producer calls are stubbed in these tests. No actual assembled snapshot or GPU
run has yet occurred; actual preparation remains a separate metadata-only check
once the source and candidate commit are clean and reviewed.

`umask 077; CARGO_BUILD_JOBS=4 ./tools/check.sh` passed all default checks,
including 418 reporting tests. Local log: `/tmp/ncm-phase2-clock-snapshot-check.log`.
