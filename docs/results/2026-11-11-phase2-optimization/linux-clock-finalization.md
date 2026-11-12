# Clock same-evidence finalization

**Status: import-only composition implemented; frozen production entry still missing.
Linux remains 72/78; Apple pending.** No GPU query, Torch import, benchmark or real
policy command ran for this checkpoint. This does not complete Phase 2 or authorize
the clock fallback.

## Boundary and ordering

`tools/bench/finalize_clock_diagnostic.py::run_then_finalize` accepts bounded JSON
input bytes, their separately supplied SHA, a fresh private launch path and one
**trusted restoration-aware runner**. `clock-finalization-inputs` is diagnostic
metadata, not a complete preparation manifest or a model/artifact schema change.
It binds the supplied collection path, eighteen jobs, six receipt-backed role
commands/budgets, library contract, frozen file/alias inventory, injection identity,
source/reference declarations, CPU source, GPU UUID/timezone and restoration cwd/
cancellation-hook expectation. Input consistency is not authentication or proof that
the inventory is complete. The future entry must validate the complete preparation
before any launch; this helper's structural checks are not that preflight.

Collection and launch paths must be canonical, fresh and disjoint. The helper creates
an exclusive 0700 launch directory and 0600 input copy before invoking the callback
**once**. The callback must stop/reap both sibling branches safely and return its
coordinator result plus the **original armed work deadline**, not a restarted budget.
There is no generic supervisor, signal, cleanup retry, GPU operation or launch CLI
in this helper. An exception preserves partial files and leaves the attempt occupied.
A return dictionary does not independently prove reaping or safe ancestry.

After the trusted runner returns:

1. Verify the retained input copy, then use the existing bounded private reader for
   the **same 95 sequence/control files**. Bind returned versus persisted coordinator
   records and original deadline; no second independent reading of reports/captures.
2. Validate the full 901-sample CPU/GPU contents and readiness/bracketing, then pass
   samples parsed from those exact CPU bytes to the sibling/restoration lifecycle
   validator. Both gates consume the same sequence/control objects.
3. Validate all eighteen process observations against the supplied library/command
   declarations. Qualify all eighteen exact report byte strings with unchanged
   frozen-reference checks, including direct current-Python versus frozen-Python
   output comparisons in fixed halves. No diagnostic timing becomes acceptance.
4. Require each gate's ordered receipt digests to match the same sequence. Intersect
   telemetry/lifecycle/process clock-domain bounds. Recheck every read evidence file
   and the input copy for visible metadata changes.
5. **Last input-validation/file-access stage:** call the existing current-file gate
   over all supplied frozen files, observed paths and aliases. Only derived-output
   serialization/publication follows. This orders the final frozen rehash; it does
   not remove intrinsic harness hashing/Git checks or the output qualifier's required
   frozen-reference archive checks.

No validation failure reaches final frozen rehash. A frozen-file failure publishes
no completion. Successful composition writes six derived reports: evidence binding,
telemetry, lifecycle, processes, reported outputs and postflight. The final result
binds their exact hashes/lengths, all consumed input identities, original deadline and
runner-return digest. All outputs are exclusive and private. `completion.pending.json`
is written/closed first, then exclusively hard-linked as `completion.json`; failed
or short publication cannot expose a partial completion marker. Staging/partial bytes
are retained, not removed or overwritten. This is not a host-crash durability claim.

The positive claims are **validation composition** and **frozen rehash after runner
return and all gates**. Complete preparation, live producer provenance, independently
proved runner reaping, hardware restoration, clock enforcement, 642-trace parity and
performance acceptance remain false. Observer-supervisor stdout remains explicitly
post-only; capture hashes do not supply missing producer-inode evidence. Metadata
rechecks are not atomic snapshots or immutable mapped-byte proof.

## Verification

Four new test methods cover composition and rejection. The three archive-consuming
methods use fresh subprocesses to avoid perturbing RSS-sensitive tests; input-only
rejections stay in the parent. The successful case invokes **all actual readers and validators**, including actual
frozen-file hashing, against temporary files: eighteen hash-verified archived codec
reports, a virtual-clock 901-sample serializer stream, synthetic lifecycle/receipt/
mapping records and ELF-magic stubs (never executed). It checks the exact call order,
95-file inventory, six report hashes, private final marker and false scope flags.
This is composed test evidence, not a new collection or historical live-provenance
proof. Existing actual CPU sibling-transport tests remain separate.

Fifteen rejection cases cover runner failure, returned-record/original-deadline/input-
copy disagreement, capture mismatch, full telemetry failure, sibling/restoration
failure, process failure, last-report output failure, frozen optional-file drift,
earlier evidence changed during later validation, transplanted gate receipt digest,
incompatible gate clocks and a false gate success flag. Each asserts the **exact
terminal stage**, absence of completion and no premature frozen rehash. Two further
cases inject partial derived-report/final-marker writes. Every attempted path rejects
reuse without calling its runner again. Invalid digest/schema/duplicate-key inputs
are rejected before invoking the runner or creating evidence.

```sh
PYTHONPATH=tools/bench python -m unittest test_finalize_clock_diagnostic -v
PYTHONPATH=tools/bench python -m unittest test_finalize_clock_diagnostic test_clock_control_evidence test_clock_evidence -v
CARGO_BUILD_JOBS=4 ./tools/check.sh
```

Final **18 focused / 391 reporting tests pass**. The initial green log predates the
additional gate-result digest/clock/flag faults and exact terminal-stage assertions.
Earlier failed/intermediate evidence, including the unexplained lifecycle-fixture
failure from the prior file-check checkpoint, remains retained; no repair is inferred.

| Private evidence under `reports/` | SHA-256 |
| --- | --- |
| `phase2-clock-finalizer-tests-v1.log` (4 pass) | `58c136b7b2a65a4b9d82d3d657153686021300a764de768ae9904ac405a602e5` |
| `phase2-clock-finalizer-tests-v2.log` (18 pass) | `ced87a3014df4ffd7f405f21f4f97d89b91681a56d16225df2ada531bf34341d` |
| `phase2-clock-finalizer-check-v1.log` (391 pass) | `1363f0b3bdcafa410991a8ec27d3e30b56fda4a84880f9757867965db7cc0b1b` |

## Remaining work

A [current upstream snapshot](linux-clock-upstream-verification.md) now verifies the
pinned checkout at `f536960`; it must be rechecked in fresh complete preparation.
A reviewed frozen entry still owes live coordinator/observer-supervisor/GPU role bindings and restoration-aware
ownership/reaping. Default-policy/no-custom-lock/no-controller confirmation,
permissions/enforcement/restoration evidence and a **new quiet window after readiness**
remain required before the fixed clock diagnostic. Its outcome cannot substitute for
the separate unchanged-criteria Linux 78-cell acceptance gate, Apple work or the final
requirement-level Phase 2 audit.
