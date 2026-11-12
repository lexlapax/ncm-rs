# Clock restoration-record validation

**Status: offline consistency component only. Linux 72/78; Apple pending.**
`tools/bench/validate_clock_restoration.py::validate` checks a supplied successful
restoration transaction, armed obligation and sequence record against separately
supplied commands, working directory, original work deadline and expected external
cancellation-hook configuration. It performs no file IO, live process inspection,
policy action, package execution or frozen-file rehash.

## Checked surface

- Schema-2 obligation/transaction, schema-1 receipts/sequence, clean status, armed
  state, no latched errors/cancellation/interrupt/parent loss/orphan observation,
  restored subreaper role and no unsupported hardware/provenance/output claims.
- Fixed production budgets: 600-second reset-request window, 10-second command
  timeout, 10-second cleanup reserve, maximum two reset attempts. Work deadline
  equals reset deadline minus ten and matches the independently supplied original
  deadline, transaction, obligation and sequence. No restarted allowance is accepted.
- Obligation writer/parent IDs agree with the supervisor lifetime. Arming clock is
  inside that lifetime, after the start of the original 600-second window and before
  set capture. Commands in the obligation and each receipt match the supplied ones.
- Exactly one successful set, worker and reset, with distinct child PIDs and receipt
  collector identities matching the supervisor. Any reset retry is **not successful**:
  the underlying transaction retains original errors even after cleanup-only success.
- Complete owned-child receipts, including exit/EOF/reaping, exact timeouts/capture
  limits, no termination or observation/report claims, finite spawn timestamps inside
  their receipt envelopes, equal observed/captured byte counts and valid hashes.
  Capture contents are not read or independently verified by this component.
- Nested, ordered set→worker→reset envelopes and reset-start stamp. Set/worker/sequence
  finish within the original work deadline. The first reset's finite owned spawn
  return time must independently fall within the original reset-request interval;
  a green `reset_request_within_deadline` flag cannot hide late evidence.
- Exact cleanup stages `set`, `before_reset`, `reset-1`; no remaining/reaped/signalled
  or errored orphan cleanup may be promoted to successful collection.
- Sequence lifetime belongs to the captured worker PID with the restoration
  supervisor as parent and is enclosed by its worker receipt. UTC/monotonic clock
  brackets share the existing 1,000 ns datetime quantization; no tolerance change.

The result binds all three supplied records by canonical SHA-256. Only
`restoration_record_consistency_verified` is true. This **does not** establish
actual command execution/authenticity, exact driver timing, enforcement, restored
hardware policy, outer reaping, sibling-observer topology or coverage. It is not the
independent eighteen-job process validator, telemetry validator or codec output
qualifier. Those separate gates and the future finalizer must bind the **same**
records; matching a caller-supplied deadline is not authentication of that deadline.
The optional internal cancellation hook is checked against an explicit caller
expectation, not silently required/enabled. Mandatory outer cancellation remains
an independent capture/entry obligation.

## Verification

Six new tests include **52 negative variants** covering failed state/cleanup/retries,
budgets/commands/writer identities, capture scope, finite/late timestamps, clock
consistency and sequence parent/enclosure/deadline. A late-reset fixture preserves
internally consistent clocks and the green deadline flag, but fails the absolute
first-reset deadline. Positive checks reject IO/process execution and verify input
records remain unchanged.

The real CPU-only integration uses the existing validated handoff fixture with
production 600/10/10 budgets, fake set/reset commands and eighteen observed fake
codec jobs. It parses persisted obligation, restoration, received-deadline and
sequence records, validates them, and confirms supervisor/set/worker/reset and all
eighteen child PIDs are absent. It takes seconds, **not** a 600-second boundary test.
There is no actual GPU policy, Torch import, codec benchmark or 900-second telemetry.

```sh
PYTHONPATH=tools/bench python -m unittest test_validate_clock_restoration test_validate_readonly_lifecycle test_verify_clock_processes -v
CARGO_BUILD_JOBS=4 ./tools/check.sh
```

Final **18 focused tests and 359 reporting tests pass**. The initial focused run
reported 24 because a directly imported TestCase caused six existing tests to be
discovered twice. Importing its module instead removed that duplication; no test
was disabled and the initial green log is retained. The previous file-check
checkpoint's unresolved lifecycle failure remains recorded, not declared repaired
by this passing run.

| Private evidence under `reports/` | SHA-256 |
| --- | --- |
| `phase2-clock-restoration-validation-tests-v1.log` (includes six duplicates) | `ffa37d0176f03d630ac95c651214b411c3dbecb61ada95b7ea1fead02c7df620` |
| `phase2-clock-restoration-validation-tests-v2.log` (18 pass) | `4ac1403598045609f793bac83dcab5acaa15830db85a9919ee524e2bce3a9a14` |
| `phase2-clock-restoration-validation-check-v1.log` (359 pass) | `4ff0994d9f8fd081e97943f9b9fa401e546005994eae5dd09173f761a5eac976` |

## Remaining boundary

Bind sibling observer/coordinator/restorer outer lifetimes and readiness, full
telemetry, captures and report/output qualification; enforce frozen rehash last.
Current upstream verification, fresh complete preparation and frozen launch roles
remain pending. Default-policy/no-custom-lock/no-competing-controller confirmation,
permission/enforcement evidence and a later new quiet window remain unresolved.
No acceptance cell or live launch gate closes from this component.
