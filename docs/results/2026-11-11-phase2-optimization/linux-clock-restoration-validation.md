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

## Sibling lifecycle follow-up

`tools/bench/validate_clock_lifecycle.py::validate` now composes the restoration
validator with coordinator, restorer outer receipt, independent observer supervisor,
telemetry worker, readiness, CPU samples and all eighteen sequence receipts. It
performs no file IO, process execution or frozen rehash.

It independently checks:

- Restorer and observer-supervisor lifetimes are **siblings under the coordinator**;
  the telemetry worker belongs only to the observer supervisor. Captured worker
  PIDs, receipt-collector identities, parents and nested lifetime envelopes agree.
  Collector, observer, policy-command, worker and all eighteen workload PIDs are
  distinct. A reparented or duplicated record cannot satisfy a green status alone.
- Clean coordinator/observer supervisor/telemetry status, no orphan cleanup,
  verified non-escalated observer-supervisor reaping and GPU-observer shutdown
  (exit zero or intentional TERM). Observer receipts have no termination, mismatched
  capture accounting or unsupported observation/output claims.
- The restorer outer receipt retains 660-second timeout, 60-second TERM grace and
  1 MiB capture cap, without extending the separately checked 600/590 deadlines.
  Capture accounting/digest and owned spawn timestamp are checked, not raw bytes.
- Readiness binds the actual declared telemetry/GPU PIDs and CPU0 affinity, matches
  the coordinator's readiness digest, is inside the CPU receipt and precedes the
  restorer outer capture. GPU launch/stop envelopes enclose CPU capture. These are
  recorded lifecycle bounds, not first/last GPU measurements or per-call clocks.
- Supplied CPU samples are indexed/ordered and inside their receipt; the first
  precedes readiness and the first/last bracket the entire restorer outer capture.
  All eighteen jobs have fixed runtime/repetition order, complete receipts, enabled
  observation, serial envelopes inside their sequence and CPU-sample brackets.
- Restoration and observer records share one consistent UTC/monotonic offset
  interval with unchanged quantization. Canonical digests bind both branches and
  all job receipts for later composition.

Only `sibling_lifecycle_consistency_verified` and the nested restoration-record
consistency flag are true. Caller-side live reaping/ancestry safety is **not proved**
by supplied records. The validator does not bind observer/outer role argv, image or
environment, authenticate producers, qualify process mappings or inspect captures.
Readiness source/GPU fields and CPU/GPU counters/duration/gaps need their content
validators. It deliberately accepts 2–901 supplied CPU samples for lifecycle
checking; that does **not** pass the separate 901-sample/full-900-second gate.
No continuous telemetry, hardware enforcement/restoration, model outputs or enforced
rehash-last ordering is claimed. The same records must reach every separate gate.

Five new tests cover **40 negative variants** of parent chains/duplicate PIDs,
failed cleanup/status, termination/capture scope, ready digests/timing, sample
brackets/clock domains and missing/reordered/overlapping job receipts. The real
CPU-only integration parses persisted records from the actual sibling coordinator,
restoration and observer supervisors, with fake set/reset commands and eighteen
fake jobs. A new **test-only** fixture mode uses production 600/10/10 restoration
budgets while retaining five-second synthetic telemetry and stub process observation
hooks; other fixture modes are unchanged. The integration reconstructs policy/worker
commands separately and uses the received handoff deadline. Recorded coordinator,
supervisor, observer and job PIDs are absent afterward. This is not real process-map,
codec-output, boundary-duration, GPU or complete finalizer qualification.

```sh
PYTHONPATH=tools/bench python -m unittest test_validate_clock_lifecycle -v
PYTHONPATH=tools/bench python -m unittest test_validate_clock_lifecycle test_validate_clock_restoration test_clock_diagnostic_coordinator test_validate_readonly_lifecycle -v
CARGO_BUILD_JOBS=4 ./tools/check.sh
```

Initial five tests pass; final **26 focused and 364 reporting tests pass** after
adding observer-receipt termination/accounting negatives. No production runner,
budget, policy, benchmark or pin changes. The earlier unresolved lifecycle-fixture
failure remains retained, not declared repaired by these passes.

| Private evidence under `reports/` | SHA-256 |
| --- | --- |
| `phase2-clock-sibling-lifecycle-tests-v1.log` | `6c11d122a07e0cab96d560738e612c3aa32c9baed53c41bdd46855d4bb8fb61e` |
| `phase2-clock-sibling-lifecycle-tests-v2.log` (26 pass) | `f5362b0975daf5fa435cb3bad26a57594044e85e1722e037938113497826f5c3` |
| `phase2-clock-sibling-lifecycle-check-v1.log` (364 pass) | `51e8437ba73128f5435bf12c92aa4820c3e7d49a49c4bb618a38953d0c93dc82` |

Next: full telemetry/readiness content validation, report/output and raw-capture
qualification, and a finalizer binding these gates to the same files before frozen
rehash. Current upstream verification, fresh preparation/frozen roles and all live
policy/permission/availability gates remain pending. Linux remains 72/78.
