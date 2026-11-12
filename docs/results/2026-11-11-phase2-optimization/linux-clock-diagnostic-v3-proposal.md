# Linux clock diagnostic v3 — conditional fallback approved, NOT RUN

This is an optional follow-up to [diagnostic v2](linux-diagnostic-v2-results.md),
not an acceptance attempt. **Linux remains 72/78; Apple remains pending.**
No clock change, new benchmark, or permission test has been performed for this
proposal. The user subsequently approved read-only option 1 first, and option 2
only if option 1 provides insufficient data. This is conditional approval of the
proposed temporary 1710 MHz/default-reset experiment, not permission to skip the
original-policy, no-controller, supervisor, permission or fresh-window checks.
Automatic continuation does not establish those prerequisites. The earlier
“yes. go” applied only to the completed v2 run.

**2026-11-12:** [Read-only v1's single attempt failed](linux-readonly-trace-v1-results.md)
at the unchanged 1 GiB capture bound, with no traced report or qualified trial joins.
Its data are documented as insufficient; no retry or larger quota is implied.
This satisfies the insufficiency-review prerequisite only. Default-policy and
no-controller confirmation, permission/enforcement checks, fake-tested restoration
supervision and a new quiet window remain required. No clock command has run.

## Hypothesis and scope

V2 retained approximately 0.177/0.198ms B1 block clusters in both native runtimes,
including CUDA-event spans, despite negligible observed SMT-sibling activity.
GPU command windows contained varying SM clocks, but one-second samples cannot
identify the clock during a block. A supported constant graphics-clock request
could discriminate whether the same spread persists under that intervention.
It is not established necessary or sufficient, and cannot explain CPU-only tails.

Use the unchanged full codec harness, not a new microbenchmark: **18 processes**,
six cyclic repetitions of accepted Rust/candidate Rust/Python, all B1/B2/B4 scopes,
checked loads, five warmups and thirty trials. Tokenizers are deliberately absent:
this experiment targets the recurring GPU block spread and cannot resolve their
acceptance cells. Report all scopes, not just B1. No production/dependency/fixture/
precision/retention changes or recompilation. Use the frozen v2 binary identities.

This is a single-condition diagnostic compared descriptively with historical v2,
**not** a contemporaneous randomized clock-policy A/B test. Neither collapsed
spread nor its persistence alone proves an exclusive cause. Cyclic runtime orders
balance positions, not all carryover pairs. No new acceptance threshold is created.

## Required authorization and launch blockers

Before preparing an executable launch:

1. First complete/analyze read-only option 1 and document why its data are
   insufficient for the stated question. The user's fallback approval covers
   requesting **1710 MHz graphics clocks on this GPU** and resetting to
   **defaults** afterward, subject to all checks below. The read-only capability
   listing does not prove lock permission or enforcement.
2. Establish that resetting to defaults is appropriate. `--reset-gpu-clocks` is
   not a general restoration of an unknown custom lock range. Current sampled
   clocks and deprecated application-clock fields do not establish the original
   requested policy. Require user confirmation of default policy; stop if another
   controller/custom policy is present or the original state is uncertain.
3. Obtain a new quiet 15-minute window and confirm no competing GPU-clock controller.
   Do not stop unrelated processes/services or change desktop configuration.
4. Implement and test a process-local cleanup supervisor **before any mutation**.
   This proposal is not a ready-to-run shell sequence. Tests must use fake commands,
   never a real lock: success, lock failure, comparator failure, timeout, interrupt,
   parent disappearance and cleanup failure. Record exact tests/results.
5. Verify noninteractive permission for both setting and resetting without changing
   credentials, sudo configuration, services, persistence mode or global tooling.
   If privileges/restoration cannot be arranged safely, stop and ask the user.

No memory-clock, power-limit, governor, voltage, fan, application-clock or boost-
policy changes are included. Never use GPU/device reset. The RTX 3060 Ti's known
PCI address is `00000000:01:00.0`; rebind address/UUID/name before launch rather than
assuming enumeration index zero still refers to it.

## Tested restoration transaction primitive — not a launch entry

`tools/bench/supervised_restoration.py::supervise_restoration` is import-only;
there is no GPU CLI or automatic policy action. It takes three reviewed argv lists
(set, one worker, reset), an explicit environment/parent and fresh private outputs.
It requires a dedicated single-threaded Linux `-S` interpreter, default SIGCHLD,
exclusive child ownership and no existing children/subreaper role. All commands,
environment and timeout bounds are checked before creating the attempt directory.

The independent supervisor writes/closes `obligation.json` before attempting set.
Even set failure, output/capture failure or an exception after command reaping
requires reset: set may have partially applied. Set runs at most once, worker at
most once; an interrupted/failed set or orphan prevents worker submission. Owned
worker/session-escaping descendants are cleaned before reset. Parent loss or a
pending interrupt cannot suppress reset. One bounded **cleanup-only** retry is
allowed; initial cleanup failures remain failed even if the second reset succeeds.
Two failed resets or unresolved owned cleanup emit an explicit stderr warning.
The private attempt directory cannot be reused. No frozen-input hashing is done.

Command captures are bounded at 1 MiB and set/reset commands at at most 10 seconds.
The absolute reset-request deadline is at most 600 seconds, fixed **before writing
the obligation**, not extended after set. A default 10-second cleanup reserve
cancels set/worker at 590 seconds. The first reset's owned `Popen` return must be
recorded no later than the absolute deadline; late/missing evidence fails the
transaction without suppressing reset or allowing another workload. The earlier
pre-capture timestamp alone cannot pass this check. See the follow-up below.

**This still is not an integrated protocol launch gate.** The future entry must
bind the fixed 600/10-second settings, clock enforcement, telemetry readiness/full
900-second completion, worker commands, policy/permissions and frozen identities
before any real policy command. No hard real-time guarantee is made: scheduling,
IO or cleanup can overrun the reserve, and command submission is not a timestamp
of the driver's policy action. If owned cleanup fails, reset is still attempted
best-effort with failure retained. Uninterruptible tasks, supervisor SIGKILL and
host failure remain outside successful containment guarantees.

`reset_command_succeeded` means only a complete owned reset command with exit zero.
`hardware_policy_restoration_verified`, `clock_enforcement_verified`, launch
provenance and output qualification remain **false**. No test establishes NVIDIA
permission, enforced graphics clocks, appropriate default policy or CUDA abort
behavior. The user's default-policy/no-controller confirmation is still pending;
automatic continuation is not that confirmation or a new quiet window.

The initial sixteen fresh-process CPU-only tests use a local fake policy file and Python
commands, never `nvidia-smi`: arm-before-set; partial set failure; set timeout;
worker failure/deadline/interrupt; set/worker output overflow; worker SIGKILL with
session-escaping descendants; unrelated survivor; parent disappearance; interrupted
reset; one cleanup retry; two failing/timed-out resets and warning; exceptions
before/after command reaping; cleanup, signal-setup and obligation-write failures;
malformed commands/env/bounds/parent; existing children/role; attempted reuse.
The fake reset itself checks `/proc` to require the worker and escaped child gone
before modifying its fake state. These tests check actual subprocess waiting, not
merely a serialized reaping flag. Staged review added a reset-timestamp failure
regression: initially the timestamp exception bypassed reset. The retained red test
fails; guarded metadata recording now preserves the failure while still attempting
reset, and the green/full suites pass. This is fake-policy evidence, not a live lock.

Commands: `python -m unittest discover -s tools/bench -p test_supervised_restoration.py -v`
and `CARGO_BUILD_JOBS=4 ./tools/check.sh`.

| Retained log under `reports/` | Result | SHA-256 |
| --- | --- | --- |
| `phase2-restoration-tests-v1.log` | Initial 12 tests pass | `6f84dcb0d24b7615eaa3ebbcfa228842cd29bcb9d3949652b5f80bf340e65e47` |
| `phase2-restoration-tests-v2.log` | Expanded 14 tests pass | `9b15bec0092dca1eba2036cfa8d1f148d31d405e9de13c4346d39344a7b89cc3` |
| `phase2-restoration-tests-v3.log` | Expanded 16 tests pass before timestamp regression | `3acef58b191915177a254b8da870b9989c2148a97adb80b700e9e2cd1d370827` |
| `phase2-restoration-check-v1.log` | Initial default checks pass; 259 reporting tests | `fd6ac64264abed9c033feca04d00716a104a3bfc81d8dba9a5201a940bc230ac` |
| `phase2-restoration-clock-red-v1.log` | Reset-timestamp failure bypasses cleanup before fix | `4e9ae95fdcb22e33063bc007abf6890ae7cbcca5e3184aec6bfa22e445fcf6c9` |
| `phase2-restoration-clock-green-v1.log` | Four exception subcases, including reset timestamp, pass | `16fd45ef9cc41df7ddbb6816b1fe52be00a3c9e89f705c5719bab05cee3c0240` |
| `phase2-restoration-tests-v4.log` | Final 16 tests pass | `88d685afcee254fe4db18b9ef75f2c3c1d4b36a0b7645c0bfb4208308e970261` |
| `phase2-restoration-check-v2.log` | Final default checks pass; 259 reporting tests | `ef0bb76238a733ac4918a6e1353fd5334147fcdf7e52bd72bae540d5807b334d` |

### Absolute reset deadline follow-up — CPU-only

Restoration obligation/result reports now use **schema version 2**. The earlier
ambiguous `lock_deadline_monotonic_seconds` result field is replaced with separate
`work_deadline_monotonic_seconds` and `reset_deadline_monotonic_seconds`; both are
persisted before set. `cleanup_reserve_seconds` must be finite, positive and less
than the lock interval. `reset_request_within_deadline` requires the first reset's
owned PID and finite post-`Popen` time within that interval. Reset retry evidence
cannot repair a missed/unverified first-request deadline. Exit-zero reset remains
separate from deadline evidence and does not prove restored hardware policy.

New bounded-child schema-1 receipts add `spawn_returned_monotonic_seconds`, null
if spawn failed or its clock read failed. This is a process-local monotonic upper
bound on spawn submission, not an exact exec/driver-call time or independent
provenance. Historical receipts are not rewritten or retroactively credited.
A clock failure after successful spawn still kills/reaps the owned child. No
model/artifact/workload schemas, fixtures, pins or acceptance limits changed.

Tests now include a real fake worker cancelled early enough for reset; the
pre-set obligation's deadline equality; a delayed reset spawn after its earlier
pre-capture timestamp; absent spawn evidence; invalid cleanup reserves; and a
post-spawn clock exception. Late/missing evidence stays failed even when fake
reset succeeds. Nineteen restoration plus fifteen bounded-child tests pass.
Tests use shortened synthetic intervals/reserves; they do **not** prove the
10-second reserve suffices under CUDA aborts or arbitrary scheduler/IO delays.

Commands (all offline/fake-command):

- Red regression: `PYTHONPATH=tools/bench python -m unittest
  test_supervised_restoration.RestorationTests.test_cleanup_reserve_and_absolute_deadlines_are_armed_before_set
  test_supervised_restoration.RestorationTests.test_late_or_missing_spawn_evidence_cannot_use_earlier_reset_timestamp
  test_bounded_child.BoundedChildTests.test_success_merged_streams_and_signal_restoration -v`.
- Focused: `PYTHONPATH=tools/bench python -m unittest test_supervised_restoration test_bounded_child -v`.
- Default: `CARGO_BUILD_JOBS=4 ./tools/check.sh`.

| Retained log under `reports/` | Result | SHA-256 |
| --- | --- | --- |
| `phase2-reset-deadline-red-v1.log` | 3 tests fail before implementation: new argument/receipt field absent | `c2a575b63d8d8fc4c867f48983466faef409d3532237569c9c140ca47e36acc2` |
| `phase2-reset-deadline-tests-v1.log` | 34 focused tests pass | `8aaa22552b332ab31010479316414f735ee5501e0d08970e7a074377982d26c3` |
| `phase2-reset-deadline-check-v1.log` | Default checks pass; 263 reporting tests | `3bc43a89743b4360d48162fdb0d7a18648d2518c3b8a3bd436f3b23cfc80c47a` |

No GPU command, permission test, policy change or benchmark ran in this follow-up.
Policy/controller confirmation and integrated readiness remain pending.

### Existing comparison runner integration audit — no launch

Inspected `tools/bench/run_phase2_linux_comparison.py` at `c82d847`; its diff
against measured v2 `fa040ed` is empty. Source SHA-256:
`1d53a932a0953100e182d16c6d366bdd153f0e6048e81b90d6042346b0661be1`.
Calling only `order` and mocked `validate_observer` under `python -S` confirms
18 codec jobs and rejects a worker initially eligible only on CPU0. No `main`,
subprocess, hardware probe, permission check or benchmark was invoked by the probe.

The actual existing order is **previous/Python/candidate**, then
**Python/candidate/previous**, then **candidate/previous/Python**, repeated twice.
The prose list of three runtimes above is not a different order authorization.
Preserve this existing rotation, full harness/scopes and six fresh processes per
runtime; copied Python reports are still not independent processes.

A direct use of this CLI as the restoration worker is **not ready**:

- Before measurement it hashes native binaries and tools and runs Git, `rustc`,
  `lscpu`, `nvcc`, GPU queries and both Python package inventories. Under set →
  worker → reset these would occur after the clock request. Preparation must be
  separated before telemetry/intervention, not performed inside the locked worker.
  The explicit minimal PATH does not establish availability of those tools.
- It requires initial eligibility on CPU2 and CPU0, then pins its launcher to CPU0
  before spawning monitors; passing an already CPU0-only worker fails preflight.
  Arrange the verified startup mask deliberately, not by relaxing CPU2/CPU0 checks.
- It inherits the whole environment and adds `TOKENIZERS_PARALLELISM=false`.
  It checks only three injection variables, not the stronger isolated launch
  contract. Freeze exact parent/child environments and document differences from
  inherited-environment v2; do not silently compare them as identical conditions.
- Workload `subprocess.run` calls lack per-child bounded captures, cancellation,
  owned process observations and monotonic receipts. An outer subreaper/deadline
  alone supplies neither per-job identity binding nor qualification of 18 reports.
- GPU query/pmon subprocesses are owned inside the runner and checked for premature
  exit only after all jobs. Their lifetime is not the required full 900-second CPU
  telemetry collection, and observer failure does not immediately stop later jobs.
  Compose telemetry readiness, live failure cancellation and full sampler cleanup
  explicitly; do not nest duplicate unreviewed monitor sets.
- `environment.json` includes raw candidate Git status and a historical
  “No clock/power/global tooling changes” limitation. Clock-run evidence must stay
  private where required and accurately label the intervention; do not publish raw
  session filenames or retain the unchanged no-clock-change claim for this run.

These are integration requirements, not reasons to rerun v2/read-only v1 or change
workloads, gates, policy or permissions. No runner code was changed in this audit.
The import-only restoration primitive and its tests do not cover these gaps.

Offline probe: `PYTHONPATH=tools/bench python -S -` imported `order` and
`validate_observer`, asserted the six rotations above, and used
`unittest.mock.patch('run_phase2_linux_comparison.os.sched_getaffinity',
return_value={0})` to require rejection of `validate_observer(2, 0)`. Retained log:
`reports/phase2-clock-runner-audit-v1.log`, SHA-256
`4a6e0ae95965dc4213f5f1c34946231f6e5570090c42e70684a7938cedab0633`.
Static source review supplies the other findings, not a live failure demonstration.

### Pure command-plan extraction — no launch

`run_phase2_linux_comparison.py::plan_jobs` now builds the same ordered argv/cwd/
report destinations without reads, hashing, probes, affinity changes or spawning.
It takes absolute roots and explicit options, returns fresh JSON-serializable job
records, and can receive an absolute `taskset` path for later identity binding.
Relative reference-script/fixture arguments deliberately retain their previous
runtime cwd. It rejects malformed counts/workloads/CPU/options but does **not**
verify files, permissions, environments, report-path freshness or binary identity.
It is not a launch manifest or authorization. No clock CLI was added.

The existing CLI now consumes this plan, built before its output creation and
telemetry startup. Its metadata probes, inherited environment, monitors, child
execution and report-copy behavior are otherwise unchanged; **the CLI remains
unsuitable for direct use as the locked worker**. This extracts the command-only
surface needed by preparation; it does not yet implement the isolated bounded
18-job worker or connect restoration/telemetry/enforcement/final qualification.
The runner is no longer byte-identical to v2: new source SHA-256
`6b3a8f8bbbfbbc6a77e76bea8d1c8191b94644918b359ff5659aac16e046a375`.
Historical pins/results are untouched; future preparation must bind this revision.

Three fake-CLI tests passed **before** extraction and pass afterward, checking
exact commands, cwd, order, report paths/copies and forwarded environment against
an independent expected-command oracle: 18 codec jobs; 45 jobs across all five
workloads without affinity; 18 tokenizer-only identical-binary-control jobs.
All CLI subprocesses and affinity mutations are mocked; fake report bodies are
not numerical qualification. Additional tests reject planner IO/probes/spawns,
compare the pure plan with CLI calls, exercise explicit taskset and independent
mutable results, and reject invalid options without outputs. No real benchmark or
GPU command ran. This establishes command-construction compatibility, not live
latency/resource equivalence or completion of any acceptance cell.

Commands:

```sh
PYTHONPATH=tools/bench python -m unittest \
  test_linux_comparison_plan.ComparisonPlanTests.test_existing_cli_codec_eighteen_commands_and_python_copies \
  test_linux_comparison_plan.ComparisonPlanTests.test_existing_cli_all_workloads_and_unpinned_commands \
  test_linux_comparison_plan.ComparisonPlanTests.test_existing_cli_identical_tokenizer_control_keeps_candidate_binary -v
PYTHONPATH=tools/bench python -m unittest \
  test_linux_comparison_plan.ComparisonPlanTests.test_plan_is_side_effect_free_and_matches_existing_cli -v
PYTHONPATH=tools/bench python -m unittest test_linux_comparison_plan test_linux_comparison_affinity -v
CARGO_BUILD_JOBS=4 ./tools/check.sh
```

The first two commands ran before implementation (legacy green/new helper absent);
the final two ran after implementation. The existing affinity suite includes a
real CPU-only child inheritance check, not a GPU test.

| Retained log under `reports/` | Result | SHA-256 |
| --- | --- | --- |
| `phase2-comparison-plan-legacy-v1.log` | 3 fake-CLI tests pass before extraction | `603a2b75a78ed9d0075972afe4d4fc68ec4aa8016b19d598f41ba77280c230df` |
| `phase2-comparison-plan-red-v1.log` | New planner test fails: missing `plan_jobs` | `0b2c0a286c164f3ec96d842c9b3c5ef02db2b3df869be9d9b7db6a5557c4c6d8` |
| `phase2-comparison-plan-tests-v1.log` | 6 planner + 6 affinity tests pass | `1043c6ebd0029011f951db79229c017a18f0e0a44108596d53b223f9c780506a` |
| `phase2-comparison-plan-check-v1.log` | Default checks pass; 269 reporting tests | `ffd2beaf087f1369cc888c52fe8b2e45768130c4503385c2e3195f792cd793e7` |

### Observer-failure cancellation hook — CPU-only

The import-only restoration primitive now accepts `external_cancel_check`, a
trusted **nonblocking** Boolean predicate. It is checked before arming, before/
during set and worker capture, and again after worker receipt construction. A true
result is latched: a later healthy reading cannot resume work. Exceptions or
non-Boolean results latch failure and retain the error type. A noncallable hook is
rejected before output creation. Reset and its cleanup-only retry never consult
this predicate, so observer loss cannot suppress the armed restoration obligation.
Schema-2 results add `external_cancel_check_enabled` and `external_cancelled`;
no historical receipts or model/artifact schemas are changed.

Three new CPU-only tests cover healthy observation, cancellation/invalid status
before arming, transient observer loss during set/worker, predicate exceptions,
and loss arriving after a **complete** worker receipt. Fake commands write a local
failure marker; reset checks actual worker reaping as before. Tests assert the
predicate is not called during reset and a transient failure remains latched.
This is a cancellation integration hook, **not** an implemented/authenticated
telemetry monitor: readiness, full 900-second coverage, bounded callback behavior,
outer supervision and final collection validation remain the caller's obligations.
No GPU commands, clock changes, permission tests or benchmarks ran.

Commands: red focused test `PYTHONPATH=tools/bench python -m unittest
 test_supervised_restoration.RestorationTests.test_external_observer_loss_stops_set_or_worker_but_not_reset -v`;
then `PYTHONPATH=tools/bench python -m unittest test_supervised_restoration -v`
and `CARGO_BUILD_JOBS=4 ./tools/check.sh`.

| Retained log under `reports/` | Result | SHA-256 |
| --- | --- | --- |
| `phase2-restoration-observer-red-v1.log` | New hook test fails before implementation | `4cbf2a617e78a834cfe1b3dcfc93848e797d00e96c76d7fe8ab0fea8d675ce68` |
| `phase2-restoration-observer-tests-v1.log` | 22 restoration tests pass | `bd1076378b771a50809fadca1fbb3ff2a21a08d36bf881637496e2869e26eb66` |
| `phase2-restoration-observer-check-v1.log` | Default checks pass; 272 reporting tests | `151259ca7b0e6b05f4b106a3c66b8d594bfd99c5fbfc849bbb6da13244e82660` |

### Bounded eighteen-job codec worker — CPU-only validation

`tools/bench/clock_codec_sequence.py::run_sequence` now consumes the shared pure
planner for six codec-only cyclic repetitions. It is import-only: no clock command,
CLI, telemetry monitor or executable launch contract is supplied. Before creating
outputs or spawning a child it checks the complete order, CPU2 taskset/report
arguments, all eighteen executable paths and mandatory observer objects. Factories
receive copied job/environment values. File existence is not frozen identity;
preparation still owes source/binary/payload/library hashes and clean-state checks.

The worker takes the restorer's **absolute work deadline**, at most 590 seconds
away, rather than starting another 600-second allowance. It runs each child through
owned bounded capture with mandatory observation and cancellation. Child, report,
observer, cancellation or callback failure stops further submission; no retry.
A complete child receipt cannot override failed observation or later cancellation.
Reports retain their original runtime-directory paths. Python-copy destinations
are recorded but copies are deferred until post-collection analysis; they must not
be counted as extra processes. Result fields explicitly leave provenance, numerical
qualification and whole-run supervision false.

Prospective clock-worker safety bounds are **1 MiB captured stdout/stderr per job**,
**16 MiB post-exit report-read limit**, and **1 GiB free-space headroom** checked
before output creation and each job. These are not a live report-write/disk quota
or an overall observation-output cap. They do not revise the failed read-only v1
quotas or authorize another attempt. Child environments are explicit PATH/HOME/
C-locale/PYTHONNOUSERSITE values plus `TOKENIZERS_PARALLELISM=false`; unknown keys
are rejected. No frozen-file hashing or metadata subprocess probes occur in this
worker. CPU0 worker placement and bounded observer callbacks remain entry obligations.

Nine CPU-only tests substitute fake Python commands for native/reference programs
and a waitable-child/CPU2 observer for full identity/library observations. They run
all 18 children, verify exact isolated child environments, owned reaping, report
binding, private/fresh outputs and deferred copies; cover child/report/capture/
deadline failures, live/late cancellation, observer/callback failure, and complete
prevalidation of a bad eighteenth observer/executable. Fixed-order, environment,
deadline and free-space failures prevent submission. No real codec, GPU policy,
quality measurement or hardware performance comparison ran.

Commands: `PYTHONPATH=tools/bench python -m unittest test_clock_codec_sequence -v`
and `CARGO_BUILD_JOBS=4 ./tools/check.sh`. Both passed on the first retained run.

| Retained log under `reports/` | Result | SHA-256 |
| --- | --- | --- |
| `phase2-clock-sequence-tests-v1.log` | 9 fake-workload sequence tests pass | `dd308fa1fc6e3b3780f212a4dbe778b39d2e15e32eacdc5ec7c08dee65fa7151` |
| `phase2-clock-sequence-check-v1.log` | Default checks pass; 281 reporting tests | `f720fc2427e19da93dadb94eaf9734b5a80a33844ae06f4db6b548c82380ad3a` |

Remaining: bind this worker to the frozen preparation and the actual supervisor's
obligation/deadline; enforce failed-result exit status in its future process entry;
compose telemetry readiness/loss/full 900 seconds, restoration, process/lifecycle
validation and all 18 output qualifications. Policy/controller confirmation,
permissions/enforcement and later fresh availability are still missing. These
component tests do not make the clock diagnostic launch-ready or close Linux cells.

### Composed restoration/worker tests — still no live launch

`test_clock_restoration_composition.py` now runs the actual restoration supervisor
and bounded codec worker together in fresh `-S` interpreters. Policy commands
only create/remove a local fake-lock file; all eighteen workloads and process
observers are substituted CPU-only fixtures. The test worker reads the actual
supervisor obligation's work deadline, passes it unchanged to the sequence, and
exits nonzero for a failed sequence. This is test transport, not the production
role/obligation authentication or launch entry.

Four tests cover all 18 successful jobs followed by reset; failure of job 2 with
no replacement/later job; observer loss during job 2; and expiry of the shared
deadline during job 1. The fake reset checks that every recorded workload PID is
already gone. The dedicated supervisor checks no owned children remain and its
subreaper role is restored; an unrelated child survives the observer-loss case.
Deadline equality and the worker's observed parent PID are checked against the
supervisor's records. A worker killed mid-cleanup may leave its sequence result
absent; that remains failure, not fabricated completion. No hardware-policy,
continuous observation or native output qualification follows from these tests.

**Historical finding at `c7d9432`:** source inspection confirmed that the read-only
entry uses a 10-second TERM grace and `bounded_child` then capped grace at ten
seconds. Two permitted reset attempts can each take ten seconds, before their additional owned-child/pipe cleanup. Do not transplant that
read-only outer wrapper as a clock-restorer wrapper: it could kill the supervisor
before its cleanup retry. The integrated clock parent needs a reviewed/tested
restoration-aware shutdown budget; these short synthetic tests do not supply one.
Scheduling/IO, supervisor SIGKILL, host failure and uninterruptible tasks retain
their stated limitations. No grace bound or live policy changed in this checkpoint.

Commands: `PYTHONPATH=tools/bench python -m unittest test_clock_restoration_composition -v`
and `CARGO_BUILD_JOBS=4 ./tools/check.sh`.

| Retained log under `reports/` | Result | SHA-256 |
| --- | --- | --- |
| `phase2-clock-composition-tests-v1.log` | 4 composed CPU-only tests pass | `88096ba14524910699b7a67dde0548b0e249e5f630bbbb61afe57d03c441ff25` |
| `phase2-clock-composition-check-v1.log` | Default checks pass; 285 reporting tests | `51d67b45277527c69d6768a03da03ab5cd5f36f708ad45ebbb4db52edb43a851` |

### Restoration-aware outer capture and grace pipe draining — CPU-only

`clock_restoration_capture.py::capture_restorer` now supplies an import-only outer
capture with a **660-second supervisor budget**, **60-second TERM grace** and
**1 MiB capture limit**. It requires a Boolean cancellation monitor and emits a
verification warning for failed/cancelled capture or capture exceptions. A receipt
is not proof of restored hardware policy. The caller still owes frozen role/parent
binding, validated restoration evidence and full telemetry/output qualification.
There is no CLI or GPU policy command in this helper.

The generic capture primitive now permits explicit grace up to 60 seconds; its
**default remains zero**, and the **read-only entry remains fixed at ten seconds**.
Set/reset command limits remain ten seconds each with at most one cleanup-only
retry; work cancellation remains 590 seconds and reset request deadline 600.
Configured waits for an interrupted command, up to four owned cleanup scans and
two reset attempts can total 40 seconds (three wait/drain pairs × 4 seconds,
four scans × 2 seconds, two commands × 10 seconds). Sixty seconds reserves margin;
it is not a bound on arbitrary scheduling, filesystem IO or uninterruptible tasks.
The 660-second outer budget is not permission to extend the inner reset deadline.
Final kill/wait/drain may add time after grace, and host/supervisor failure remains
outside guaranteed containment. A full clock entry must bind these budgets.

A new red test exposed a separate grace bug: the old wait loop stopped reading
stdout, so a child trying to flush 100,000 bytes from its TERM handler blocked
and was killed (`returncode=-9`, exhausted grace). Both ordinary and over-cap
captures failed that test. Grace now drains the pipe while retaining capture caps;
excess bytes are discarded, and a failed capture writer disables further writes
but still allows draining. Errors/cancellation remain failed receipts even when
child shutdown finishes. This does not repair or reinterpret historical captures.

Three new bounded-child tests cover normal/over-cap grace draining, write-error
shutdown and selector-error cleanup. Review of the drain change exposed a selector
`OSError` being misclassified as lost child ownership, skipping reaping. That red
regression explicitly cleaned its still-owned fake child; local selector-error
handling now retains capture failure without abandoning owned cleanup. Actual
`waitid` ownership loss still prohibits a later numeric group signal. Four outer-capture tests cover normal exit with the fixed budgets;
mandatory/invalid cancellation and capture-setup failure; bounded escalation; and
an actual independent restoration supervisor whose two fake reset commands each
sleep six seconds. The first reset fails, the cleanup-only retry succeeds, and the
supervisor flushes more than 128 KiB before exit: **over twelve seconds of cleanup
complete without exhausted grace**. The interrupted collection and initial reset
failure remain failed; only the local fake-lock file is restored. No GPU, codec,
permission or live policy experiment ran. The final tests add strict Boolean
monitor-result rejection; initial green evidence is retained too.

Commands:

```sh
PYTHONPATH=tools/bench python -m unittest test_bounded_child.BoundedChildTests.test_grace_drains_output_without_truncating_supervisor_cleanup -v
PYTHONPATH=tools/bench python -m unittest test_bounded_child.BoundedChildTests.test_grace_selector_error_does_not_lose_owned_child_cleanup -v
PYTHONPATH=tools/bench python -m unittest test_bounded_child test_clock_restoration_capture -v
CARGO_BUILD_JOBS=4 ./tools/check.sh
```

| Retained log under `reports/` | Result | SHA-256 |
| --- | --- | --- |
| `phase2-restoration-grace-drain-red-v1.log` | Two grace-drain subcases fail before fix | `ece793baf44ded22670c7d7d9bf7d1f60c369b418afe9b00ddd262f5384953f2` |
| `phase2-restoration-grace-tests-v1.log` | Initial 21 focused tests pass | `c275b621825ecf5924f195d1c1333547bc8f19b6c2d7b967d22027a87a27d830` |
| `phase2-restoration-grace-check-v1.log` | Initial default checks pass; 291 reporting tests | `d5b935d949cadb40d27926a7d7c372ba8b37dbcfcbb44c7e38c9f6709a6138fa` |
| `phase2-restoration-grace-tests-v2.log` | Boolean-guard revision: 21 focused tests pass | `f4d43c6136bcdc0c477bc0a5849b451468fbe94b4804d8b52c0d2791bb2b3cde` |
| `phase2-restoration-grace-check-v2.log` | Boolean-guard revision: default checks pass; 291 reporting tests | `d7727ed798f4d7fffcd812e3982c023d98ec0a76bc93ef381abcd96342c508af` |
| `phase2-restoration-grace-selector-red-v1.log` | Draft drain change misclassifies selector error; owned child not reaped | `7026cedcb5be4a3507e5183df2598e893df52ccbea090ce625db0fea31e3f8b2` |
| `phase2-restoration-grace-tests-v3.log` | Final 22 focused tests pass | `8cd823198d5d79bb61d0e2a7cdd391a90cfe04e35f0a7e9754036618043b2e13` |
| `phase2-restoration-grace-check-v3.log` | Final default checks pass; 292 reporting tests | `811aeb650c14ac9025c202dc273b57dd0c60f73d6fb27a46cf281c3452a9b5be` |

## Readiness-gated restoration lifecycle — CPU-only integration

`clock_diagnostic_coordinator.py::coordinate` now connects the real telemetry
readiness/monitor lifecycle to `capture_restorer`: one restoration supervisor,
660-second capture/60-second TERM grace, no retry. Shared lifecycle extraction in
`readonly_trace_coordinator.py` preserves the read-only entry's arguments, result
fields and default limits. Clock results use `restoration_supervisor_status`, not
`sequence_status`; their provenance/telemetry/coverage/output qualification flags
remain false. The caller still supplies trusted reviewed argv/environment; requiring
an absolute executable and the exclusive restoration-directory argument is **not**
authentication of the role or policy command.

An ordinary failed restorer receipt retains the full sampler lifetime. Observer
failure cancels the outer capture, whose grace allows reset cleanup before return.
Captured SIGINT/SIGTERM propagates to coordinator cleanup rather than waiting out
900 seconds after interruption. An invalid readiness record never invokes the
restorer; action arguments are validated before spawning the observer. No new
frozen-file hashing or metadata subprocesses are introduced during collection.

Six new CPU-only tests use fresh interpreters for the actual coordinator, outer
capture, restoration transaction and bounded codec worker, with fake policy/workload/
identity observers and synthetic telemetry; invalid argv uses an in-process spawn mock. They verify all 18 jobs then reset; job-two
failure stops later work but preserves five-second sampling; observer failure
cancels an in-flight first job and still resets; coordinator TERM still resets;
malformed readiness prevents all policy/work; invalid restorer argv prevents even
observer spawn. Fake reset requires recorded workload PIDs absent. Coordinator
checks retain no direct children and restore affinity. The real 600/590-second
inner budgets and full 900-second telemetry are **not** exercised by these shortened
fixtures. Success is lifecycle composition, not codec or clock qualification.

**Ancestry remains a launch blocker.** Do not put the clock coordinator/restorer
under the read-only generic adoption-and-KILL supervisor: parent disappearance
must leave the independent restorer alive to reset. Conversely, the observer
branch still needs separately supervised orphan/session-escape cleanup. These
new tests exercise ordinary/TERM cleanup, not whole-launch coordinator SIGKILL
with a surviving observer subtree. Freeze/review the role tree and test these
failure paths before launch; independent restoration primitive tests alone do
not close that integration requirement. Final role/deadline binding, actual
per-job image/library observation, full telemetry/output qualification, policy/
controller confirmation, permissions/enforcement and a new quiet window remain.
No GPU command, clock-policy test, hardware sampling or model benchmark ran.
Linux remains **72/78**, Apple pending; no Phase 2 completion is implied.

Commands (private retained logs created exclusively with `umask 077`):

```sh
PYTHONPATH=tools/bench python -m unittest test_clock_diagnostic_coordinator test_readonly_trace_coordinator -v
CARGO_BUILD_JOBS=4 ./tools/check.sh
```

| Retained log under `reports/` | Result | SHA-256 |
| --- | --- | --- |
| `phase2-clock-coordinator-tests-v1.log` | Initial 13 focused tests pass | `5a7ed4dd51b5e19d85d532460f094c5e393f77be0a70bc71d5490200b097dde2` |
| `phase2-clock-coordinator-tests-v2.log` | Final 14 focused tests pass, including coordinator TERM | `906999749f96bed41f892937d9281ccdaf89d68e0210563140585d92fea2656f` |
| `phase2-clock-coordinator-check-v1.log` | Default checks pass; 298 reporting tests | `e7e856729d766626d56fafb67175439651e54c1dea05f3d9c1f3ff4771faa9e1` |

## Independent observer branch and lost-output regression — CPU-only

The clock coordinator now requires a distinct supervised telemetry worker, rather
than directly owning that worker. Its readiness validator checks the existing
sample metadata, then requires the declared worker to be the sole child of the
owned observer supervisor and a non-zombie procfs process with that parent. The
coordinator checks its own waitable supervisor before/after readiness. These are
sampled relationships, not independent role authentication, immutable identity,
continuous liveness or permission to signal an indirectly owned PID. Missing or
inconsistent snapshots fail before the restorer starts. Read-only readiness and
its default limits remain unchanged.

The composed CPU tests now use the existing real `supervised_worker.supervise`
only on the observer branch, separately from `supervise_restoration`:

```text
coordinator
  observer supervisor -> telemetry worker -> CPU/GPU sampler processes
  restoration supervisor -> set / fixed codec worker / reset
```

The observer and restoration supervisors are siblings. No generic adoption-and-
KILL supervisor surrounds the restorer. The launch entry still must construct,
freeze and verify these roles/argv/parents/budgets; this import-only component does
not generate or authenticate them. Test wrapper commands are not a production
entry and their shortened timeouts are not the launch limits.

Nine coordinator tests now cover the previous six scenarios plus:

- Actual coordinator SIGKILL during the first fake workload: both supervisors
  detect parent loss, the restorer reaps work and resets fake policy, the observer
  supervisor reaps its worker plus two session-escaped samplers. Their reports
  remain failed; coordinator completion is absent. A separate test-only adopter
  **waits/reaps**, without generic termination on the successful path; an unrelated
  child survives. PID absence and retained cleanup reports are checked.
- Telemetry worker self-SIGKILL after workload startup: its supervisor adopts/reaps
  both sampler processes; observer-branch failure cancels the clock work and the
  independent restorer still resets. No borrowed numeric worker PID is signalled
  by the coordinator/test parent.
- Mocked procfs readiness rejection for a different/additional child, changed
  parent, zombie/dead state, inconsistent stat PID or invalid declared PID.

The initial parent-loss test wrongly required exit1 for both branches; Python
returned exit120 for one. Parent loss can close the restorer's capture pipe during
final output flushing. The corrected test requires **nonzero exit and retained
failed cleanup evidence**, never inferring restoration from an exit code. The
failed initial log remains retained; no workload or acceptance criterion changed.

Related source review found a genuine lost-output failure: after unsuccessful
reset attempts, a stderr warning exception could skip role restoration and final
`restoration.json`. A new fake-command regression failed before the fix. Warning
write exceptions now append `warning_delivery_error:<type>` and allow final role
cleanup/evidence writing. Two failed resets still leave fake policy unresolved,
record failure and allow no further work; this does not claim the warning reached
the user or that hardware policy was restored. All 23 restoration tests pass.

Commands (exclusive private logs, `umask 077`):

```sh
PYTHONPATH=tools/bench python -m unittest test_clock_diagnostic_coordinator test_readonly_trace_coordinator -v
PYTHONPATH=tools/bench python -m unittest test_supervised_restoration.RestorationTests.test_failed_warning_pipe_does_not_skip_role_cleanup_or_retained_failure -v
PYTHONPATH=tools/bench python -m unittest test_clock_diagnostic_coordinator test_readonly_trace_coordinator test_supervised_restoration -v
CARGO_BUILD_JOBS=4 ./tools/check.sh
```

| Retained log under `reports/` | Result | SHA-256 |
| --- | --- | --- |
| `phase2-clock-observer-branch-tests-v1.log` | 17 tests; initial exact exit1 expectation fails on exit120 | `31e36c1b7f3d891d524a53652c7eeb0c7bfdf0ca0a8898df5b7f93db42c205bf` |
| `phase2-clock-observer-branch-tests-v2.log` | 17 focused tests pass | `05441ad46cb78ef22c233acf2e228624f451459b1b87952e9d3294d22a0ed026` |
| `phase2-clock-warning-pipe-red-v1.log` | Warning exception bypasses final result before fix | `7a327e7d7cdbd4ec7f3ab11a900b5f42b03a4f84e37b0ba881326ef479c91e77` |
| `phase2-clock-observer-branch-tests-v3.log` | All 40 focused tests pass | `b72d761882a91948155846ea6fb366153790c864bee1627c210a9d6911049fbf` |
| `phase2-clock-observer-branch-check-v1.log` | Default checks pass; 302 reporting tests | `4e030a7ae7d8e5fc89e063d22d6bbb9f3b2b020045cb665299344ba9500ee323` |

This tests failure containment only with surviving branch supervisors and fake
policy/workloads/telemetry. It does not guarantee cleanup after supervisor SIGKILL,
host failure, arbitrary IO/scheduling stalls or uninterruptible CUDA tasks. Full
900-second coverage, actual per-job observation/qualification and frozen production
entry bindings remain pending, as do default-policy/no-controller confirmation,
permissions/enforcement and a fresh window after readiness. No GPU command,
permission test, real telemetry or benchmark ran. Linux remains **72/78**; Apple
pending. No acceptance cell closes.

## Fixed-job process observation binding — CPU-only

`clock_process_observation.py::process_factory` now connects frozen job/env/image/
library declarations to the existing real `ProcessObservation`, for use by
`clock_codec_sequence.run_sequence`. It constructs all 18 independent observers
before issuance; exact job dictionaries/environment must match on each callback,
including the eighteenth job before the worker creates outputs or launches work.
Ordinals may be issued only once. Inputs are copied so subsequent caller mutation
cannot silently alter the declared commands, images, libraries or environment.

The factory requires all three runtime image declarations and nonempty required-
library inventories, the CPU2 taskset prefix, fixed runtime/repetition order, and
the isolated environment. Python's declared command path may be a reviewed venv
symlink; its separately declared ELF identity is used for `/proc` image checks.
There is no path resolution, filesystem read/stat, file hashing or metadata
subprocess in factory construction/issuance. Tests prohibit those operations.
File identities must have path/device/inode/size/digest metadata, but the factory
**does not independently verify the declarations**. Frozen preparation must do so
before collection and bind the exact qualified Python image, native binaries,
launcher, forbidden injection identity and required libraries.

The existing observer still samples only the owned waitable child: command/env
match, executable inode, CPU2 affinity, injection absence and executable maps.
Required libraries must appear in complete workload samples. No mapped file bytes
are hashed during workloads/remaining telemetry; small mapping metadata digests
are retained as before. The unchanged observer limits remain 3,000 recorded/transient
samples combined, 50 ms sampling interval, 8 MiB mapping inventory and 16 MiB serialized
observation bound. Missing samples/limits/mismatches fail, not disable observation.
Lazy extra libraries remain post-only; no continuous injection-absence, immutable
mapped bytes, independent provenance or full-output qualification claim follows.
Sampling adds diagnostic host work relative to historical v2; it is not an
acceptance-latency equivalence claim.

Five new CPU-only tests cover:

- Actual real `/proc` observations for 18 fake codec children, in the fixed order,
  under taskset CPU2, with a real required libc mapping and Python argv symlink.
  All receipts bind PIDs/commands, complete workload observations and owned reaping;
  Python copies remain deferred. No Torch/CUDA model or GPU command is used.
- Independent copied state and single issuance, with IO/hashing/probes forbidden
  during construction/issuance.
- Live wrong image/environment or forbidden mapped-identity detection, and absent
  required libraries. Each stops after the first child; missing-library failure
  remains failed even when that child's exit/report succeeded. Environment mismatch
  values are not serialized. The forbidden-library test uses libc as a surrogate,
  not a real injected CUPTI process.
- Last-job cwd/environment/ordinal drift before first spawn, incomplete
  runtime/library/identity declarations and wrong CPU placement.

The broader suite exposed an existing **test fixture race**, not a new observation
failure: its first forced empty mapping snapshot could be discarded as mixed exec
before reaching the startup-incomplete counter. A deterministic synthetic second-
image-read change reproduces that failure. The fixture now maintains its injected
empty mapping until the intended startup classification actually occurs; the
original counter/completion assertions remain. Permanent incomplete snapshots
still fail qualification; later full samples are still mandatory. No production
observer logic, time allowance, numerical tolerance, sampling or benchmark retries
changed. Initial and deterministic red evidence are retained.

Commands (exclusive private logs, `umask 077`):

```sh
PYTHONPATH=tools/bench python -m unittest test_clock_process_observation -v
PYTHONPATH=tools/bench python -m unittest test_clock_process_observation test_clock_codec_sequence test_observe_readonly_process -v
PYTHONPATH=tools/bench python -m unittest test_observe_readonly_process.ProcessObservationTests.test_incomplete_exec_startup_is_not_promoted_without_later_full_sample -v
CARGO_BUILD_JOBS=4 ./tools/check.sh
```

| Retained log under `reports/` | Result | SHA-256 |
| --- | --- | --- |
| `phase2-clock-process-binding-tests-v1.log` | Five new tests pass | `c19dc53ffc2b10668c897794f62e5e033f9e10a1cc53548ec831dcbd468cc76d` |
| `phase2-clock-process-binding-tests-v2.log` | 25 tests; existing startup fault-injection race fails | `3345f8338209ea30ed217f6a9debf2be3eb99ded5222524ad9d0cacc03e98154` |
| `phase2-clock-process-startup-red-v1.log` | Deterministic startup fixture regression fails before fix | `b77aba548ee888005fc642a41d4bc549150fdb11848910efec171be4d9382d1f` |
| `phase2-clock-process-binding-tests-v3.log` | All 25 focused tests pass | `ba56b9b702ec30b71c0724d8e55599689ca4b240378a44cb3388b3743cf832a2` |
| `phase2-clock-process-binding-check-v1.log` | Default checks pass; 307 reporting tests | `458ec1501d152638eb6653dc2a7fd193108a534a692895f552bd22dd47891df6` |

Still not a launch: bind this factory, actual library/image inventories, roles and
absolute budgets into frozen preparation/entry and the final lifecycle/process/
telemetry/output qualification. Default-policy/no-controller confirmation and
permissions/enforcement remain unanswered/unverified; request a new quiet window
only after readiness. Linux remains **72/78**, Apple pending.

## Armed worker role/deadline handoff — CPU-only

`clock_worker_handoff.py::validate_handoff` now supplies an import-only check between
the armed restoration supervisor and the bounded codec worker. It returns the
persisted absolute work deadline; it does not create a new allowance, execute a
command, hash frozen source/payload/library files or authorize a clock intervention.
Caller-provided reviewed role commands/image/environment/private paths still need
binding to the future frozen preparation and production entry.

The helper requires a fresh single-threaded Linux `-S` worker with default SIGCHLD,
no children/subreaper role, isolated environment, expected cwd and CPU0. It reads
at most 1 MiB from `obligation.json` in an owned non-symlink mode-0700 directory;
the leaf must be regular/non-symlink. Duplicate JSON keys/nonfinite values fail.
It checks armed schema 2, the fixed 600-second reset budget, 10-second command
budget/reserve, two maximum reset attempts, and reviewed set/worker/reset argv.
The worker command retains CPU0 taskset explicitly; post-exec interpreter argv
must match the running worker. Parent and worker executable inode/device,
command/environment and affinity are checked from procfs/current process state.
The parent must still be the expected process with unchanged ancestry/start ticks,
and this worker must be its sole child. Unexpected argv/env contents are not
serialized by the helper.

New **additive schema-2 obligation fields** `supervisor_pid` and
`supervisor_parent_pid` are written by `supervise_restoration` before set, from its
own lifetime record. The handoff requires these to agree with live ownership and
ancestry. Historical obligations are not rewritten or credited with these fields.
No model/artifact/workload schema, dependency pin, fixture or tolerance changes.

The work/reset deadline difference must remain exactly the reserved ten seconds;
finite monotonic values, the arming clock and current time must agree. The supplied
work deadline must still be live and at most 590 seconds away. All checks precede
workload submission. Command/environment/image consistency is not independent producer
authentication; procfs samples cannot prove continuous ancestry/image immutability.
Directory ancestors and declaration provenance remain trusted entry obligations.
Actual set success/enforcement and restored hardware policy are not established
by this helper; neither is a hard real-time deadline guarantee.

Four CPU-only test methods exercise:

- The real default-budget restoration transaction, checked handoff, bounded worker
  and real process-observation factory over all 18 fake codec children. The received
  deadline equals both obligation and restoration result, the ten-second reserve
  is unchanged, all workload observations complete, all children are gone before
  fake reset, and failure/qualification scope flags remain honest. The test runs
  for a few seconds, **not** to the 600-second boundary; it proves no worst-case
  CUDA cleanup time or full 900-second telemetry coverage.
- Nine role/command failures: changed set argv, parent argv/PID/affinity, role image,
  expected environment, self argv, supervisor ancestry or writer PID.
- Four time-contract failures: changed lock budget, inconsistent work deadline,
  expired deadline and pre-arming clock.
- Six malformed/private-input failures: Boolean schema version, duplicate JSON key,
  nonfinite value, symlink leaf, oversized file and public restoration directory.

Each of the nineteen rejected handoffs exits the worker nonzero, creates no sequence
or workload-PID file, and still runs the supervisor's fake reset. Deliberate test
obligation tampering is confined to new temporary fake-policy transactions; no
historical evidence is altered. Policy actions in tests are local file operations,
not GPU utility commands. Frozen file identities are collected before these tests
launch the restoration process, not while fake workloads run.

Commands (exclusive private logs, `umask 077`):

```sh
PYTHONPATH=tools/bench python -m unittest test_clock_worker_handoff -v
PYTHONPATH=tools/bench python -m unittest test_clock_worker_handoff test_supervised_restoration test_clock_process_observation -v
CARGO_BUILD_JOBS=4 ./tools/check.sh
```

| Retained log under `reports/` | Result | SHA-256 |
| --- | --- | --- |
| `phase2-clock-handoff-tests-v1.log` | Four test methods pass | `19b8b36a690b364b82dd4699a8d400d64a0763d6ae7cc0f33a8e59488e683c70` |
| `phase2-clock-handoff-tests-v2.log` | All 32 focused tests pass | `5cb2b5908e473e999834ae8866f2c9b68a384a15c3c5504ea1c5426de30fbc14` |
| `phase2-clock-handoff-check-v1.log` | Default checks pass; 311 reporting tests | `51d913d1d62eefff6e5173a57e561ae61b4809096ae95a156444b4cda6cd4667` |

Still required: freeze/verify preparation, source/tools/actual image/library
inventories and all production role commands; bind the tested sibling-supervisor
entry and handoff; qualify full telemetry/lifecycle/process/output evidence with
frozen rehash last. Policy/controller confirmation and permissions/enforcement
remain missing, followed by a new quiet window only after readiness. No GPU
command, permission test or benchmark ran. Linux remains **72/78**, Apple pending.

## Python frozen-preparation coverage audit — metadata only

The [offline Python audit](linux-clock-python-preparation-audit.md) verifies all
110 archived v2 members and current agreement with six reports' three source pins
and 36 distribution versions. It distinguishes Torch distribution version from
runtime build tag, and historical source/metadata pins from current-only interpreter,
config and startup-file snapshots. Historical interpreter/library byte equivalence
is not established. Existing native loader preparation does not cover Python;
startup hooks and lazy dependencies remain explicit preparation/observation work.
Preserve intrinsic benchmark hashing and Git checks rather than claiming run-wide
absence; the added collector's frozen rehash still belongs last. Documented metadata
reproduction passes with system Python `-S`, no Torch import or GPU command. This
is not launch readiness or a change to the qualified benchmark.
The subsequent [file-only preparation component](linux-clock-python-preparation-audit.md#file-only-preparation-component)
now freezes the reviewed Python metadata/startup surface without executing it;
seven new synthetic tests and actual metadata-only calls pass. Default checks pass
318 reporting tests. The three current snapshots agree; a retained red/green test
rejects same-byte image inode replacement. Native/Python loader
coverage, current upstream verification, full frozen entry binding and all live
policy/permission/availability gates remain pending.
The subsequent [standalone loader component and actual preparation](linux-clock-loader-preparation.md)
cover six targets, 35 current-only libraries and bounded utility receipts. Native
accepted/candidate resolution agrees; two cuBLAS SONAME paths differ across native
and Python. Torch's explicit ctypes preloads mean these standalone listings must
not be promoted wholesale into required runtime maps. Seven new tests, 22 focused
and 325 reporting tests pass. Preload inventory/required-map review, current upstream
verification, full entry/qualification and live policy gates remain pending.
The [file-only preload follow-up](linux-clock-loader-preparation.md#file-only-preload-candidate-follow-up)
now inventories 16 source-declared patterns/15 ELF candidates in the reviewed site,
including three candidates outside the earlier standalone set. Optional NVTX is
absent in this root; no runtime winner or global absence is inferred. Six new tests,
20 focused and 331 reporting tests pass. Required-map/known-candidate binding,
current upstream verification, frozen entry/final qualification and live gates
remain pending; no Torch import, GPU query, benchmark or policy change occurred.
The [sample-time known-candidate guard](linux-clock-loader-preparation.md#sample-time-known-candidate-guard)
now rejects mismatched prepared path identities, including first observations,
without requiring optional files to load. Clock factories require explicit known
inventories containing every mandatory identity; read-only defaults/schemas stay
unchanged. CPU tests cover optional absence, first-child failure/latching, invalid
inventory and independent no-IO construction; 22 focused/333 reporting tests pass.
Actual inventory/subset compilation, frozen role binding, independent clock
postflight and full entry/qualification remain pending, as do all live policy gates.
The [pure observation-library compiler](linux-clock-loader-preparation.md#compilation-of-the-observation-library-contract)
now joins the retained preparation records: required native 11/11 and Python 10,
known 40 identities, 50 path bindings. All eighteen unchanged commands construct
observers without starting children; no fresh file verification is implied. Five
new tests, 12 focused and 338 reporting tests pass. Independent clock postflight,
upstream verification and fresh full preparation/frozen entry remain pending;
no runtime/acceptance or live policy gate closes.
The [independent process-record validator](linux-clock-process-validation.md) now
checks all eighteen receipts/observations against frozen declarations and the
original deadline, including rehashed bad-map rejection and optional-known identity
checks. Six new tests include forty negative variants and persisted real CPU-fake
sequence evidence; 23 focused/344 reporting tests pass. Full restoration/observer
lifecycle, telemetry, outputs/capture files, final frozen rehash and launch binding
remain separate gates. No actual GPU/clock/benchmark validation occurred.
The [final current-file component](linux-clock-process-validation.md#final-current-file-component)
now checks all supplied frozen files, optional known candidates, mapped paths and
alias targets, retaining unknown libraries as post-only. Nine new tests include
actual eighteen-job CPU-only transport; final 31 focused/353 reporting tests pass.
An initial full-check failure in the existing read-only lifecycle test is retained;
its sequence failure cause is unestablished, telemetry/cleanup completed, and the
assertion now includes sequence details. No guard/budget was changed. This component
does not enforce rehash-last ordering, authenticate/complete the preparation inventory
or qualify restoration/telemetry/outputs. Full finalizer and frozen entry remain
pending, as do current upstream verification and all live launch gates.

## Fixed collection and cleanup contract

After approval and successful supervisor tests, freeze a machine-readable manifest
with this protocol and supervisor hashes, user authorization, UTC time, device
identity, pre-change read-only policy output, repository/status/source/fixture/
payload/Python identities and exact subprocess commands. Recheck v2 native hashes,
accepted checkout cleanliness, CPU 2/3 topology and CPU 0 eligibility. Any mismatch
stops before mutation; do not opportunistically rebuild. Require fresh paths:

- `reports/phase2-linux-clock-diagnostic-v3/` (comparator output)
- `reports/phase2-linux-clock-diagnostic-v3-cpu.jsonl`
- `reports/phase2-linux-clock-diagnostic-v3-prelaunch.json`

The future supervisor must perform this fixed sequence:

1. Start CPU telemetry for exactly **900 seconds**, one-second intervals, workload
   CPU 2 / observer CPU 0; verify metadata and the first sample. Pin diagnostic
   launcher, GPU monitors and cleanup observer to CPU 0. No concurrent analysis,
   builds or debuggers during collection.
2. Arm cleanup **before attempting** the clock request. The cleanup obligation
   starts even if the set command reports failure, since partial application is
   possible. Record command output/return codes, with bounded command timeouts.
3. Request `nvidia-smi -i <verified-device> --lock-gpu-clocks=1710,1710` once.
   On failure, do not benchmark: perform cleanup, preserve evidence and stop.
4. Run the existing comparison runner once with `--cpu 2 --observer-cpu 0
   --repeats 6 --workloads codec`, the accepted checkout and saved baseline binaries.
   No replacement process, exclusion, extension or measurement retry.
5. Request `nvidia-smi -i <verified-device> --reset-gpu-clocks` immediately after
   comparator completion/failure, or at a **600-second lock deadline**, whichever
   is first. The supervisor must stop/reap only its own diagnostic children before
   deadline cleanup; interrupted reports remain incomplete. It must not terminate
   unrelated work. Parent disappearance must also trigger the armed cleanup.
6. If cleanup fails, make at most one bounded cleanup-only retry, retain both
   results and notify the user immediately if reset remains unconfirmed. Do not
   run more benchmarks or report restoration as successful. A supervisor cannot
   guarantee cleanup through host failure; disclose that limitation.
7. Record post-reset read-only state and the reset command result. Do not claim
   arbitrary prior-policy restoration, or infer it merely from a low idle clock.
   Wait for the CPU sampler's full 900 seconds; no favorable-result early stop.

The supervisor implementation must resolve races between normal completion and
watchdog cleanup and verify no child/lock remains active. Its tests and actual
cleanup evidence are launch/delivery requirements, not assumed capabilities.

## Analysis and stop conditions

Verify all 18 commands, report hashes/identities and outputs. Preserve copied Python
reports without counting them as independent processes. Run the existing full-codec
summarizer with `include_execution=True` and retain all per-process median/p95 and
CUDA-event spans, alongside every command's CPU/GPU telemetry and boundary flags.
Use the unchanged 10,000 paired whole-process resamples/seed 20261110 descriptively;
no transplant into acceptance. Retain all timings, including outliers and first-call
costs. Report B1 wall/event process ranges and max/min spread for each native
runtime, including comparison with all six v2 processes.

Report requested versus observed clocks, available clock-event reasons, power and
temperature. Successful set return code alone is not proof of fixed clocks. Missing
counters, observed deviations, insufficient one-second resolution, thermal/power
limits or incomplete telemetry constrain the interpretation; do not discard their
associated timing samples. Memory clocks remain dynamic and are a residual factor.

Outcome must be one of: incomplete/unenforced intervention; spread persists under
the observed intervention; or reduced spread without established exclusive cause.
Report cleanup separately, including any uncertainty. **Every outcome leaves 72/78
unchanged.** No further sampling until favorable. Define any subsequent acceptance
protocol separately, covering all 78 cells and both comparators with unchanged
limits. If no defensible next step exists, stop with evidence, limitations and the
specific user input needed. A separately specified read-only investigation remains
an alternative if clock-policy changes are declined.
