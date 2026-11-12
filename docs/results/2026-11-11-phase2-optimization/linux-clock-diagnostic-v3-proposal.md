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
