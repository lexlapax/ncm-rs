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

Command captures are bounded at 1 MiB, set/reset commands at at most 10 seconds,
and set/worker share a cancellation deadline at most 600 seconds from arming.
**This is not yet the protocol's complete 600-second reset-request deadline:**
owned cleanup and reset command bounds are additional work. The future integrated
entry must reserve cleanup time, enforce its absolute deadline, and bind actual
clock enforcement, telemetry readiness/full 900-second completion, worker commands,
policy/permissions and frozen identities before any real policy command. This
primitive does not silently extend the protocol deadline or establish launch
readiness. If owned cleanup fails, it still attempts reset best-effort and retains
failure; uninterruptible tasks, supervisor SIGKILL and host failure remain outside
successful containment guarantees.

`reset_command_succeeded` means only a complete owned reset command with exit zero.
`hardware_policy_restoration_verified`, `clock_enforcement_verified`, launch
provenance and output qualification remain **false**. No test establishes NVIDIA
permission, enforced graphics clocks, appropriate default policy or CUDA abort
behavior. The user's default-policy/no-controller confirmation is still pending;
automatic continuation is not that confirmation or a new quiet window.

Sixteen fresh-process CPU-only tests use a local fake policy file and Python
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
