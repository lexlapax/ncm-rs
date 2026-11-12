# Read-only trace launch readiness — historical prelaunch checklist

**2026-11-12 update:** metadata preparation/comparison passed at `b7f61b5` and the
user confirmed a fresh quiet window. The [single fixed attempt failed](linux-readonly-trace-v1-results.md)
on AT's 1 GiB capture limit; AP/CP completed, nine later jobs were not started.
Full telemetry/cleanup finished. No retry, quota increase or acceptance pass.
The checklist/commands below describe the prelaunch review, not a new launch authorization.

This is a readiness checklist for the selected diagnostic, **not Phase 2
completion or performance acceptance**. Linux remains 72/78; Apple remains pending.
The user selected read-only first, conditional clock control only if insufficient.
A continuation prompt is not confirmation of a quiet window.

## Contract-to-implementation review

| Requirement | Concrete implementation/evidence | Boundary / remaining action |
| --- | --- | --- |
| Unchanged accepted/candidate binaries, source, payloads, fixtures and explicit environment | `prepare_readonly_launch.py`, archive-bound `readonly_trace_preflight.py`; entry repeats full metadata preflight and compares all prepared identities/commands | Current clean-worktree preparation must run after publication; no historical library parity invented |
| One fixed AP CP AT CT / CP AT CT AP / AT CT AP CP sequence, no retry | Existing sequence/finalizer guards; private preparation marker and fresh collection/launch paths | Any failure retains evidence; no repeat-until-green |
| CPU2 workloads; CPU0 entry/supervisor/coordinator/telemetry | Fixed taskset role commands and `observer_affinity(0)`; prepared jobs taskset CPU2; sampler starts eligible on {0,2} then pins {0} | Actual CPU topology/affinity rechecked by preparation and full CPU validator |
| Every job observes executable, command/env, affinity, maps and required libraries | Entry always supplies `process_factory`; native plus injection dependencies required for traced jobs; reader checks actual map identities too | Sampling is not whole-lifetime or immutable mapped-byte proof; incomplete/access-denied startup cannot qualify |
| Independent surviving supervisor, exclusive reaping, parent loss | Fresh `-S` Linux interpreters, one Python/OS thread, default SIGCHLD, no existing children/subreaper; internal roles verify live parent PID/image/command and private-plan digest | Supervisor SIGKILL, host failure and uninterruptible tasks remain uncontained; fake tests do not prove CUDA abort behavior |
| Safe ordinary interruption of outer entry | Outer bounded runner sends TERM with up to 10 s grace for supervisor cleanup before final group KILL/reap; ownership loss never permits a later signal | Grace is additional cleanup time, not a workload extension. Native jobs retain default immediate cleanup |
| Fixed bounds | Sequence 600 s, jobs 120 s, telemetry 900 s, captures 1 GiB/6 GiB and 12 GiB headroom unchanged; coordinator 950 s plus bounded cleanup, supervised worker 970 s, outer capture 990 s plus at most 10 s grace | File/read/polling limits are not filesystem quotas; all failure paths remain failed |
| No policy commands or hidden fallback | Entry exposes only launch and internal roles; telemetry commands are reconstructed/checked; no clock/power/install/rebuild path | Clock fallback remains separately conditional on insufficiency, policy/controller/permission/restoration checks |
| Full telemetry before frozen-input rehash | Real outer runner returns only after owned child reaping; composed telemetry/lifecycle/process/trace/output validators precede `verify_files` | Raw receipt/hash consistency is not independent production; source/command trust remains explicit |
| Source/UUID/timezone/lifecycle/raw outputs bound | Private preparation copy and digest; entry plan; current metadata preflight; 64 consumed raw files/five derived reports plus `entry-result.json` | Final entry success is required, not a partial component `completion.json` alone |
| Output correctness and attribution | Unchanged codec qualifier, source-bound 2,008-query layout, six B1 joins, null warmup timings, all four three-process groups | Instrumented timings are diagnostic; no 78-cell acceptance replacement or 642-trace parity claim |
| Fresh availability | Launch requires explicit `--quiet-window-confirmed` and reviewed preparation SHA | **Still requires a fresh user-confirmed quiet window after real preparation/review** |

The entry's preflight comparison ignores only changing probe timestamps/PIDs and
allowed `.pi/` untracked counts, not heads, pins, loader paths, utility/tool identities,
GPU identity, timezone file/link, command/environment or fixed limits. Raw Git status
stays in private 0700 directories. Nonempty Python path/home/timezone overrides are
rejected in the launch interpreter; children receive only the explicit isolated env.
Internal roles hash only their small metadata handoff, not frozen inputs during jobs
or the remaining observation window.

## Entry and layout

`tools/bench/readonly_trace_entry.py` requires `python -S` and the repository checkout
that produced the reviewed tool inventory. Internal roles are not independent launch
shortcuts: each requires the private preparation copy, plan digest, expected live
parent command/image, exact isolated environment, project cwd and CPU0 affinity.
The telemetry role must use its exact declared output directory.

Proposed fresh paths (none is permission to execute):

- Collection: `reports/phase2-linux-readonly-trace-v1/` (coordinator creates it).
- Reviewed metadata: `reports/phase2-readonly-entry-review-v1/`.
- Independent metadata comparison: `reports/phase2-readonly-entry-review-v2/`.
- Entry: `reports/phase2-linux-readonly-trace-v1-launch/` (entry exclusively creates).
- Readiness working summary: `reports/phase2-readonly-entry-readiness-v1.json`.

Metadata-only preparation, after the commit is clean:

```sh
python -S tools/bench/prepare_readonly_launch.py "$PWD" \
  "$PWD/reports/phase2-linux-readonly-trace-v1" \
  "$PWD/reports/phase2-readonly-entry-review-v1" \
  --candidate-head "$(git rev-parse HEAD)" \
  --gpu-pci 00000000:01:00.0 --gpu-name 'NVIDIA GeForce RTX 3060 Ti'
```

Repeat metadata preparation into the separate `-v2` directory to exercise the
entry's current-versus-reviewed comparison. This runs only bounded Git/loader/GPU
identity probes, never a benchmark or sampler. Review the resulting manifest/hash
and record the checklist before asking for availability. Do not alter the checkout
between the final reviewed manifest and launch; current HEAD/pins are rechecked.

Only after explicit fresh confirmation, the launch role takes the reviewed manifest,
its SHA-256, the new entry directory and `--quiet-window-confirmed`. Do not supply
that flag automatically from an earlier choice or continuation. The outer command
needs enough time for 900-second observation plus preflight/cleanup/analysis; that
does not change the fixed measurement or sequence bounds.

## Tests and retained failures

Ten entry tests cover absent availability, `-S`, fixed preflight comparison,
mandatory observation/injection dependencies, parent command/plan/image rejection,
actual internal-role transport and cancellation under the surviving supervisor,
bounded TERM-to-KILL escalation, ownership loss during grace, and rejection of an
incomplete finalizer. Transport uses actual fresh role processes but **stubs metadata
preflight, coordinator/telemetry workload callbacks and output qualification**; no
GPU/model command executes. Earlier tests separately exercise actual coordinator,
observer, lifecycle, full numerical and composed validators. No test alone is a
real 900-second CUDA collection.

Initial transport fixtures omitted GPU PCI/name fields and failed before spawn;
those fixture fields were supplied, without weakening production guards. A new
negative map test initially passed a traced record missing an injection dependency;
the offline reader now mirrors the entry's dependency union. One full check then
failed only because its old error-text assertion no longer matched; the message
retains its prior prefix, with the stronger dependency check unchanged.

| Retained log under `reports/` | Result | SHA-256 |
| --- | --- | --- |
| `phase2-readonly-entry-tests-v1.log` | Initial 7 tests, 2 fixture failures | `616d41bc43c0d76a8f7f7be8de076547b96d8f316ef09f8b15a130756632c318` |
| `phase2-readonly-entry-tests-v2.log` | 7 pass | `fcc1ca558f7f80985c0a3986bdbc918985fa85439d104f07063fa75e47b3df9e` |
| `phase2-readonly-entry-tests-v3.log` | 9 pass | `d92134de6bb81521d83b3c0a5a19cba325f82fb89f0e061e9dc3504315ee4fcb` |
| `phase2-readonly-entry-tests-v4.log` | 9 pass | `2af5b0ff43dbef8b2cf86341a4cf779464282b6bcfa5f6bdb660722ce7cd0679` |
| `phase2-readonly-entry-tests-v5.log` | Final 10 pass | `80196176da6f965157ccca10f76fe63e66ac1e774c033966308df33b7e659f4c` |
| `phase2-readonly-entry-injection-deps-red-v1.log` | Missing-dependency regression fails before reader fix | `9103c8c7c83deca6ffc515f6ac8825e398edeafba4b4c7fc7ed689b8cc230f97` |
| `phase2-readonly-entry-injection-deps-green-v1.log` | Dependency rejection passes | `f321eee6f7718f5fa6a39e581c9b19ece821491ed2807cb09800d429fa082c78` |
| `phase2-readonly-entry-check-v1.log` | 241 reporting tests pass | `5e89211dc882a74c0bc1a49d137cf736d3798ec8cdeb2839f70a23e9ee37320c` |
| `phase2-readonly-entry-check-v2.log` | 242 tests, one error-text assertion failure | `2858c289506636997578b697ce8dc1e8db83c3c7f39c0070fa6baadc510b4a9f` |
| `phase2-readonly-entry-check-v3.log` | Final default check: 243 reporting tests pass | `d34a680d2839af6a0a1ebef9f4d9bfd30494d60cbd3448b4b35b8a92a9448e4a` |

Commands: `python -m unittest discover -s tools/bench -p test_readonly_trace_entry.py -v`;
`PYTHONPATH=tools/bench python -m unittest test_verify_readonly_processes.ProcessPostflightTests.test_traced_injection_dependencies_must_be_observed -v`;
`CARGO_BUILD_JOBS=4 ./tools/check.sh`. Current metadata evidence and availability are
not established by these logs. Phase 2 completion still requires resolving Linux
acceptance, then actual Apple correctness/performance/resource work and the final
requirement-level audit.
