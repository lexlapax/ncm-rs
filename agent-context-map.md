# Agent context map

Last updated: 2026-11-12 UTC. Read first; load linked context lazily.

## Current state and user decisions

- **Latest authorization (supersedes earlier pending-choice entries below):** user
  approved read-only investigation option 1, followed by clock-controlled option 2
  only if option 1 provides insufficient data. Read-only v1 ran once after explicit
  fresh quiet-window confirmation and failed at the fixed capture limit; no retry.
  Clock fallback is still unrun and needs confirmed default-policy/no-controller
  state, permissions, tested restoration and a new quiet window. No automatic
  continuation establishes those facts.
  [Selected protocol and durable preparation evidence](docs/results/2026-11-11-phase2-optimization/linux-readonly-trace-v1-proposal.md)
  cover fixed sequence/quotas, report receipts, trace joins/output qualification,
  frozen file/environment preflight, loader/Git guards and the worker supervisor.
  Prior metadata checks match 40 pinned files and 14 canonical libraries; clean
  worktrees verified at `156d976`/`3307dff`. Byte/receipt consistency is not live
  provenance; DT_NEEDED listings do not cover lazy driver/JIT libraries. Source/
  lock remain anchored to `d44beaa`, manifests to measured v2 `fa040ed` (retaining
  the approved codec_memory example). Raw Git status logs remain private.
  Telemetry/coordinator/validator prerequisites are implemented and tested with
  synthetic observers and archived v2 CPU compatibility. Ordinary sequence failure
  preserves the full sampler lifetime; observer failure cancels in-flight work.
  The sampler starts with affinity {0,2}, then confirms {0}; other observers use
  CPU0. Explicit environments differ from inherited v2. All 901 CPU samples,
  counter/schedule arithmetic, GPU units/UUID/timezone and 12 receipt brackets are
  checked offline. Gaps remain explicit, not per-call clocks or continuous coverage.
  Exact earlier tests, failures and limitations are in the selected protocol.
  `observe_readonly_process.py` now samples only owned waitable children: executable
  inode, expected command/environment equality, affinity and executable mappings.
  Optional hooks connect bounded capture, sequence and coordinator; observation
  failure stops later jobs. Eleven CPU-only tests include all 12 fresh children;
  default checks pass **196 reporting tests**. Initial failures exposed differing
  kernel-map/path device numbers (28/29): both are now retained, with kernel mapping
  symlink and path-inode checks, not conflated. Direct map_files stat was denied;
  no permissions changed. No mapped-byte/whole-lifetime proof is claimed, and no
  GPU/model run occurred. Follow-up tests fixed silent factory disabling and
  non-atomic exec/exit snapshots: incomplete samples are counted, never promoted;
  later complete workload observation remains required. Startup access denial is
  recorded as not-ready; persistent denial cannot qualify. Detected mismatches
  cannot be erased by a later child exit. Fixed five-case fake-child batches retain
  all four earlier race failures and final five passes. Logs and
  mapping limitations are in the selected protocol.
  Latest: `verify_readonly_processes.py` cross-checks all 12 process records against
  receipts, declared commands/environments, image/injection identities and native
  libraries. Its postflight function rechecks all prepared/declaration/loader/library
  files and current mapped paths, keeping additional libraries explicitly post-only.
  It preserves kernel/path device distinctions and never claims immutable mapped
  bytes. Nine tests include actual CPU-child writer/reader compatibility; default
  checks pass **205 reporting tests**. No GPU/model collection or policy change.
  `prepare_readonly_launch.py` now composes a preparation-only metadata manifest:
  jobs, tools/utilities, loader/Git guards, GPU identity, IANA timezone and limits.
  It requires `python -S`, never creates the collection directory or starts sampling,
  and explicitly leaves readiness/availability/collection false. Eleven offline tests
  pass; default checks pass **216 reporting tests**. GPU/loader/Git probes were mocked
  in these new tests; the real preparation CLI has not run. Source/library rechecks,
  path-alias rejection and partial private evidence are covered. Actual frozen
  manifest/current-clean-state verification and final launch entry remain pending.
  `diagnostic_clock.py` now records explicit UTC/monotonic lifetime envelopes;
  `validate_readonly_lifecycle.py` binds all child/process samples, readiness,
  parent/PID chains and CPU/GPU observer cleanup to supervisor receipts. Six tests
  include twelve actual CPU-only process observations under the independent
  supervisor with five-second synthetic observers; default checks pass **222 tests**.
  An initial generated-fixture NameError and its debug evidence are retained; the
  fixture was corrected without changing production limits. A deterministic
  clock-recording/deadline regression also failed before moving that stamp ahead
  of the final startup guard; the red/green logs are retained. New fields do not
  retrofit historical receipts or prove live provenance.
  `finalize_readonly_trace.py` now composes telemetry, lifecycle, process, trace/output
  and frozen-file validators after a single trusted runner returns. It binds 64 raw
  evidence files and five derived reports, preserves failed attempts and refuses
  reruns. Ten fresh-interpreter tests use real validators over archived outputs and
  synthetic telemetry/maps/traces; full checks pass **232 reporting tests**. Updated
  hashes do not hide bad logits; incomplete/mismatched evidence blocks frozen rehash.
  This is import-only orchestration, not independent live-reaping proof: the final
  entry must still supply reviewed commands and an actual exclusive-reaper runner.
  `readonly_trace_entry.py` now supplies fixed -S/CPU0 roles, live parent-command/
  image checks, exact isolated environments, mandatory native/injection dependency
  observation, actual exclusive-reaper supervision and the composed finalizer.
  Outer cancellation gives the supervisor up to 10 seconds TERM grace before KILL;
  native job limits/cleanup remain unchanged. Ten entry tests include real role
  transport/cancellation with workload/preflight/qualification stubs; full checks
  pass **243 reporting tests**. Initial fixture failures, missing-injection-dependency
  red/green and an error-message regression are retained. No CUDA abort guarantee.
  [Readiness checklist, exact commands and evidence](docs/results/2026-11-11-phase2-optimization/linux-readonly-trace-v1-readiness.md)
  maps requirements to the implementation. Published tooling: `b7f61b5`; two actual
  metadata preparations and launch-time comparison passed at that clean HEAD.
  [Read-only v1 failure and evidence](docs/results/2026-11-11-phase2-optimization/linux-readonly-trace-v1-results.md):
  explicit “quiet window now”; 2026-11-12 00:28:13–00:43:14 UTC, one attempt.
  AP/CP completed; AT exceeded 1 GiB capture, killed/reaped; jobs 4–12 unstarted.
  All 901 CPU samples/full 900-second collection completed; 900 GPU rows. Existing
  separate CPU/GPU content validators pass, not the twelve-job gate. Supervisor
  cleanup complete; eight recorded child PIDs later absent. No retry/quota change.
  Finalizer correctly failed before frozen postflight; no completion/entry result.
  Truncated trace: 1,433/2,008 elapsed queries, no traced report/qualified joins;
  insufficient attribution data, not a clock-cause finding. Initial offline alias
  counting error retained/corrected using existing ELAPSED names. Compact 36-member
  archive published; full 1 GiB capture retained locally by hash, not public raw
  reproduction. Clock-restoration integration and policy/controller/permission
  checks remain before new availability; no clock intervention is ready or performed.
  Supervisor SIGKILL/host failure cannot guarantee cleanup. Linux remains 72/78,
  Apple pending; no GPU-policy/production changes or completion. Post-attempt
  `CARGO_BUILD_JOBS=4 ./tools/check.sh` passes 243 reporting tests; all 36 archive
  hashes and telemetry revalidation pass in clean `git archive b7f61b5` scratch.
  Checks/log hashes and raw-trace reproduction limitation are in the result.
  Latest preparation: `supervised_restoration.py` adds an import-only independent
  set/one-worker/reset transaction, obligation written before set, owned/adopted
  cleanup, parent-loss/interrupt handling and at most one cleanup-only retry.
  No GPU command/CLI is supplied or run. Sixteen fake-policy CPU subprocess tests
  pass, including actual escaped-child reaping before reset, unrelated survivors,
  partial set failure, deadline/overflow, interrupted/timed-out reset, serialization
  exceptions and setup failures. Staged review's reset-timestamp regression failed
  before guarding metadata recording so it cannot bypass reset; red/green retained.
  That checkpoint's 16 focused tests/default checks passed 259 reporting tests.
  Follow-up now separates absolute reset and earlier work deadlines, persisted
  before set: default 600 seconds with a 10-second cleanup reserve. A new owned
  post-Popen timestamp must bound first-reset submission within the deadline;
  late/missing evidence fails even after exit-zero reset. Restoration reports are
  schema 2; bounded-child receipts add a spawn-time field, no historical retrofit.
  Red missing-argument/field tests retained; **34 focused tests and 263 reporting
  tests pass**, including delayed spawn and post-spawn clock-error cleanup.
  [Clock proposal](docs/results/2026-11-11-phase2-optimization/linux-clock-diagnostic-v3-proposal.md)
  records commands/log hashes and limitations: no hard real-time guarantee or
  hardware restoration proof. Next: integrate fixed deadline settings, telemetry,
  enforcement/frozen worker bindings; obtain still-pending default-policy/no-
  controller confirmation and permission evidence, then a fresh window. No real
  GPU command, lock/reset, permission test or benchmark ran in this follow-up.
  Offline runner audit at `c82d847` (then source-identical to v2) confirmed the existing
  previous/Python/candidate cyclic order: 18 codec jobs, not a new runtime order.
  CPU0-only startup is rejected by the required CPU2/CPU0 eligibility checks.
  Source review finds worker-time preflight hashing/probes, inherited environment,
  unbounded per-job calls, internal monitors and misleading no-clock-change metadata
  must be separated/bound before integration. Details/source hash/probe log are in
  the clock proposal. That audit changed no code or live policy.
  Latest: extracted pure `plan_jobs` from the existing runner and made its CLI
  consume the same plan. Three independent fake-CLI command/cwd/order/copy tests
  passed before and after extraction; planner tests reject filesystem IO/probes.
  Six new planner + six affinity tests pass; default checks
  pass **269 reporting tests**. Legacy-green/new-helper-red logs and new runner
  source hash are recorded in the proposal; frozen historical pins are unchanged.
  This is command construction only, not a frozen launch manifest or GPU parity.
  Next: bind the plan into an isolated bounded worker with full telemetry and
  restoration; the existing CLI is still not suitable as that worker. Policy/
  controller confirmation, permission evidence and later fresh window remain pending.
  Further offline preparation adds a trusted nonblocking external cancellation
  hook to restoration: observer loss/errors latch failure, including after a
  complete worker receipt, but never suppress reset. No actual telemetry monitor
  is supplied by the hook. Three new fake-policy tests cover healthy/transient/
  invalid/error/late-loss cases; **22 restoration tests and 272 reporting tests
  pass**. Red/green logs and additive schema-2 fields are in the clock proposal.
  No GPU command or benchmark ran. Remaining work is integrated bounded jobs,
  bound readiness/coverage, enforcement and outer supervision; live policy/
  controller confirmation and permissions remain required before a new window.
  Latest offline component: `clock_codec_sequence.py` executes the fixed 18-job
  pure plan through bounded owned captures, explicit isolated environments,
  mandatory observers and the restorer-supplied absolute work deadline. It stops
  on first failure without retry; Python report copies are deferred. Prospective
  bounds (1 MiB capture/job, 16 MiB post-exit report read, 1 GiB free headroom) and
  limitations are recorded in the proposal; read-only v1 bounds are unchanged.
  Nine real CPU-only/fake-workload tests pass, including all 18 children and
  cancellation/prevalidation failures; default checks pass **281 reporting tests**.
  No GPU/model run or clock command occurred. Still not launch-ready: frozen
  preparation/entry, live observation binding, telemetry/restoration composition,
  final qualification and the unanswered policy/controller/permission gates remain.
  Four new composed CPU-only tests run the real restoration supervisor and bounded
  worker together with fake policy/workloads: exact deadline handoff, 18 successes,
  failed-worker exit, observer loss and deadline abort, owned cleanup before reset,
  and unrelated-child survival. Default checks pass **285 reporting tests**.
  Source audit found an outer-wrapper blocker: read-only TERM grace is capped at
  10 seconds, insufficient for two 10-second reset attempts plus cleanup. Do not
  reuse it for clock restoration without a reviewed/tested shutdown budget.
  Tests use shortened limits, not live CUDA or a production entry. No policy,
  benchmark, permission or grace-bound change in that checkpoint.
  Latest: `clock_restoration_capture.py` adds an import-only 660-second outer
  capture/60-second cleanup grace; generic explicit grace now permits 60 seconds,
  while native default zero and read-only fixed ten remain unchanged. A red test
  exposed stdout backpressure during grace; capture now keeps draining under the
  cap, including after writer failure. A real fake-policy supervisor completes
  two six-second reset attempts and >128 KiB final output after cancellation;
  failed collection/first reset stay failed. Review also caught a selector-error
  path skipping owned reaping; its red/green regression is retained. Final
  **22 focused tests and 292 reporting tests pass**. Proposal/HANDOFF retain red/green logs and limitations.
  No GPU/policy/permission/benchmark run. Still need frozen role/budget binding,
  full telemetry/restoration composition, final qualifications and the unanswered
  default-policy/no-controller confirmation before readiness and a new quiet window.
  Latest offline integration: `clock_diagnostic_coordinator.py` gates restoration
  on telemetry readiness and now requires a separate observer-supervisor branch.
  Readiness PID must be its sole live child in sampled procfs relationships; not
  independent provenance or continuous liveness. Nine clock-coordinator CPU tests
  include two surviving sibling supervisors after coordinator SIGKILL, telemetry-
  worker SIGKILL/session-escaped sampler reaping, unrelated-child survival and
  prior readiness/failure/TERM cases. Generic adoption-and-KILL must not surround
  the restorer. The frozen production entry must still bind this tested role tree.
  A lost warning-pipe regression initially skipped final restoration evidence;
  warning errors now remain failed while role cleanup/result writing proceed.
  An initial test's exact exit-1 assumption failed with Python exit120 after parent
  loss; it now requires nonzero plus actual retained cleanup evidence. All failures
  retained in the proposal. **40 focused tests and default checks with 302 reporting
  tests pass**. Synthetic shortened telemetry/fake policy only; no GPU command,
  permission test or benchmark. Read-only defaults unchanged. Still owed: frozen
  roles/deadlines, actual per-job observation and full telemetry/output qualification;
  default-policy/no-controller confirmation, permissions and a later fresh window.
  New `clock_process_observation.py` prebuilds 18 independent real process observers
  from copied declarations, with per-runtime image/library identities, exact job/env
  binding and one-time ordinal issuance. It preserves Python argv symlink spelling
  separately from ELF identity and performs no construction-time IO/hashing/probes.
  Five new CPU-only tests exercise actual `/proc` observation across all 18 fake
  jobs, live mismatches, missing libraries and prelaunch drift. **25 focused tests
  and default checks with 307 reporting tests pass**. A broader run exposed an
  existing startup fault-injection race; deterministic red/green coverage fixes
  only the fixture, preserving production sampling rules/assertions. Logs/hashes
  are in the clock proposal. Frozen hardware/entry binding and postflight remain
  pending; no GPU command, policy action or benchmark. Linux stays 72/78; Apple pending.
  Latest: `clock_worker_handoff.py` validates a bounded private obligation, fixed
  600/590/10-second budget relationship, reviewed commands and actual worker/parent
  argv/image/env/CPU0/ancestry before returning the original absolute deadline.
  Schema-2 obligations now add supervisor/parent PID fields; no historical rewrite.
  Four CPU-only test methods exercise all 18 jobs with real observations under
  default restoration budgets, plus nineteen rejected handoff cases with reset
  and zero workload submissions. **32 focused tests and default checks with 311
  reporting tests pass**; exact evidence/limitations are in the clock proposal.
  No GPU/policy/permission/benchmark run. This is a tested handoff component, not
  the frozen preparation/production entry or full qualification; those and the
  unanswered policy/controller/permission/fresh-window gates remain pending.
  [Offline Python preparation audit](docs/results/2026-11-11-phase2-optimization/linux-clock-python-preparation-audit.md)
  verifies all 110 v2 members, six Python metadata sets, three historical source
  pins and current 36 distribution versions. No historical interpreter/library
  byte equivalence is established; five current-only file hashes remain snapshots.
  Torch distribution version 2.15.0 differs from runtime build 2.15.0+cu132; retain
  both. Current pip is extra to the benchmark lock; the existing .pth startup shim
  must be accounted for, not disabled. Preserve intrinsic harness hashes/Git checks
  while forbidding added collector probes/early rehash. Documented offline
  reproduction passes; no Torch import, GPU command or benchmark. Docs-only work;
  default checks not rerun (prior fef3b08: 311 reporting tests). Next: explicit
  current-only Python/loader/startup inventory in frozen preparation, then entry
  binding and final qualification. Linux 72/78 and all live authorization gates remain.
  Latest: import-only `prepare_clock_python.py` now binds the seven historical
  records, three source pins, current venv/ELF/link chain, distribution metadata,
  requirements lock, static Torch build tag and explicitly reviewed startup files.
  It executes no interpreter/package/probe and keeps current-only/provenance gates
  false. Seven synthetic tests cover drift, unsupported inputs, inventory races and
  bounds; **12 focused tests and default checks with 318 reporting tests pass**.
  Three actual file-only component preparations agree: 3 historical sources, 43
  current-only files, 37 distributions (extra pip retained), 3 interpreter links;
  launch_ready=false. A retained red/green case now rejects same-byte interpreter
  inode replacement during preparation; final checks bind device/inode/path too.
  Exact commands/hashes are in the Python audit. No Torch import,
  GPU query, clock action or benchmark. Remaining: Python/native loader inventory,
  current upstream check, frozen full entry/role bindings and final qualification;
  policy/controller/permission/fresh-window gates remain unanswered or unverified.
  Latest: [standalone loader preparation](docs/results/2026-11-11-phase2-optimization/linux-clock-loader-preparation.md)
  adds `prepare_clock_libraries.py`: all-six-target prechecks, bounded glibc listings,
  canonical roots/library bounds and final device/inode/path rechecks. Seven new
  tests; **22 focused and 325 reporting tests pass**. Initial fixture-path failure
  retained. Actual preparation: six complete utility receipts, 35 current-only
  libraries/3,547,116,642 bytes; native accepted/candidate resolution agrees.
  Native/Python differ on two cuBLAS SONAME paths. Reviewed Torch startup explicitly
  preloads CUDA libraries, so standalone listings are NOT its runtime mapping
  contract; no environment/library changes or performance attribution. Next:
  explicit preload-candidate inventory/required-map review, current upstream check,
  frozen entry and final qualification. No Torch import/GPU query/clock/benchmark;
  launch_ready=false, Linux 72/78, Apple pending; live policy gates unchanged.
  Latest: file-only `prepare_clock_preloads.py` inventories the three reviewed Torch
  preload layouts in one site-packages root without importing/executing candidates.
  Six new tests; **20 focused and 331 reporting tests pass**. Actual inventory:
  16 source entries, 15 ELF candidates/1,956,767,857 bytes; optional NVTX pattern
  absent here, no ambiguous matches. Twelve identities overlap the standalone
  inventory; cuSOLVER/nvJitLink/NVRTC-builtins add three current-only files (38-file
  union across preparations, not a new simultaneous snapshot). Runtime selection,
  complete sys.path/closure and launch readiness stay false. Loader document now
  records a proposed mandatory core-map subset versus frozen optional candidates.
  Next: implement/test required-map and known-candidate binding, current upstream
  check, frozen entry/final qualification. No GPU query/Torch import/benchmark or
  policy change; live authorization/availability gates and Linux 72/78 remain.
  Latest: known-candidate sampling rejects first-sample identity drift without
  requiring optional presence; observed fields cannot override declarations.
  `clock_library_contract.py` now purely joins prepared records and selects native
  required mappings plus interpreter/_C/core-Torch mappings. No IO/hashing/probes.
  Retained-record compilation: required 11/11 native, 10 Python; 40 known identities,
  50 path bindings; all eighteen commands construct observers without live work.
  Latest: [clock process-record validator](docs/results/2026-11-11-phase2-optimization/linux-clock-process-validation.md)
  independently checks eighteen receipts, mapping digests/identities, mandatory and
  optional candidates, sequence-local clocks/deadlines and declared injection absence.
  Six new tests include 40 negative variants and persisted writer/reader evidence
  from eighteen real CPU-only fake jobs. **23 focused and 344 reporting tests pass**.
  This is not producer authentication, restoration/telemetry/output qualification
  or final file rehash. Latest: `verify_clock_files.py` rechecks supplied frozen
  files, known libraries/images, alias targets and current mapped paths, with unknown
  libraries explicitly post-only. Nine new tests include real eighteen-job CPU-only
  observer→validator→file-check transport; **31 focused/353 reporting tests pass**.
  The first full check failed an existing read-only lifecycle sequence; cleanup and
  telemetry completed, but its assertion omitted sequence details. That diagnostic
  is now included; one focused and one full rerun pass, cause unestablished/red log
  retained. No guard/budget changes. [Scope and evidence](docs/results/2026-11-11-phase2-optimization/linux-clock-process-validation.md#final-current-file-component)
  distinguish a file-check component from enforced finalization ordering.
  Latest: [restoration-record validation](docs/results/2026-11-11-phase2-optimization/linux-clock-restoration-validation.md)
  independently checks the armed obligation, 600/10/10 budgets, original deadline,
  set/worker/reset command/clock order, first-reset spawn and cleanups. Six new tests
  include 52 negatives and persisted default-budget fake-policy/eighteen-CPU-job
  transport; **18 focused/359 reporting tests pass**. Initial focused discovery
  duplicated six imported tests; module import corrected it, original log retained.
  Latest: [sibling lifecycle validation](docs/results/2026-11-11-phase2-optimization/linux-clock-restoration-validation.md#sibling-lifecycle-follow-up)
  binds that transaction to coordinator/observer-supervisor/telemetry parent chains,
  readiness clocks/digest, outer 660/60 capture, CPU-sample brackets and eighteen job
  receipts. Five new tests include 40 negatives and real sibling fake-policy/default-
  budget transport with five-second synthetic telemetry/stub process observations;
  **26 focused/364 reporting tests pass**. No live ancestry/provenance, mapped-byte/
  output quality or hardware restoration claim. Latest: [clock telemetry contents](docs/results/2026-11-11-phase2-optimization/linux-clock-telemetry-validation.md)
  reuses unchanged CPU/GPU parsers for all 901 samples, binds raw receipts/readiness
  and brackets the restorer plus eighteen jobs. Six new tests cover 43 negatives;
  **21 focused/370 reporting tests pass**. Retained red/green readiness-time cases
  reject local time before sampling and a first GPU sample after readiness. Virtual-
  clock serialization/sparse GPU fixtures are not real collection or enforcement;
  gaps/unavailable values remain explicit. Latest: [clock report qualification](docs/results/2026-11-11-phase2-optimization/linux-clock-output-qualification.md)
  checks all six reports/runtime in fixed 1–3/4–6 halves with the unchanged codec
  checker and frozen Python reference. Current Python outputs compare directly to
  frozen Python, avoiding tolerance chaining. Three new tests include eighteen
  hash-verified historical report members/synthetic receipts and 43 altered cases;
  **7 focused/373 reporting tests pass**. Retained first failure was a no-op order
  mutation in the fixture, corrected without weakening guards. This is not fresh
  collection, capture integrity, producer authenticity or performance acceptance.
  Latest: [sequence evidence-file binding](docs/results/2026-11-11-phase2-optimization/linux-clock-evidence-binding.md)
  reads 74 private files through bounded no-symlink descriptor traversal, binds
  frozen jobs/deadline, embedded/individual records, capture hashes and report
  pathname identities, then rechecks visible metadata. Eight new tests include
  real eighteen-CPU-job transport and isolated archived outputs through the actual
  qualifier; **17 focused/381 reporting tests pass**. Capture inode provenance is
  explicitly unavailable (same-byte pre-read replacement allowed); no atomic/file-
  producer/whole-run proof. Latest: [control-side binding](docs/results/2026-11-11-phase2-optimization/linux-clock-evidence-binding.md#control-side-follow-up)
  adds 21 files (95 total in one bounded reader), six declared role receipts/capture
  digests, CPU report inode and GPU digest checks. Observer-supervisor stdout stays
  explicitly post-only; coordinator/observer-supervisor/GPU launch bindings remain
  owed by the frozen entry. Six new tests include predeclared real sibling fake-policy
  transport through the lifecycle validator; **25 focused/387 reporting tests pass**.
  No new producer authentication or live policy proof. Latest: [same-evidence finalizer](docs/results/2026-11-11-phase2-optimization/linux-clock-finalization.md)
  invokes one trusted restoration-aware runner, then gates those 95 files through
  full telemetry, sibling/restoration, process and eighteen-report qualification;
  receipt digests/clock domains agree, metadata rechecks precede frozen rehash last.
  Exclusive private completion marker appears only after all gates/output writes.
  Four new test methods use actual validators with archived outputs/synthetic clocks,
  maps and policy; 15 gate failures/two publication failures retain partial evidence.
  **18 focused/391 reporting tests pass**. Import-only composition is not a frozen
  entry or independent reaping/provenance/hardware proof. Latest: [current upstream snapshot](docs/results/2026-11-11-phase2-optimization/linux-clock-upstream-verification.md)
  at `f536960` verifies raw pinned commit `c702742`, replacement objects disabled,
  HEAD/index and 39 tracked files/877,235 bytes; zero untracked/ignored entries.
  Accepted/candidate Git guards pass, production/manifests unchanged; candidate's
  four allowed session entries stay private (counts only). Ten upstream probes and
  six native guards completed; 25-member compact archive consistency/privacy checked.
  **15 focused/391 reporting tests pass**. V1 retained before stronger v2 commit/object
  checks; no import-closure/historical-library/atomic-snapshot or launch claim.
  Next preparation audit found a [file-count admission blocker](docs/results/2026-11-11-phase2-optimization/linux-clock-preparation-admission.md):
  current tool paths plus retained declarations form 303 unique planned files; even
  a 261-file subset exceeds the clock postflight's 256-file cap. This is path-count
  evidence, not fresh target hashes or complete inventory. **9 existing file tests
  pass**; default suite not rerun (prior 391). Requested approval to raise only the
  clock postflight count to 512, preserving 8 GiB total/2 GiB per-file and every
  collection/deadline/telemetry/acceptance limit. No change applied; do not omit files
  or split batches to evade the cap. Need that decision AND the unanswered default-
  clock/no-custom-lock/no-controller answer; continuation establishes neither.
  Safe offline progress while those answers remain pending: [early fail-closed admission](docs/results/2026-11-11-phase2-optimization/linux-clock-preparation-admission.md#early-fail-closed-admission-follow-up)
  now shares pure declaration checks between finalizer and postflight. A retained red
  test showed 257 files reached the callback; now oversized/conflicting declarations
  reject before filesystem inspection, output creation or collection. No file IO/
  hashes added to this early check; actual file rehash remains last. Four new tests
  cover exact count/byte/alias boundaries, no-IO admission and seven finalizer faults;
  **17 focused/395 reporting tests pass**. No limit change: the 303-file plan still
  fails the 256-file cap. Fresh complete preparation/entry and live readiness gates
  remain pending. No quiet window requested, GPU/benchmark work or acceptance closure;
  Linux 72/78/Apple pending.

- Public repository: https://github.com/lexlapax/ncm-rs, `main`, remote `origin`.
- **Phase 2 is reopened for the additional five-step optimization plan approved
  on 2026-11-11.** The user requested the revised plan be committed/pushed before
  implementation. Phase 3 remains unstarted and blocked on this additional work.
  The earlier remediation is complete: production candidate `083c1e5`, evidence:
  `docs/results/2026-11-10-phase2-acceptance/README.md`. Both platforms pass all
  78 comparisons against previous Rust and qualified Python under unchanged
  limits. Final fresh CUDA, then Metal captures pass 642 traces/2,500 decisions
  each, zero ID disagreements; full real-device checks and tokenizer probes pass.
  CUDA codec latency and Apple checked-load/host-memory deficits are remediated.
  Original/intermediate failures remain historical evidence, not overwritten.
  User authorized implementation, research and commit/push for this extension.
  Phase 3 is not started and needs the next work authorization; no generator/CLI
  work was done. The active thread objective now includes completing the expanded
  Phase 2; planning publication alone does not complete it.
  Large licensed release archives remain separately pending authorization; this
  is a scoped codec/tokenizer result, not full-model inference or a release.
- [Approved optimization plan/research](docs/PHASE2-OPTIMIZATION-REVIEW.md):
  (1) freeze/refresh accepted baseline, (2) independent lower-risk experiments,
  (3) Linux gate then Apple gate, (4) latency/CPU/memory/startup/teardown evidence,
  (5) conditional profile-backed escalation and final completion audit. Step 4
  applies throughout. Source hashes/static estimates are under
  `docs/results/2026-11-11-phase2-optimization-review/`; new runtime evidence is
  pending. Baseline is `3307dff` (production `083c1e5`) plus qualified Python;
  unchanged 1.05 median/memory and 1.10 p95 limits apply against both comparators.
  Candidate ledger must cover prepared state, reconstruction reuse, MLX resampling,
  tokenizer APIs/structures, loading copies and conditional higher-cost ideas.
  Preserve historical acceptance/negative evidence and all existing contracts.
  Revised `docs/PLAN.md`, `docs/PHASE2.md`, review and status pages distinguish
  accepted earlier work from pending new gates. No production edits or new GPU
  runs in this planning revision. Default `CARGO_BUILD_JOBS=4 ./tools/check.sh`
  passes (24 report-tool tests); research JSON/hashes/arithmetic and local links
  checked. Plan published as `013f745`. Ask on consequential ambiguity/blockers;
  do not mark the active objective complete. Preserve `.pi/`.
- Additional optimization baseline checkpoint:
  [report](docs/results/2026-11-11-phase2-optimization/README.md), locked
  `acceptance-v1.json` and `baseline.json` pin current accepted Rust/Python before
  production edits. New clean worktree `.reference/phase2-optimization-baseline`
  at `3307dff`; saved binaries in adjacent `-bin` exactly match accepted benchmark
  hashes. Three-process rotating Linux matrix collected, CPU 2, with telemetry;
  all output/point/memory checks pass but diagnostic audit is **73/78**, not an
  acceptance pass (five GPT-2/Qwen process envelopes). Identical binaries show
  ~9% B1 reconstruction placement/process difference; no speedup is claimed.
  Fresh `tools/check.sh --cuda` passes (29 report-tool tests), trained capture
  passes 642 traces/2,500 decisions and cleared-environment tokenizer probes pass.
  Nine-process lifecycle and fresh CUPTI (1,312 kernels) retained. New tested
  `tools/bench/run_process_resources.py` records whole-child CPU diagnostics,
  not request CPU/allocation counts; v1/v2 preserve the inherited pre-exec RSS
  counter finding and explicit v2 labeling/streamed launcher hashing. No global
  settings or unrelated processes changed. No production edits or Apple run yet.
  Clean-checkout baseline summary/audit reproduction and archive/hash checks pass.
  Published baseline checkpoint: `cc965d8`. Its five uncertainty cells remain
  historical evidence, not erased by subsequent runs.
- Provisional lower-risk Linux candidates:
  [report](docs/results/2026-11-11-phase2-low-risk/README.md), source/archives in
  `evidence.json`; complete opportunity ledger at
  `docs/results/2026-11-11-phase2-optimization/experiments.json`.
  Diagnostic-only System wrapper/example counts Rust allocation requests (not
  native allocations or RSS); two explicit example tests pass. HF `encode_fast`
  removes 1,344 BERT allocation requests/89 strings, no GPT-2/Qwen reduction.
  TinyStories in-place all-occurrence merge compaction removes 259 requests/trial;
  exhaustive overlap/rank regression covers 2,047 binary strings plus long cases.
  Neither tokenizer change has an established material latency gain yet.
  Untraced reconstruction reuses encode's latent/immutable snapshot storage via a
  shared decoder tail; traced reconstruction retains independent decode and all
  observations. Extended tiny/trained tests verify untraced snapshots after another
  decode; trained logits/IDs/snapshots match traced values exactly. Fresh CUDA
  checks and 642 traces/2,500 decisions pass; tokenizer goldens/rejections pass.
  Matched B1/B2/B4 medians 3.0003/4.0104/5.1257 -> 2.2853/2.6712/3.3667 ms;
  diagnostic kernels 1,312 -> 918. Fresh lifecycle retained. Matrix is **74/78**:
  unresolved process envelopes at B1 resident block, Qwen decode, BERT/GPT-2 load;
  all pooled points/memory pass. No waiver or final Linux gate; no Apple run yet.
  Clean-checkout summaries/negative audit and evidence hashes verified. Default
  report tests remain 29; 17 tokenizer unit tests and 2 allocation-example tests
  pass. Next: prepared state/loading-copy investigations, matched uncertainty
  controls and full Linux gate before Mac availability recheck and MLX work.
- Prepared-state checkpoint (still provisional):
  [report](docs/results/2026-11-11-phase2-prepared-state/README.md) preserves two
  incremental matrices versus `bcf2be4`, not substitutes for final accepted-Rust
  comparisons. Shared positions/codebook norms and CUDA bounded geometry pass
  fresh 642-trace/2,500-decision qualification; final `tools/check.sh --cuda`
  passes. Three cache unit tests and a dedicated real-CUDA owner/drop/isolation
  test pass; live used-pool memory stabilizes and returns to baseline. Cache is
  backend-owned, clones start empty, logical bounds 64 entries/1 MiB. Profile
  kernels 918 -> 892, then geometry removes 70 H2D copies/24,912 bytes. Latest
  B1 block median ratio **1.1106 versus bcf2be4** remains unresolved; no final
  keep/acceptance claim. Both matrices/profiles/failed attempts are archived;
  clean-source summary/profile reproduction and hashes verified. Earlier 74/78
  audit remains historical, not a gate for this newer candidate. Raw-upload PoC
  tests pass; 58 tensors read back exactly, isolated medians 13.512509/7.379914 ms
  decoded/raw. **Public WeightBackend little-endian byte-upload method needs user
  approval; production loading remains unchanged.** No RSS/full-load claim.
  Next: API decision, hot-name/trace attribution, final resource/lifecycle evidence,
  controlled timing investigation and full Linux gate before Apple. No new Apple
  work, Phase 3 or completion claim. Snapshot binaries at
  `.reference/phase2-geometry-snapshot-bin`; original intermediate positions/norms
  executable hashes are recorded but those binaries are not retained. Preserve
  `.reference/phase2-low-risk-baseline{,-bin}` and all older baselines.
- Linux controlled follow-up remains **incomplete, 72/78**:
  [report](docs/results/2026-11-11-phase2-linux-controls/README.md),
  [working completion audit](docs/results/2026-11-11-phase2-optimization/completion-audit.md).
  Fresh full matrix is 74/78; predeclared 30-process controls for codec/BERT/GPT-2/
  Qwen retain all 360 processes. Six native-comparator cells remain open (B1 block
  median uncertainty; three GPT-2 execution p95 cells; codec/Qwen load p95).
  No noise waiver; all Python/matrix-memory cells pass, not an overall pass.
  Audit adds explicit full-codec control opt-in and binary/Python environment
  binding without changing limits; 36 report tests pass. Fresh CUDA checks,
  642 traces/2,500 decisions, tokenizer probes, nine-process lifecycle and
  45-process CPU/resource output validation pass. Six whole-model probes verify
  60 drops to zero live pool usage, 25,680-byte stable prepared state, and net
  retained-output savings ~1.07/2.16/4.35 MB B1/B2/B4 (not a host-RSS gain).
  Three unchanged-binary GDB captures establish 168 direct model format calls /
  7,992 requested bytes per reconstruction; trace/typed-name plumbing deferred.
  CUDA kernel/layout/graph/table/ownership decisions and tokenizer deferrals are
  recorded in the ledger; MLX-specific work and final dual-platform gates remain.
  A shared-target diagnostic baseline build caused stale release artifact reuse;
  a cache test failed and binary hashes identified old accepted binaries. Local
  CUDA/model release outputs were cleaned/rebuilt; fresh checks pass and frozen
  candidate hashes match again. Earlier measurements predate this incident and
  are binary-bound. **Use isolated targets for future cross-worktree builds.**
  Preserve `.reference/phase2-memory-baseline` (diagnostic example added locally)
  and `.reference/phase2-memory-bin`; no historical baseline overwritten.
  Clean staged-checkout reproduction matches both summaries and all three negative
  audits exactly; source/archive hashes, archive scope and local links pass.
  Offline inspection of every affected native process finds GPT-2/Qwen/codec-load
  slowdowns in both implementations around 05:47–05:49 UTC, plus a later Qwen spike.
  This is temporal association, not established causality or grounds to exclude
  samples. GPU locking is not established as necessary/sufficient for CPU-only
  failures; prioritize a quiet window and read-only CPU scheduling/frequency
  telemetry. The report records ranges/timestamps; all native ranges/peak-repeat
  identities are verified against recorded report hashes, and 36 report-tool
  tests pass. No new runs or policy changes.
  Do not change clocks/power/services or stop unrelated work without approval.
  No Apple run yet.
- **Stopped for handoff at the user's request; expanded Phase 2 is not complete.**
  Read [continuation notes](docs/results/2026-11-11-phase2-optimization/HANDOFF.md)
  before resuming. Latest addition is the read-only Linux CPU sampler and six
  tests (42 report tests total). Ambient observation identifies CPU 2/3 as SMT
  siblings; CPU-3 auditor overlap confounded the earlier resource-only run, not
  acceptance timings. One predeclared resource refresh used the same 45 jobs/
  binaries/inputs with observer CPU 0 and no concurrent analysis/build/debugger
  work. All outputs qualify, and 301 telemetry samples cover the entire run.
  New raw archive/report is linked from the Linux controls report; v1 is retained.
  No policy, production, public loader API or acceptance-gate change. Linux stays
  72/78; MLX remains blocked. No background collection is left running. `.pi/`
  remains intentionally untracked. Next: quiet/read-only-instrumented Linux
  investigation, then remaining completion-audit requirements; do not mark complete.
- Continuation: offline diagnostic preparation resumed after the handoff; no new
  acceptance, resource, GPU or Apple measurements. Rechecked HEAD/origin at
  `c4bcb69`, only `.pi/` untracked before edits; all 64 directly bound inputs in
  the retained final Linux audit match their hashes, still **72/78**.
  [Fixed diagnostic v2 protocol](docs/results/2026-11-11-phase2-optimization/linux-diagnostic-protocol-v2.md)
  proposes one 54-process run with read-only telemetry, **not yet run** and not an
  acceptance replacement. Quiet-window confirmation remains pending. Runner adds
  opt-in `--observer-cpu`: reject workload SMT siblings, pin launcher/GPU observers,
  retain explicit workload CPU, restore launcher affinity. Prior unpinned observers
  are a possible confound, not an established cause. Six tests include real Linux
  child inheritance/override; all **48 report tests** and default
  `CARGO_BUILD_JOBS=4 ./tools/check.sh` pass. Exact local log/hash in the protocol.
  No production/API/pin/tolerance changes. Next: confirm quiet window, bind/check
  all preflight identities, execute the fixed diagnostic once, then decide a
  defensible next intervention; Linux gate still blocks MLX and completion.
- **Diagnostic v2 completed once after explicit quiet-window confirmation**:
  [results](docs/results/2026-11-11-phase2-optimization/linux-diagnostic-v2-results.md).
  Launch HEAD/origin `fa040ed`; exact binary/source/Python/payload identities rechecked
  and manifest saved before launch. All 54 processes qualify; 901 samples/900 seconds
  bracket every command (17:01:56–17:06:47 UTC on 2026-11-11). All scopes/processes,
  telemetry and frozen pre-run protocol retained. Five CPU/load-tail issues did not
  recur, but native B1 block clusters persist: median ratio 1.10076, interval upper
  1.12231. CPU3 median/max activity 0/0.9901%; one-second GPU clocks cannot identify
  block clocks or establish cause. Outcome **unresolved recurrence**; Linux remains
  **72/78**, no control transplant, repeat, policy/production change or Apple work.
  Analysis/identities/coverage and clean-checkout summary reproduction verified;
  all 48 reporting tests and `CARGO_BUILD_JOBS=4 ./tools/check.sh` pass.
  No new full CUDA/Metal check is claimed. Evidence
  archive excludes binaries/weights/large traces; `.pi/` remains untouched.
  Next: separately predeclare a discriminating investigation. A bounded fixed-GPU-
  clock experiment is a hypothesis requiring explicit temporary-policy approval,
  capability/original-state checks, restoration and a new quiet window; not yet
  approved/performed, not proven necessary/sufficient. Read-only finer-grained
  observation is an alternative. No background collection remains running.
- Offline follow-up proposal (not approved/run):
  [clock diagnostic v3](docs/results/2026-11-11-phase2-optimization/linux-clock-diagnostic-v3-proposal.md)
  specifies one 18-process codec-only investigation, optional 1710 MHz request,
  reset-to-default policy caveat, 600-second cleanup deadline and 900-second CPU
  capture. Read-only capability query lists 1710 MHz; lock support/permission and
  restoration are NOT proven. Supervisor implementation/failure-path tests remain
  required before launch; no executable lock wrapper exists yet. User must confirm
  default policy, approve the temporary change and provide a new quiet window.
  Docs-only links/diff checks and 48 reporting tests pass; no runtime/policy change.
  The completed v2 is not rerun; Linux remains 72/78 and Phase 2 incomplete.
- Offline resident-block source inspection at `41ab9e5` is recorded in the
  [completion audit](docs/results/2026-11-11-phase2-optimization/completion-audit.md#offline-resident-block-timing-inspection).
  Benchmark source/CUDA ops and inspected block/evaluate sections match accepted
  `3307dff`. The block synchronizes before end-event enqueue: event spans include
  submission/idle gaps, not just kernel duration. Matching wall/event clusters
  therefore do not establish a device-clock cause. Verified with direct source
  inspection and byte/section comparisons; docs diff/link checks and all 48
  reporting tests pass. No new benchmark, hardware test, policy change or waiver.
  Next read-only protocol must discriminate kernel duration from submission gaps
  and label instrumentation overhead; investigation choice/new quiet window remain
  pending. Linux 72/78.
- [Offline v2 pool/trace inspection](docs/results/2026-11-11-phase2-optimization/linux-diagnostic-v2-results.md#offline-pool-counter-and-retained-trace-follow-up)
  checks all twelve archived native reports: B1 pool counters are constant within
  each runtime across fast/slow repetitions (64 MiB reserved, 128 MiB threshold).
  This does not exclude allocation latency or identify a cause. Older CUPTI
  captures are reconstruction/B4 scopes, not v2 B1 trial traces; no retrospective
  attribution. Published standard-library reproduction verifies archive/member
  hashes and all rows; ran successfully. Docs diff/new-link checks pass. No new
  measurements, settings changes or acceptance waiver; Linux remains 72/78.
  Next: predeclare a bounded trace investigation if the read-only path is chosen;
  actual execution still needs a fresh quiet window.
- [Read-only trace proposal](docs/results/2026-11-11-phase2-optimization/linux-readonly-trace-v1-proposal.md)
  now specifies 12 fixed native plain/traced processes, frozen v2 binaries,
  time/output caps and strict B1 window attribution. Source-derived queries
  110–145 cover B1 block first/warm/retained calls before stage profiling; no
  full-harness trace confirms that mapping yet. Offline checks on three old B4
  captures verify 36 two-event/query windows each, not current B1 causality.
  Docs links/diff and schedule arithmetic/source checks pass. No new run/policy
  change; parser/runner and required tests are not implemented. Next: user choice,
  implement/test the selected diagnostic, then obtain a fresh quiet window and
  bind launch prerequisites. Linux remains 72/78; no acceptance or Apple progress.
- Offline prerequisites under `tools/bench/`: `cupti_trace_boundaries.py`,
  `cupti_kernel_windows.py`, `bind_cupti_trials.py`, `qualify_diagnostic_codec.py`,
  `bounded_child.py`; 64 tests covering host boundaries, kernel/copy/memset correlation,
  overlap arithmetic, source/hash/PID declarations, ordinal joins and report-output
  qualification through the unchanged codec validator/frozen v2 Python references. Synthetic full-layout coverage
  is not a real current B1 trace. Each old B4 capture maps 972 kernels, 396 copies
  and 72 memsets to 36 windows; retained-B4 first/30-trial timing joins match
  exactly. Warmup timings were not serialized and stay null. Byte declarations do not prove same-child
  launch provenance or output quality. Bookkeeping/overhead stays unattributed;
  actual launch/report provenance, completeness and integrated collection remain
  pending. The bounded child primitive passes fake CPU success/failure/limit/
  interrupt/cleanup tests, keeps the leader waitable until group cleanup, and
  records failed/unconfirmed outcomes. Two red-to-green tests exposed setup-time
  cancellation/deadline gaps, now checked before attempting a spawn. Fake child
  environments are minimal; the original environment-bearing failure log is
  restricted/unpublished. It is not a full launcher; aggregate quotas,
  frozen manifest/order, telemetry/affinity and supervision across collector loss
  remain unresolved. No benchmark or policy change. The output qualifier passes archived regression fixtures,
  not a new capture, oracle run or acceptance attempt. User choice/quiet window still block launch; no new measurements/settings. `CARGO_BUILD_JOBS=4 ./tools/check.sh`
  passes with 112 reporting tests. Supported memory activities are matched in both
  directions; unknown/ambiguous/cross-window records fail, overlap is not counted
  twice, and equal-timestamp host records sort deterministically. Initial
  boundary-suite check failed inherited-RSS isolation; archive work now runs in
  a child, preserving the existing resource assertion. Kernel prototype initially
  missed RUNTIME launches; retained fixtures exposed both variants, now tested.
  Checks/log hashes and limitations are in the read-only proposal; docs diff/link
  checks pass. Linux remains 72/78; Apple and completion remain pending.
- Within EVERY phase: finish Linux/CUDA work and validation, then macOS/MLX.
  Both gates before advancing. Phase 3 generator/CLI work is out of this goal.
- Phase 1 is complete on both platforms ([acceptance](docs/PHASE1.md)). Phase 2
  implements shared FP32 Sudoku codec semantics and TinyStories/GPT-2/Qwen/BERT
  text tokenizers, **not full text-model inference**. No generator, inference CLI,
  official task-quality result, generalized importer or inference release exists.
- Production execution stays Python-free. Offline conversion/reference/measurement
  and native CUDA/MLX libraries are expected. Never silently change pins, schemas,
  established goldens or numerical tolerances; unsupported options must fail.
- Local checks replace GitHub build CI: `tools/check.sh`, opt-in `--cuda`/`--mlx`.
  Never restore build CI without approval. Metadata-only upstream watcher remains.
- `ssh macstudio` is authorized. Recheck availability/competing work before later
  runs, inspect worktree, then pull reviewed pushed commits into
  `~/projects/lexlapax/ncm-rs`. Never overwrite unrelated work or global tooling.
  Announce the first Phase 2 MLX run only after Linux acceptance.
- Update this map before EVERY commit, including amend/fixup. No agent attribution
  or co-author trailers; preserve upstream licenses and technical provenance.
- Persistent goal state is untracked `.pi/`; leave it untouched and unstaged.

## Historical Apple measurement acceptance

- `docs/PLAN.md` and `docs/PHASE2.md` now distinguish the accepted original
  baseline from the reopened performance-remediation extension.
- Independent Python MLX baseline passes all 642 unchanged trained traces and
  2,500 decisions against the CUDA oracle. Three-process codec/tokenizer evidence,
  raw telemetry and baseline qualification are published in
  `docs/results/2026-11-10-phase2-mlx/`. The extension fixes the summary's hidden
  dependency on ignored baseline files using checksum-pinned committed evidence.
  A clean-checkout test reproduces all historical metrics exactly; only current
  summarizer source hashes differ. Historical reports are not rewritten.
- Host-only artifact loads reproduce high native RSS; diagnostics remain at
  `reports/phase2-mlx-load-diagnostic`. No loader fix or efficiency claim.
- Final summary-tool tests pass locally. On the M1 Ultra candidate worktree,
  `./tools/check.sh --mlx` passed on 2026-11-10 with the pinned MLX prefix,
  including real-Metal components/operators and 11 report-tool tests; the exact
  measured source hashes were verified equal to the candidate files before this
  check. MLX reference asset tests also pass in `.reference/mlx-venv`.
- Preserve `.pi/` and the Mac candidate worktree's unrelated/uncommitted state.

## Task routing

| Need | Load |
| --- | --- |
| Phase 2 contract / acceptance / remediation | [final dual-platform acceptance](docs/results/2026-11-10-phase2-acceptance/README.md), [docs/PHASE2.md](docs/PHASE2.md), [plan](docs/PLAN.md), [research and performance extension](docs/PHASE2-PERFORMANCE.md), [initial research/PoC](docs/results/2026-11-10-phase2-performance-research/README.md) |
| Approved additional Phase 2 optimization | [five-step plan, research and pending evidence checklist](docs/PHASE2-OPTIMIZATION-REVIEW.md) (implementation authorized; no new speedup result) |
| Shared runtime and model | [core](crates/ncm-core/src/lib.rs), [config](crates/ncm-core/src/config.rs), [model](crates/ncm-model/src/lib.rs), [blocks](crates/ncm-model/src/blocks.rs), [codec](crates/ncm-model/src/codec.rs) |
| CUDA adapter / known native fixes | [adapter](crates/ncm-backend-cuda/src/codec.rs), [vectorized resampling](crates/ncm-backend-cuda/src/resample.rs), [pool policy](crates/ncm-backend-cuda/src/runtime.rs), [operators](crates/ncm-backend-cuda/src/ops.rs), [remediation](docs/results/2026-11-10-phase2-cuda-remediation/README.md) |
| Phase 2 oracle and complete CUDA results | [oracle audit](docs/results/2026-11-10-phase2-oracle/README.md), [codec report](docs/results/2026-11-10-phase2-codec/README.md) |
| Tokenizer identities, safety and parity | [tokenizer report](docs/results/2026-11-10-phase2-tokenizers/README.md), [source lock](tools/reference/tokenizer-sources.lock.json), [canonical lock](tools/reference/tokenizers-v1.lock.json) |
| Fixtures / offline tools | [fixtures](tests/fixtures/README.md), [reference README](tools/reference/README.md) |
| Artifact trust/limits | [docs/ARTIFACTS.md](docs/ARTIFACTS.md), [ncm-io](crates/ncm-io/src/lib.rs), [runtime checks](docs/results/2026-11-10-phase2-runtime/README.md) |
| Architecture / upstream caveats | [architecture](docs/ARCHITECTURE.md), [research](docs/RESEARCH.md) |
| Checks / reporting / compatibility | [testing](docs/TESTING.md), [reporting](docs/REPORTING.md), [upstream policy](docs/UPSTREAM.md) |
| MLX codec / current parity evidence | [adapter](crates/ncm-backend-mlx/src/codec.rs), [Apple parity report](docs/results/2026-11-10-phase2-mlx-codec/README.md), [shared trained capture](tests/support/trained_codec.rs) |
| Phase 2 Linux acceptance / benchmarks | [Linux gate](docs/results/2026-11-10-phase2-linux/README.md), [summary tool](tools/bench/phase2_summary.py), [runner](tools/bench/run_phase2_linux.py) |
| Historical baselines | [results index](docs/results/README.md), [Apple Phase 1](docs/results/2026-11-10-mlx-phase1/README.md) |

## Pins, machines and caches

- Upstream NCM: `c70274273f489abc73832b2b98584a8483b6b203`.
- HF checkpoints: `hazyresearch/ncm-checkpoints` at
  `d342ce8aaea58830eef3b8ad473664673e41f692`. OWT weights remain absent.
- Candle `=0.11.1`; default workspace needs no GPU SDK. Explicit CPU diagnostic
  and CUDA paths never silently fall back. Rust floor **1.99.1** is tested;
  previous 1.88 failed Candle AVX-512 intrinsics, not all intermediate versions.
- Linux: RTX 3060 Ti 8 GiB, driver 615.78.08, CUDA 13.4. Command:
  `NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 ./tools/check.sh --cuda`.
  Default GCC 16 is not the tested CUDA host compiler.
- Linux Torch reference: `.reference/venv`, Python 3.14.9, Torch 2.15.0+cu132;
  `tools/reference/requirements-linux.lock`. Clean source: `.reference/upstream-ncm`.
- Sudoku source: `artifacts/upstream/sudoku/easy`; canonical inspected bundle on
  both machines: `artifacts/canonical/sudoku-easy-qualified`. Its name alone does
  not confer inference qualification. Original file/tensor hashes are committed
  under `docs/results/2026-11-10-sudoku-artifacts/raw/`.
- Tokenizer source: `artifacts/upstream/tokenizers-v2`; canonical:
  `artifacts/canonical/tokenizers-qualified`. Separate `.reference/tokenizers-venv`
  uses `requirements-tokenizers.lock` (Transformers 4.57.6, Tokenizers 0.22.2,
  regex 2025.9.18 / Unicode 16). Do not replace the Torch environment with it.
- Rust HF Tokenizers `=0.22.2`, only Oniguruma feature; no HTTP/remote code. Candle's
  pre-existing Tokenizers 0.23.2 remains separate. Tiny byte BPE uses locked
  fancy-regex 0.17.0 and Unicode-16 regex-syntax tables. Asset licenses are retained.
- Mac: M1 Ultra 64 GiB, macOS 26.7.3, Xcode 27.2, Rust 1.99.1. Noninteractive SSH:
  `export PATH=/opt/homebrew/bin:$PATH`. Optional Xcode Metal compiler is missing;
  do not change global Xcode. Tested project-local prebuilt native MLX route works.
- MLX bindings: mlx-rs 0.34.0 / mlx-sys 0.7.0 Git
  `adf439bf5361d8a02955de7aaf651b0c45900945`; native MLX 0.32.2, mlx-c
  `a341b4925024b88b2c593468f16e12f5e17315da`. `.reference/mlx-venv` is offline-only;
  native include/share/lib are copied into `.reference/mlx-native`.
- Mac candidate testing used a separate worktree `.reference/phase2-mlx-candidate`
  with the main checkout's native prefix and target cache. Main checkout is updated
  only by pulling pushed commits. Preserve unrelated work; existing CPU work was
  left running, builds limited to two jobs. Benchmark acceptance must assess noise.
- Mac build: `MLX_SYS_PREBUILT="$PWD/.reference/mlx-native"`,
  `RUSTFLAGS="-C link-arg=-Wl,-rpath,$MLX_SYS_PREBUILT/lib"`,
  `RUSTDOCFLAGS="$RUSTFLAGS"`, `CARGO_BUILD_JOBS=8`, then `./tools/check.sh --mlx`.
  Tested wheel needs macOS 26.2+; absolute rpath is a spike setup, not packaging.

## Evidence and boundaries

- Phase 1: all 18 fixed operator cases pass CPU/CUDA/MLX at unchanged
  `atol=1e-5, rtol=1e-4`; all 238 checkpoint tensors read back byte-exact.
  Metadata-only manifest integrity is NOT authenticity or semantic validation.
  Root/leaf symlinks are rejected, but trusted parents/no concurrent hostile
  mutation remain required. Input byte limits are NOT peak-RSS bounds.
- Historical Linux checked-load median Rust/Python ~202/241 ms. Apple ~939/282 ms
  with repeated-load host peak RSS ~4.40/0.41 GiB: Rust slower/higher RSS. Native MLX
  active arrays return to zero, cache retains ~155 MiB; counters do not explain
  total host RSS. Investigate in Phase 2 before any efficiency claim.
- Phase 1 composed resampling was slower than PyTorch. Compiled Torch FP16 loses
  rounding on one fixture: compiled mixed precision remains unsupported, no waiver.
- Phase 2 reference budgets were declared before implementation in `docs/PHASE2.md`.
  Canonical trained oracle is CUDA; tiny oracle is CPU. Original-Python trained
  CPU/CUDA diagnostic retains one failing blank-board level-zero projection value
  (22,784-term dot, channel 113, ~0.001054 difference). Never use this to exempt
  candidate GPU failures or silently relax tolerances.
- Config/loading: strict initial codec family, exact 58 F32 runtime tensors,
  checked IDs/options, full host preflight before device upload; training buffers
  are separate. Both tiny and trained inventories match. No unsafe Send/Sync.
- Codec `n_layers=4` means head count, not decoder depth. Decoder is bidirectional,
  Q/K RMS-normalized and exact-erf GELU. GroupNorm spans positions/grouped channels.
  Preserve exact area bins, half-pixel interpolation and independent snapshots.
- Candle fixes: CPU batched transposed convolution misindexes a multi-position
  second batch, so only CPU dispatches separate batch slices. CUDA remains native
  batched. Native CUDA argmin tie order was wrong; selection now reduces values,
  then masked indices on-device. Tests cover wide/signed-zero ties and both fixes.
- Complete tiny codec: all 236 traces pass CPU/CUDA; snapshot/uninstrumented checks
  yield 252 floating comparisons / 112,212 elements; all IDs exact.
- Trained CUDA codec: all 642 traces and 2,500 code/output decisions pass. Two
  mixed-board near ties (levels 1/2) remain visible. Worst normalized error uses
  **94.2%** of the budget at blank-board level-zero channel 113. Do not infer Apple
  parity from Linux's narrow margin. Two fresh native traces are byte-identical.
- Reconstruction matches Python on 445/445 tokens and 5/5 sequences; recovery of
  original inputs is 437/445 (eight mixed-board differences), not losslessness or
  Sudoku-solving quality. Uninstrumented outputs match traced outputs.
- Historical native trace: `reports/phase2-codec/cuda-v2`, SHA-256
  `a3776cac0ba75eaeb9ebc9a30ebd03c131f0b4470f12f136b7ccc95fc25e0fa0`.
  Python reference traces: `reports/phase2-oracle/v1/`. Durable Phase 2 archive
  publication remains pending. Phase 1 archive exists at research prerelease
  `phase1-fixtures-v1`, not an inference release.
- Tokenizers: all four pinned profiles pass 356 encode and 2,314 decode comparisons,
  including bytes/Unicode/specials/postprocessing; Sudoku goldens unchanged.
  Tiny pickle conversion interprets bounded data opcodes, never unpickles or calls
  constructors. Canonical HF JSON comes from offline trusted-code-disabled wrappers.
  Four real-asset mutations/profile are rejected; fresh conversions are identical.
- Historical Linux measurements: `reports/phase2-linux-v1`; all 30 fresh-process raw reports,
  telemetry and first logits are committed under `docs/results/2026-11-10-phase2-linux`.
  Three processes, five warmups, thirty trials; matching output hashes/counts and
  44,856 native benchmark logits pass. B1 reconstruct Rust/Python 18.890/4.668 ms;
  warm codec load 184.49/267.61 ms; host HWM ~559–562/1530–1540 MiB. Native pool
  counters exclude context/cuBLAS allocations; not interchangeable with Torch's
  allocator or device-global snapshots. Occupied workstation/clock variation disclosed.
- Historical instrumented level spans identify intermediate composed resampling as the main
  native cost. Original-Python fullgraph Inductor preflight fails on NumPy-based
  Phi module indexing; no compiled timing baseline is qualified. Tokenizer timings
  retain both skip modes and special/non-special throughput; Qwen native loading
  is slightly slower in pooled medians. Original-source vs canonical load formats
  differ. No model/operator/tokenizer math, dependency pin or golden changed.
- MLX now implements the existing backend contract with opaque immutable arrays
  and an explicit GPU stream (restores a CPU-scoped caller); no model duplication.
  All 18 operators, 140 components and complete tiny codec pass; 642 trained traces
  and all 2,500 decisions pass. Two fresh captures are identical, SHA-256
  `f20ffe0f36a82c80a81e91a61b61950968821369b6b41d0f13cbf9e732875a0f`.
  Same two near ties; reconstruction 445/445 versus Python, input recovery 437/445.
- Initial MLX full-span convolution failed two blank-board projection values despite
  exact IDs. Retained in the Apple report. Equivalent native FP32 matmul for
  zero-padding/full-span convolution fixes parity; no precision/oracle/budget change.
  Worst component still uses 94.28% of budget. Large v1/v2 traces remain ignored
  under `reports/phase2-codec/` on both machines; Mac also has `mlx-v2-repeat`.
- The shared trained-capture harness refactor preserves the CUDA trace hash exactly.
  Mac tokenizer probe passes all four profiles, including rejection cases and a
  cleared environment without Python. Canonical tokenizer assets are now on Mac.
- Historical Linux `tools/check.sh --cuda` passes: 30 base Rust tests, real-CUDA codec
  suite, all 18 unchanged operators, fmt/Clippy/docs/syntax. Twelve report-tool tests
  and all 18 offline reference tests pass. Tokenizer probe previously also passed
  with cleared env and no Python on PATH. Logs: `.reference/phase2-linux-check.log`,
  `.reference/phase2-linux-reference-tests.log`; reports retain exact fingerprints.
  Shared-harness/final Linux checks: `.reference/phase2-mlx-shared-linux.log`,
  `.reference/phase2-mlx-final-linux.log`; all 18 offline tests also rerun.
  Mac `tools/check.sh --mlx` passes with four new codec tests; logs copied locally:
  `.reference/phase2-mlx-v2-check.log`, `.reference/phase2-mlx-v2-components.log`.

## Provenance and next actions

- Phase 1 closure `650cc7d`; Phase 2 oracle `97a3125`, checked runtime `8492c27`,
  components `a4a6344`, complete codec `bb77fef`, tokenizers `0396b56`, Linux runtime
  gate `fd1b482` (pushed). Preserve old reports/source hashes;
  do not relabel historical runs. Benchmark outputs are synthetic fixtures, not
  task-quality data or an inference release.
- **CUDA remediation milestone:** vectorized FP32 gathers/area reduction, pinned
  optional `candle-nn = 0.11.1` fused CUDA norms/softmax, immutable views and explicit
  default-pool retention. No shared model/loader/tokenizer math or goldens changed.
  CUDA default release threshold is now 128 MiB; explicit per-process shared-pool
  API allows zero/custom policy. Not preallocation or a hard cache limit. Tests
  cover real-CUDA policy/opt-out, broad resampling, stable ties and snapshots.
- User authorized stopping `llama-server.service` (inactive, not disabled), closed
  Chrome, then authorized continuing with Hyprland running. No other app was stopped
  or global tooling/power changed. Local CUPTI sample traces identify launch count
  16,756 → 1,312 for load + two B1 reconstructions. B4 bimodality was allocation:
  fast/slow cuMemAllocAsync totals 0.636/20.497 ms with identical kernel work;
  retention gives 0.642 ms. Extra reusable reservation is disclosed (96 vs 64 MiB).
- Predeclared bands are in the research `acceptance-v1.json`, written before
  production changes. Final interleaved old/new/Python reports are archived with
  source/binary hashes, environment and telemetry in the remediation report.
  B1/B2/B4 reconstruction old/new/Python medians: 18.165/3.285/4.659,
  26.681/4.003/5.081, 28.949/5.134/5.859 ms. All codec median/p95 Python bands
  pass descriptively. Warm load 195.88/194.45/286.25 ms; native host HWM ~580 MiB
  remains favorable. Failed vectorized B4 and earlier comparison rounds retained.
- Correctness: final `reports/phase2-codec/cuda-remediated-v1` passes all 642
  trained traces/2,500 decisions with unchanged budgets; same two near ties and
  437/445 input recovery. Native trace SHA-256:
  `d180a52c011c8af15812d10941de8574b1671c2cd23de67e9abc828f7c5ecba4`.
  360 synthetic CUDA rows / 753,984 values pass against pinned Torch CUDA (maximum
  budget fraction 0.628). FP64-coordinate alternative still fails 228 values;
  this GPU check is distinct from the earlier Torch-CPU comparison.
- Validation: final `tools/check.sh --cuda` passed (log
  `.reference/phase2-cuda-remediation-final-check.log`, also published). Final
  `./tools/check.sh` passes, including 20 report tests; 21 offline reference tests
  pass. A clean staged checkout with no ignored inputs exactly reproduces all six
  matrix summaries (historical summarizer provenance excepted), six CUPTI profiles
  and the control summary; evidence/source fingerprints and doc links verified.
  Logs: `.reference/phase2-remediation-publication-check.log` and
  `.reference/phase2-remediation-reference-tests.log`. Staged code/archives reviewed;
  `.pi/`, executables, weights and large model-trace archives are not staged.
- **Extended Linux acceptance passes; next is Apple remediation.** Fixed 30-triplet
  TinyStories and nine-triplet GPT-2 affinity controls close execution uncertainty.
  Qwen's first nine-triplet control still failed uncertainty; a fixed 30-triplet
  identity/path control (only allowed for byte-identical native binaries) passes.
  All negative runs are retained; no tokenizer optimization or proven path cause
  is claimed. Whole-matrix audit passes 78 comparisons with unchanged bands and
  qualified traces. Fresh-process startup/load/execute/drop/exit evidence is now
  separate from steady-state timing. See the Linux acceptance report for metrics.
- Latest `./tools/check.sh --cuda` passes, including 22 report tests; log
  `.reference/phase2-linux-acceptance-check.log`, also published. Audit:
  `reports/phase2-linux-acceptance-v2.json`; controls are archived in the new report.
  A clean staged checkout exactly regenerates acceptance from committed archives
  and passes all 22 report tests; staged scope/diff checked before publication.
  Shared lifecycle harness compiled against both old/current CUDA and exact solved
  outputs agree across all nine processes. No production change in this follow-up.
- Mac availability/state rechecked after `7aee602`: main was clean and safely
  fast-forwarded; old binaries saved in `.reference/phase2-mlx-baseline-bin`, clean
  old source in `.reference/phase2-mlx-baseline` (`5dc9626`). Historical candidate
  worktree inspected and untouched. First remediation diagnostics announced/run.
  Fresh old load ~944 ms and host HWM 4.81 GB reproduce the deficits. Allocation
  stacks identify `bounded_read → RawVec::finish_grow → realloc`; VM maps show
  freed large malloc regions. CPU sample: 2,187/2,222 main-thread samples in SHA-256.
- Shared candidate: reserve checked read length + EOF byte; deserialize authenticated
  BPE/WordPiece tokenizer models directly into the same runtime wrapper; enable
  pinned sha2 0.10.9 `asm` only on macOS/AArch64 (explicit sha2-asm 0.6.4 addition).
  No checksums/preflight/path checks, payloads, precision or algorithms removed.
  Details/evidence: `docs/results/2026-11-10-phase2-shared-loading/README.md`.
- Linux revalidation precedes Apple candidate tests: full `tools/check.sh --cuda`
  passes (`.reference/phase2-shared-load-final-check.log`), all 642 traces/2,500
  decisions and real tokenizer goldens/rejections pass; 21 offline reference tests
  pass. Fresh three-way matrix and 30-triplet controls yield 78 passing comparisons
  (`reports/phase2-linux-shared-load-acceptance-v2.json`). Qwen load 194.80/107.75/
  172.34 ms old/new/Python. Linux codec/tokenizer benchmark binaries remain byte-
  identical after the Apple-only feature addition; Linux SHA features unchanged.
- Shared change pushed as `74c2ed3`, pulled on Mac; full `tools/check.sh --mlx`
  passes. First codec diagnostic: 180.06 ms / 293.63 MB HWM versus old 944.17 ms /
  4.81 GB and Python 270.58 ms / 309.10 MB. Qwen is faster (93.79 ms) but still
  634.52 MB HWM; allocation stack identifies its separate read-buffer reallocations.
- Follow-up reserves checked tokenizer file length + EOF byte, with boundary tests.
  Returned through Linux first: full CUDA checks, real tokenizer goldens/rejections,
  fresh whole matrix and 30-triplet controls pass all 78 comparisons. Negative
  audits retained. Explicit optional `load_controls` applies unchanged paired-process
  uncertainty to checked loads too; malformed-trial regression passes. Evidence
  and exact configs are in the shared-loading report's `raw/` directory.
- SSH restored by user; Mac availability/worktrees rechecked, clean main pulled
  through `2c195ed`. Full Metal checks pass. Tokenizer-buffer candidate improves
  Qwen to 86.97 ms / 455.25 MB HWM, but fresh Python is 145.97 ms / 288.59 MB:
  memory still fails. No unrelated desktop work, tooling or credentials changed.
- User requested reconsidering Rust structures for memory AND performance. New
  candidate packs TinyStories bytes/u32 offsets, stores merge output IDs without
  redundant ranks, borrows reverse lookups and moves valid UTF-8 output storage.
  HF loading avoids full-vocabulary clones for size validation and untagged merge
  trees, keeping the pinned runtime BPE and complete tokenizer visitor unchanged.
  First packed decoder regression retained; ownership fix passes controlled timing.
- Linux first: full CUDA checks, all tokenizer goldens/rejections, complete matrix,
  30-triplet TinyStories/BERT controls and nine full-codec load-control triplets
  pass all 78 comparisons. Qwen load 223.22/89.90/196.78 ms old/new/Python; HWM
  194.49/81.56/240.10 MB. New checked-load control validates every codec report;
  Apple controls explicitly have no affinity claim and require activity telemetry.
  New MLX lifecycle/three-way tools await real Apple execution. Full details and
  all failed/final evidence: `docs/results/2026-11-10-phase2-compact-tokenizers/`.
- Compact candidate pushed/pulled as `45743d0`; full Metal checks and real
  tokenizer goldens pass. Qwen improves to 65.82 ms / 326.83 MB HWM but still
  exceeds Python's memory band. Retained as failed memory diagnostic. Follow-up
  reserves canonical BPE vocab/merge containers from pinned profile capacities;
  tests enforce bounds. Full Linux CUDA checks and fresh matrix + 30-triplet
  controls again pass all 78 comparisons; unchanged-codec load control is explicitly
  reused with source binding. Evidence appended to the compact-tokenizers report.
- Capacity follow-up pushed/pulled as `083c1e5`. Full Metal checks and real
  tokenizer goldens pass; 642 trained traces/2,500 decisions qualify unchanged,
  zero ID disagreement. Fresh Apple matrix: codec load 945.87/179.99/272.21 ms
  old/new/Python, HWM 4749.16/293.91/308.69 MB; Qwen 189.29/63.65/144.78 ms,
  HWM 633.88/69.81/290.98 MB. Repeated-load probes plateau with zero active MLX
  bytes after drops. All latency point estimates and host-memory ratios pass;
  75/78 cells meet conservative process envelopes. GPT-2 encode and Qwen
  encode/decode versus old Rust still need fixed additional process controls.
- Separate `.reference/phase2-mlx-lifecycle-baseline` at `5dc9626` has ONLY copied
  lifecycle example/support files; baseline build and remote lifecycle command
  succeeded. **SSH agent became unavailable again before report transfer.** Retry
  denied; user asked to unlock. Mac lifecycle report/final check logs remain remote;
  do not infer lifecycle figures or final acceptance. Next use a process-local
  persistent SSH connection after access returns, without global configuration edits.
- Published qualified Apple measurements and explicit pending diagnostic:
  `docs/results/2026-11-10-phase2-mlx-remediation/README.md`. Audit now binds Python
  control sources as well as native ones; mutation regression and local checks pass,
  latest Linux gate still passes 78 comparisons under the stronger audit. A clean
  staged checkout without ignored artifacts exactly reproduces Linux acceptance
  and both Apple summaries; all 24 report-tool tests pass.
- Final follow-up: SSH restored; availability rechecked and a process-local
  persistent connection used (no global SSH edits). Mac main pulled `42caaec`.
  Retrieved/validated lifecycle and final logs. Fixed 30-triplet GPT-2/Qwen controls
  resolve all three uncertain cells: all 78 Apple comparisons pass. Linux also
  passes 78 under the final audit. All negative rounds remain archived.
- Final `tools/check.sh --cuda`, then `--mlx`, pass; fresh native trained traces
  captured CUDA first then Metal both pass 642/2,500 unchanged. Cleared-environment
  tokenizer probes pass on both. Evidence/configs/logs are in the final report.
  Apple lifecycle medians: old/new/Python full process 1050.72/287.30/468.01 ms;
  startup 39.84/39.68/106.10 ms. Not disk-cold or steady-state timing. Clean staged
  checkout exactly reproduces both 78-cell audits and all four matrix summaries;
  all 24 report tests and evidence hashes pass. Control binaries match matrices.
  All 21 offline reference-tool tests and the final default publication check pass.
  Plan, tracker, extension contract, root README, testing/architecture status and
  results index now consistently record completion without a generator/release claim.
- **Next: await the next task; Phase 3 is unstarted, not blocked by Phase 2.**
  Preserve historical worktrees, goldens and `.pi/`. No service, unrelated process,
  clock/power policy or global tooling was changed for the final measurements.
