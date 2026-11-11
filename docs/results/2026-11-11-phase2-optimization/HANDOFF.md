# Phase 2 continuation notes

Work stopped at the user's request after committing/pushing the current checkpoint.
**The active objective is incomplete; do not mark it complete.** Start with
`agent-context-map.md`, `git status`, then the [working completion audit](completion-audit.md).
No background benchmark or telemetry collection remains running.

Continuation completed the single [fixed diagnostic v2](linux-diagnostic-v2-results.md)
after the user confirmed the quiet window. All 54 processes, output qualification,
identities and coverage by 901 samples/900 seconds verify. Five CPU/load-tail
failures did not recur, but B1 block bimodality persists in both native runtimes;
its median ratio is 1.10076 (interval upper 1.12231). Outcome: **unresolved
recurrence**, not acceptance. All 48 reporting tests and clean-checkout analysis
reproduction pass. No production or system-policy change; Linux remains 72/78.
No background collection remains running. Do not repeat this diagnostic.

## Retained state

- Approved five-step plan: `docs/PHASE2-OPTIMIZATION-REVIEW.md`; earlier planning
  checkpoint `013f745`. Phase 3 remains unstarted and requires separate authorization.
- Accepted baseline: repository `3307dff`, production `083c1e5`, plus unchanged
  qualified Python. Current provisional production behavior is `d44beaa`; later
  checkpoints add evidence/reporting/diagnostics, not production optimizations.
- Prepared state, reconstruction snapshot/latent reuse and tokenizer allocation
  improvements are implemented. Linux correctness and full-model lifetime checks
  pass. Native output-retention savings are measured; no host-RSS saving or broad
  CPU improvement is claimed. Full experimental dispositions are in
  [experiments.json](experiments.json).
- **Linux acceptance: 72/78 after all 360 predeclared control processes.** Failed
  native-comparator cells: B1 resident-block median uncertainty; GPT-2 encode,
  decode and skip-special-decode p95; codec and Qwen checked-load p95. Do not waive,
  discard, transplant old controls or repeatedly sample until passing.
- Latest full CUDA check/qualification remains valid for the unchanged production
  candidate: 642 traces/2,500 decisions; cleared-environment tokenizer probes pass.
  The new read-only telemetry tooling brings the separately run report suite to
  42 tests. No new Apple run has started; Linux gate must close first.

## Latest investigation and useful next action

The native timing distributions show both GPU block timing clusters and CPU-only
slowdown windows. The cause is not established. The user has now approved
**read-only option 1 first, with clock-controlled option 2 only if its data are
insufficient**. Neither has run. Finish/test integrated collection before asking
for a fresh quiet 15-minute window; the v2 window is spent. One-second GPU samples
cannot identify per-block clocks, and clock locking is not established as
necessary/sufficient. The [v3 proposal](linux-clock-diagnostic-v3-proposal.md)
remains conditional on documented insufficiency of option 1, confirmed default
policy/no competing controller, supported-clock/permission checks, a tested
restoration supervisor and a fresh window. Reset-to-default is not restoration
of an unknown custom policy. No unrelated policy/process changes are authorized.

The alternative [read-only trace proposal](linux-readonly-trace-v1-proposal.md)
defines 12 fixed native processes (plain/traced), unchanged frozen executables,
strict time/output bounds and B1 trial-attribution checks. Offline inspection of
three historical B4 traces found the expected 36 two-event/query windows. New
offline boundary/kernel/memory helpers, declared report/trial binder and diagnostic
output qualifier plus a bounded child primitive have 64 tests, including archived
fixtures, synthetic layouts and fake CPU commands; default checks pass with 112
report tests. Each historical B4 capture maps 972 kernels,
396 copies and 72 memsets to 36 windows, including RUNTIME host calls; this is
not current B1 evidence. Bookkeeping/overhead activity remains unattributed.
Large fixture parsing is child-isolated after an inherited-RSS test failure; logs are retained. The ordinal binder verifies declared
hashes/source layout and retains unavailable warmup timings as null; it does not
prove same-child launch provenance or output quality. The separate output qualifier
reuses the existing full codec validator and frozen v2 Python references, without
acceptance timings or new oracle runs. It passes archived regression fixtures;
no new capture exists to qualify. The child primitive enforces time/output limits,
retains receipts and tests owned-group cleanup before reaping. A follow-up guard
rejects cancellation/deadlines observed during setup before attempting a spawn;
two red-to-green tests cover that gap. Fake child environments are minimal, and
the original environment-bearing red log is restricted/unpublished. It is not a full
launcher or containment for collector SIGKILL/escaped descendants. A subsequent
import-only `readonly_trace_sequence.py` adds fixed order, exclusive plan/output,
shared deadline, streaming aggregate cap and free-space checks with nine offline
tests. Optional collector-side report receipts now bind fresh report hashes with
PID/argv/capture hashes; missing/nonregular/oversized/visibly changing reports fail.
Sequence report paths are unique and prevalidated. Ten new offline tests cover
this integration; latest default checks pass 131 reporting tests. The report read
cap is not a live disk quota. No independent producer identity or output quality
is established by these receipts. The integrated offline analyzer now connects
all 12 receipt/report/capture sets to four output-qualification groups and six
receipt-PID B1 trace joins, preserving all windows and per-condition instrumentation
comparisons. Three new tests (large fixtures subprocess-isolated) include 13 negative
bindings and real archived v2 outputs with synthetic traces; updated-hash corrupted logits fail
the actual numerical validator. Latest default checks pass 134 reporting tests.
This is consistency/qualification tooling, not independent process provenance or
new GPU evidence. Frozen-file preflight now matches v2 binary/source/fixture/
manifest/payload identities and retained CUPTI library/sources, validates CPU2/CPU0
and constructs the fixed argv plus isolated environments. Forty pinned entries
match locally, plus one observed taskset identity; no command was executed. The
explicit PATH/HOME/C-locale/PYTHONNOUSERSITE environment differs from v2's inherited
environment and requires native-library resolution review. Six offline tests and
the final default check pass (140 reporting tests); a test fixture filename
collision was fixed with the failed log retained. Bounded glibc loader --list and
Git metadata guards now pass eight tests (148 reporting tests in the full check).
Actual loader inspection finds 14 canonical libraries, with matching native codec
resolutions; lazy-loaded/JIT/driver libraries and live mappings are not covered.
Actual uncommitted-tooling rejection is retained; recheck clean worktrees after
publication. Source/lock stays anchored to d44beaa; Cargo manifests use measured
v2 fa040ed to retain the already-approved codec_memory example declaration.
Raw Git status evidence is private, not for publication. Actual clean-worktree
verification subsequently passed at 156d976 against accepted 3307dff. A dedicated
Linux subreaper supervisor now handles worker death/parent loss and adopted
session-escaping descendants, with eleven fake-CPU tests including failure paths
and unrelated-child survival. It still needs composition with sequence/telemetry;
supervisor SIGKILL/host failure are not guaranteed cleanup. Default checks now pass
159 reporting tests. An initial inherited-RSS resource assertion failure was fixed
by running its unchanged large-then-small measurements in one fresh interpreter;
the original threshold and allocation are unchanged, failed log retained. The new
telemetry worker adds atomic first-sample readiness, full CPU completion structure,
owned GPU shutdown and sequence cancellation on observer failure. CPU sampler
startup affinity is {0,2}, then verified {0}; observer/worker placement is CPU0.
Eight short synthetic-observer tests pass, including an in-flight cancellation
that prevents jobs 2–12 and a red-to-green delayed-readiness deadline guard; full
checks pass 167 reporting tests. No real 900-second window or GPU query was run
by this new worker. A subsequent import-only coordinator now waits for bound
first-sample readiness, runs the fixed sequence, cancels it on observer loss and
preserves the full sampler lifetime after ordinary benchmark failure. Tests compose
it with the actual independent subreaper and synthetic CPU observers: coordinator
SIGTERM/SIGKILL, observer failure/early exit, startup/deadline/output errors,
missing/malformed metadata and sequence setup failure. Eight tests pass and the
full check passes 175 reporting tests. The initial deadline test incorrectly assumed
job 2 could not start despite short jobs; its fixture now keeps job 1 in flight,
without changing production limits. All failed evidence remains in the protocol.
An offline telemetry validator now checks all CPU samples/counter arithmetic and
schedule/UTC-monotonic consistency, parses GPU units/UUID/local timezone, retains
optional-data gaps and brackets all 12 declared receipt windows. Its exclusive CLI
binds raw inputs by digest; ten tests include the real CPU serializer under a virtual
clock and the retained v2 CPU archive in a fresh interpreter. Default checks pass
185 reporting tests. GPU gaps and unavailable values remain explicit; coarse
bracketing is not continuous sampling or per-call clock attribution. No new real
collection ran. Frozen manifest/entry, launch-bound source/UUID/timezone, runtime
library/process provenance, supervisor/collector/telemetry cross-bindings and
remaining integration checks are still pending. Owned-child observation now adds
sampled executable/command/environment/affinity and executable-map checks via
optional bounded-child/sequence/coordinator hooks. Eleven CPU-only tests include the
fixed 12-child sequence and stopping on observation failure; default checks pass
196 reporting tests. Follow-up coverage rejects a factory silently returning None;
exec/exit snapshots with missing proc entries or uninstalled fields are counted as
incomplete, never promoted. Later complete workload observation is mandatory;
startup access denial cannot qualify without it, and later child exit cannot erase
a detected mismatch. Fixed five-case fake-child batches retain earlier race
failures and final passes. Initial failures exposed kernel maps reporting device 28 while
pathname/executable stat reports 29. Both identities are retained separately with
kernel map-link/path-inode checks, not treated as identical mapped-byte proof.
Direct map_files stat returned PermissionError; no permission/global-tooling changes were
made. Samples cannot prove whole-lifetime absence or immutable mapped bytes. Frozen
launch entry and final cross-bindings/readiness review remain pending. A new
postflight helper now validates all 12 process/receipt/declared-image/library sets
and rechecks prepared inputs, declaration files, loader/native libraries and mapped
paths after collection. Additional libraries are explicitly post-only, not invented
prelaunch pins or immutable mapped-byte proof. Nine tests include a real CPU-child
writer/reader compatibility test with synthetic loader declarations; default checks
pass 205 reporting tests. Caller-side enforcement of full telemetry/supervisor
completion before hashing, receipt/observation time binding and the final frozen
manifest/entry remain pending. No GPU/model benchmark was run. Selection is resolved; tooling
readiness and a fresh quiet window still block launch. Neither proposal is ready
to execute. No new measurements or settings changes occurred.

Latest preparation work adds `tools/bench/prepare_readonly_launch.py`: a `python -S`
metadata-only CLI composing the existing file, environment, loader and worktree
guards with tool/utility identities, exact GPU PCI/name/UUID, an explicit IANA
system-timezone symlink, fixed jobs and telemetry commands/limits. It reserves no
collection directory, starts no collection and always leaves readiness/availability
false. Eleven offline tests pass; default checks pass **216 reporting tests**.
Probes in these new tests are mocked; the real preparation CLI has not run. Current
clean-state/frozen-manifest verification, the launch entry, lifecycle/time/raw-file
bindings and enforced postflight ordering still need completion. See the selected
protocol for exact commands/log hashes and preparation limitations. Do not request
the quiet window until those remaining prerequisites pass review.

New `tools/bench/sample_linux_cpu.py` records CPU activity, driver frequency, SMT
siblings, pressure and system counters without process names or policy changes.
CPU **2 and 3 are SMT siblings**. Use observer CPU **0**, not CPU 3, for CPU-2
workloads. The earlier auditor-on-CPU-3 overlap affected **resource-only** collection,
not acceptance timings. A single fixed resource refresh has now corrected that
known confound: same 45 jobs/binaries/inputs, no concurrent audit/build/debugger
work, 301 telemetry samples covering the whole collection, all outputs qualified.
It is not a new performance gate; original resource evidence remains preserved.

The public `WeightBackend` raw-byte upload proposal is also **not approved or
implemented**. The isolated PoC is complete and production loading is unchanged.
A bounded defer/no-go outcome is allowed; do not silently extend the API or loader
ownership/trust boundary. No dependency, tolerance, schema, goldens or pin changes.

## Evidence and reproducibility

- Latest report: `docs/results/2026-11-11-phase2-linux-controls/README.md` and
  `evidence.json`; original controls archive `raw/linux-controls-v1.tar.xz`.
- New archive: `raw/cpu-resource-refresh-v2.tar.xz`, extracted under `reports/`,
  contains `phase2-cpu-telemetry-v1/` and `phase2-optimization-gate-resources-v2/`.
  It includes the fixed plan, checksums, complete raw reports, telemetry and summary.
- Existing live directories: `reports/phase2-optimization-gate-v1/`,
  `reports/phase2-optimization-{gate,controls}-threeway-v1/`,
  `reports/phase2-optimization-gate-resources-v{1,2}/`,
  `reports/phase2-optimization-{model-memory,metadata}-v1/` and
  `reports/phase2-cpu-telemetry-v1/`.
- Fresh trained capture: `reports/phase2-codec/cuda-controls-final-v1`; qualification
  at `reports/phase2-optimization-gate-v1/qualification.json`.
- The controls report documents clean-checkout reproduction of both summaries and
  three negative audits. Correct reproduction is still 72/78 controlled, not green.
- Useful read-only capture command (new output path required):
  `python tools/bench/sample_linux_cpu.py NEW.jsonl --cpu 2 --observer-cpu 0 --seconds 60`.
  This does not certify an idle workstation or authorize subsequent acceptance runs.

## Build and remote safety

- **Use isolated Cargo target directories across worktrees.** A diagnostic baseline
  build in the shared target caused stale accepted-baseline artifacts to be reused
  during a later current-source check. The new cache test caught it. The failed
  check/hashes are preserved; local CUDA/model release outputs were cleaned and
  rebuilt, checks passed and exact frozen candidate hashes were restored.
- `codec_bench` candidate hash:
  `034b609cb16af0a8d7c95eb11ffb056e49d0f0cdb2e98d775e716673adbe41ac`.
  Accepted baseline hash:
  `26437962fc06d0f43280e2b2b5608cacfa6d5931ab14f1f9d89c4a4f6cfc35a9`.
- Preserve all `.reference/*baseline*` worktrees/binaries and snapshots. The separate
  `.reference/phase2-memory-baseline` intentionally has the diagnostic example
  added locally; memory binaries live in `.reference/phase2-memory-bin/`.
- CUDA checks: `NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 ./tools/check.sh --cuda`.
- After the Linux gate, recheck `ssh macstudio` availability and competing work,
  inspect remote state, announce MLX testing and pull reviewed pushed commits into
  `~/projects/lexlapax/ncm-rs`. Do not overwrite unrelated work or assume SSH is live.
- `.pi/` is intentionally untracked persistent session state. Do not stage it.
  Keep weights, executables, ignored local reports and private state out of Git;
  reviewed compact evidence archives are committed separately.

## Completion boundary

Still required: resolve Linux acceptance, qualify Apple shared changes and vectorized
MLX resampling, complete Apple resource/correctness/performance evidence and both
final gates, finalize MLX opportunity dispositions and inspect every row of the
completion audit against the retained final candidate. Earlier Phase 2 acceptance
and green tests alone cannot close this expanded goal.
