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
tests; default checks now pass 121 reporting tests. These primitives do not replace
whole-run supervision, telemetry/affinity, frozen identities or integrated
launch/report provenance, which remain pending. Selection is resolved; tooling
readiness and a fresh quiet window still block launch. Neither proposal is ready
to execute. No new measurements or settings changes occurred.

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
