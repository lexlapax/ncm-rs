# Expanded Phase 2 completion audit — INCOMPLETE

This is a working requirement-to-evidence checklist, **not a completion claim**.
The objective is to complete the five approved optimization steps within Phase 2:
freeze the accepted baseline; independently evaluate lower-risk opportunities;
finish Linux then Apple correctness/performance gates; measure resource/lifecycle
costs throughout; disposition conditional opportunities and publish a verified
final candidate. Phase 3, generator/CLI work and a licensed inference release are
outside this objective. An uncertain or missing gate is not achieved.

The requested handoff is preserved in [continuation notes](HANDOFF.md). Offline
preparation led to one [completed fixed diagnostic](linux-diagnostic-v2-results.md)
after explicit quiet-window confirmation: 54 qualified processes, all identities
and 901-sample coverage verified. B1 block bimodality persists; outcome is unresolved
recurrence, not a replacement acceptance gate. No runtime gate is closed by it.
The resource-only CPU observer placement was corrected in a fixed 45-job refresh,
with complete read-only telemetry and qualified outputs. No acceptance failure was
waived: Linux remains 72/78 and Apple work remains pending.

Contract: [approved five-step plan](../../PHASE2-OPTIMIZATION-REVIEW.md),
[locked limits](acceptance-v1.json), [experiment ledger](experiments.json).

| Requirement / verification surface | Inspected evidence | Remaining before completion |
| --- | --- | --- |
| Step 1: accepted Rust `3307dff` / production `083c1e5`, unchanged qualified Python; source/binary/artifact/workload identity | [baseline.json](baseline.json), [baseline report](README.md); saved binaries match historical accepted executable hashes | Recheck final-candidate identities against final matrices on both platforms |
| Step 1: locked 78-cell matrix; median/memory 1.05 and p95 1.10 limits; process-level uncertainty | [acceptance-v1.json](acceptance-v1.json); baseline negative 73/78 audit preserved | Final 78/78 on each platform, including every comparator and unchanged limits |
| Step 2: reconstruction reuse; independent arbitrary-code decode and immutable snapshots; all named traces | [lower-risk report](../2026-11-11-phase2-low-risk/README.md); extended tiny/trained tests; exact untraced/traced output checks | Final candidate Apple validation; keep decision after both gates |
| Step 2: HF ID-only API and TinyStories in-place merges, independently measured | Lower-risk report, independent patches, 3-process allocation/timing reports; 2,047 exhaustive overlap/rank strings and long cases | Final Apple evidence; no general tokenizer latency claim from small warm-fixture gains |
| Step 2: position table, codebook norms, prepared resampling metadata | [prepared-state report](../2026-11-11-phase2-prepared-state/README.md); two incremental matrices/profiles; retry/concurrency/isolation/drop tests | Full model resource/lifecycle evidence; final controls; Apple shared-state checks |
| Step 2: hot weight-name / trace-label work | [Three native GDB captures](../2026-11-11-phase2-linux-controls/README.md): 168 direct model formatting calls / 7,992 requested allocator bytes per reconstruction; shim-only coverage failure preserved | Further typed-name/trace plumbing explicitly deferred; no isolated latency/RSS benefit is claimed |
| Step 2: checked loading copies/raw upload | Prepared-state report: 58 tensors / 59,586,048 bytes, bit-exact native readback, three processes, 5 warmups/30 trials; guarded alignment/ownership/overflow tests | Public API change has not been approved and is not implemented. Final disposition must not imply full-load/RSS gains from upload-only timing |
| Step 2: vectorized MLX resampling | Qualified coordinate/reduction constraints documented in review | Not started; Linux gate first; real Metal PoC, expanded numerical cases, memory/startup tradeoff and disposition required |
| Step 3: Linux correctness, `tools/check.sh --cuda`, fresh 642 trained traces / 2,500 decisions, tokenizer probes | [Controls checkpoint](../2026-11-11-phase2-linux-controls/README.md): clean rebuilt checks, fresh capture/qualification and cleared-environment probes pass; frozen benchmark hashes restored/verified after documented stale-build incident | Recheck if candidate changes; correctness alone does not close performance |
| Step 3: Linux performance versus accepted Rust AND Python | [Controls checkpoint](../2026-11-11-phase2-linux-controls/README.md): all 360 predeclared processes complete; fresh matrix 74/78, controlled audit **72/78**, all failures retained | Six acceptance comparisons remain open. Completed v2 diagnostic retains B1 block clusters despite quiet/isolated observation; five other failures did not recur, not a waiver. [Selected read-only v1](linux-readonly-trace-v1-results.md) ran once after reviewed preparation/fresh availability: AP/CP complete, AT exceeded fixed 1 GiB capture, nine jobs unstarted; full telemetry/cleanup complete. No qualified traced joins or frozen postflight; data insufficient. No retry/quota change. Conditional clock fallback still needs policy/controller/permission/restoration/fresh-window gates; a separate acceptance gate remains required. Never selectively exclude runs or widen bands |
| Conditional clock diagnostic: owned restoration, absolute deadline, original policy, permissions, telemetry and fixed 18-job binding | [Clock proposal and runner audit](linux-clock-diagnostic-v3-proposal.md): fake-command restoration/deadline/external-cancellation tests pass; latest default checks pass 311 reporting tests. Four handoff test methods cover default-budget restoration plus 18 real CPU-child observations, and nineteen rejected role/command/budget/private-obligation cases that still reset without workloads. The handoff validates live parent/self bindings and returns the original deadline; schema-2 obligations add supervisor/parent PIDs. Frozen preparation/entry and full final qualification remain missing. Five new CPU-only process-factory tests cover actual `/proc` observations for all 18 fake jobs, frozen command/env drift, image/injection mismatches and missing required libraries, without construction-time IO. Python command symlink spelling stays separate from its ELF identity; frozen hardware entry/postflight remain pending. Nine CPU-only coordinator tests cover readiness-gated restoration, 18 fake jobs, shortened sampler lifetime after ordinary failure, observer loss and coordinator TERM/SIGKILL. A separate observer supervisor reaps session-escaped samplers after telemetry-worker SIGKILL while the sibling restorer resets fake policy; unrelated child survives. Readiness binds the sampled sole live worker to the observer supervisor. A retained broken-warning-pipe red/green test preserves final failed restoration evidence/role cleanup. Read-only lifecycle regressions also pass. An opt-in 660/60-second outer capture passes a real 12-second fake cleanup-retry test; grace stdout backpressure and selector-error ownership handling have retained red/green regressions. Four composed CPU-only tests exercise actual supervisor/worker deadline handoff, failure exits and owned cleanup with fake policy/workloads. Observer loss/errors latch failure without suppressing reset; the new coordinator monitors its owned telemetry process, but the tested samples/commands are synthetic. A bounded 18-job worker passes nine fake-workload CPU tests, not frozen hardware/output qualification. Runner audit at `c82d847` matched v2; subsequent pure-plan extraction preserves exact commands/cwd/order/copies in before/after fake-CLI tests. Offline order/affinity probe confirms 18 jobs and CPU0-only startup rejection | Not launch-ready: bind frozen preparation and production roles/deadlines; bind the CPU-tested sibling-supervisor topology into a frozen production entry (no generic adoption-and-KILL ancestor around the restorer); validate actual per-job identities, full telemetry/outputs and private/intervention metadata. Default-policy/no-controller confirmation and permissions remain missing; request a fresh window only after readiness. No clock command or benchmark ran; no acceptance cell closes |
| Step 3: Apple availability/state, reviewed pushed commits, Linux-first ordering | No new Apple run; historical baseline/worktrees preserved | Recheck competing work/access, announce first run only after Linux gate, pull reviewed commits, preserve old baselines |
| Step 3: Apple `tools/check.sh --mlx`, fresh trained qualification and cleared-environment tokenizer probes | Earlier acceptance is historical only | New final-candidate evidence required; shared fixes return through Linux first |
| Step 3: Apple 78-cell matrix against accepted Rust and qualified Python MLX | Not collected for this extension | Complete strict acceptance, retaining negative runs and controls |
| Step 4: median/p95/throughput, comparable host RSS/HWM, first load/execution, all codec batches/scopes and tokenizer profiles | Full raw baseline/lower-risk/prepared-state reports retain these fields | Final summaries on both platforms; allocator counters are not host RSS substitutes |
| Step 4: CPU/allocation attribution | Baseline and [fresh 45-process retained-candidate diagnostics](../2026-11-11-phase2-linux-controls/README.md); all raw outputs qualified; System/GDB scope distinctions explicit | Apple comparisons pending; CPU totals are not per-request CPU and no broad native CPU improvement is established |
| Step 4: startup/teardown, repeated model load/drop, bounded long-lived cache growth | [Fresh nine-process lifecycle and six whole-model memory captures](../2026-11-11-phase2-linux-controls/README.md): 144 requests/process, stable prepared state, zero live pool usage after all 60 model drops | Apple lazy ownership/drop evidence; recheck retained candidate if production changes |
| Step 4: reproduce from clean checkout; raw favorable/negative evidence, hashes, exact commands and limitations | Both earlier audits and checkpoint summaries/profiles reproduced; source/archive identities verified | Reproduce final summaries/audits and verify final archive contents/source binding |
| Step 5: remaining CUDA layout/selection/resampling/convolution profile; CUDA graphs | [Profile-backed deferrals/no-go decisions](../2026-11-11-phase2-linux-controls/README.md) and ledger: no justified persistent duplicate layouts, qualified custom kernel or bounded graph break-even | Revisit only if final profiles change priority; no speculative kernel is required for closure |
| Step 5: compact merge lookup / heap BPE; BERT per-token decoder cleanup | Ledger records bounded deferrals using measured warm latency/allocation work and semantic/maintenance costs; no new structure or decoder shipped | No general unseen/long-input benefit claim is justified; such future claims require separately labeled workloads |
| Step 5: MLX fast norms, compilation, attention | Research only; explicit array-input and bounded-specialization constraints recorded | Re-profile after Apple lower-risk work; evidence-backed dispositions with precision/setup/memory costs |
| Step 5: level-zero tables and new loader ownership | [No-go for this iteration](../2026-11-11-phase2-linux-controls/README.md): 2,916,352-byte tables without established break-even; ownership/trust change not justified by upload-only PoC | Do not describe unimplemented tables or ownership changes as accepted optimizations |
| Contracts: pins, schemas, goldens, FP32 tolerances, first-index ties, complete authenticated preflight, Python-free production, retained output semantics | Current checkpoints preserve these; raw-upload remains an isolated example | Final diff/source/test audit; document reporting-tool extensions without changing model/artifact/workload schemas |
| Publication and closure | Plan/baseline/lower-risk/prepared-state checkpoints pushed; context map updated in every commit | Final evidence index, ledger, both acceptance reports, synchronized plan/tracker/context; inspect actual artifacts before marking goal complete |

## Offline resident-block timing inspection

Inspected source at `41ab9e5` against accepted `3307dff`, without running a
benchmark or changing policy. `git diff 3307dff --
crates/ncm-backend-cuda/examples/codec_bench.rs` is empty. Direct byte comparisons
also establish that CUDA `ops.rs` is unchanged, the `decoder_block` /
`decoder_block_traced` section of model `blocks.rs` is unchanged, and the CUDA
`evaluate` implementation is unchanged. This does **not** establish identical
compiled code, process state, allocation placement or runtime performance.

Timing boundaries in `codec_bench.rs::timed` and `call` explain an important
interpretation limit:

- A device synchronization precedes the start event. The wall clock starts after
  that event is recorded.
- The resident block executes and calls `evaluate`, which checks the output and
  synchronizes the device, **before** the end event is enqueued. Wall timing then
  includes end-event recording and synchronization.
- Consequently, the CUDA-event span is not a sum of kernel execution durations.
  It can include GPU idle intervals during host submission or between the inner
  synchronization and end-event enqueue. Matching wall/event clusters alone do
  not separate host overhead from device execution time or establish a clock cause.
- Output destruction and the ID-hash check occur outside the timed closure; the
  returned tensor remains alive through end synchronization. The block receives
  a retained latent prepared before the scope loop, follows reconstruction, and
  uses one separately recorded first call plus five warmups and thirty retained
  trials. No change to these boundaries or scope order is proposed here.

This narrows the next read-only investigation's question: attribute the spread to
kernel durations versus submission/idle gaps with per-operation trace evidence,
while reporting instrumentation overhead and retaining the uninstrumented controls.
It is not yet an executable protocol or authorization for a new run. Existing
one-second telemetry and event spans cannot supply that attribution retrospectively.
No acceptance cell closes; Linux remains **72/78**, Apple pending. Verification:
source inspection, `git diff`, and Python byte/section comparisons; no new hardware
checks or measurements.

[Offline pool-counter inspection](linux-diagnostic-v2-results.md#offline-pool-counter-and-retained-trace-follow-up)
verified all twelve archived native reports: B1 reserved current/peak are 64 MiB
in both implementations; used counters are constant within each runtime across
both timing clusters. These aggregates do not attribute per-request allocation
latency. Existing optimization CUPTI captures cover two reconstructions, not v2
resident-block trials; no retrospective trace attribution or gate closure follows.
The report includes executable standard-library reproduction with archive/member
hash checks; it ran successfully without new measurements.

## Current stop conditions and boundaries

- Unresolved performance uncertainty/regressions are not waived because a scope's
  source appears unchanged. Preserve initial and follow-up runs.
- Public `WeightBackend` raw-byte upload awaits explicit review. Existing loading
  remains unchanged. A documented bounded no-go/defer outcome is permitted by the
  approved plan; an unimplemented API is never described as implemented.
- Missing hardware evidence, an incomplete process run, or unsupported features
  are pending/errors, not a CPU fallback or passing result.
- The active objective must remain incomplete until every required row has an
  evidence-backed final disposition and both runtime gates actually pass.
