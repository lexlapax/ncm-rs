# Expanded Phase 2 completion audit — INCOMPLETE

**Latest additional Apple evidence:** the user-selected [quiet-window matrix and control](tiny-borrow-apple-quiet-gate-v1.md) each pass 77/78. The controlled batch-4 p95 against previous Rust remains open. Earlier controls remain retained; investigate the Mac performance and measurement strategy before another full gate.

**Latest candidate:** TinyStories borrowing `5200924` passes the [initial and controlled Linux gates](tiny-borrow-linux-gate-v1.md) 78/78; its [Apple initial gate](tiny-borrow-apple-gate-v1.md) is 78/78 and fixed control is 76/78. Both batch-4 block p95 comparisons remain open.

**Previous candidate:** MLX projection-bias `a323b16` has a [76/78 Apple control](mlx-addmm-apple-gate-v1.md); TinyStories RSS/HWM remain open. A [finite current-runtime MLX memory observation](mlx-memory-addmm-v1.md) is retained.

**Previous candidate:** MLX affine normalization `2850331` passes Linux and Metal
correctness, but its [Apple fixed control](mlx-affine-apple-gate-v1.md) is
**75/78**; block p95 and Qwen encode remain open.

**Previous integrated candidate:** user-approved strict-FP32 projection
`928894b` passes the [Linux controlled gate](fused-linux-gate-v1.md) **78/78**,
full CUDA checks and fresh trained parity. Apple correctness passes, but the [fixed Apple control](fused-apple-gate-v1.md)
is **72/78**; six performance/memory cells remain open. The previous candidate histories
below remain retained.

**Historical TinyStories candidate:** `fb6c75e` improves Linux diagnostic decode,
but fresh [Linux controls](tinystories-linux-gates-v1.md) remain 71/78 and
75/78 after a user-confirmed quieter window. Apple testing of this candidate
is pending; do not rerun unchanged. The previous candidate is described below.

**Historical:** the [CUDA convolution candidate](conv-bias-cuda-v1.md) `b0a0058`
passes the [Linux controlled gate](conv-bias-linux-gate-v1.md) 78/78. Its first
[Apple control](conv-bias-apple-gate-v1.md) remains 74/78 despite passing Qwen;
Chrome was active during collection. After the user quieted the Mac, a
separately predeclared [quiet-window control](conv-bias-apple-quiet-gate-v1.md)
reached 75/78; batch-4 decoder block and TinyStories decode remain open. Prior
failed and passing candidate results remain historical evidence.

**Previous candidate:** the user approved the [GPT-2 byte-level dependency patch](gpt2-bytelevel-review-v1.md).
Candidate `1f7d2a8` passes the fresh [Linux 78/78 gate](bytelevel-final-linux-gate-v1.md);
runtime-equivalent `3a3fa86` passes full MLX checks and quality, but its
[Apple audit remains 76/78](bytelevel-final-apple-gate-v1.md) after one fixed
control. Historical results belong to the prior candidate.

**Previous candidate:** `621c565` passes the Linux CUDA check, trained parity and
78/78 controlled audit. Runtime-equivalent Apple candidate `884d8f6` passes the
full MLX check and trained parity but remains 77/78 after one fixed GPT-2
control. [Linux evidence](blocknorm-linux-gate-v1.md), [Apple evidence](blocknorm-apple-gate-v1.md).

**Historical update:** combined Linux candidate `4ab4d3a` passed 78/78; the newer combined Apple candidate `2df27c3` has a fresh 69/78 initial audit, including resident-block regressions. The committed MLX linear gather at `125d88d` passes trained parity and both hardware checks, but has not completed final acceptance. The additional BERT dependency change has been approved; final gates remain pending. Historical status paragraphs and diagnostic rows below do not supersede these results. [Linux evidence](final-linux-gate-v1.md); [Apple initial gate](final-mlx-initial-v1.md); [MLX evidence](mlx-linear-poc-v1.md); [BERT review](bert-ascii-review-v1.md).

Historical combined candidate `6c95d21` passes the full Linux CUDA check; its fresh
initial and controlled performance audits are **77/78**, with Qwen checked load
still failing after a fixed follow-up. B1 block and BERT checked load now pass.
[Evidence](combined-linux-gate-v1.md). Apple remains pending.

Historical candidate `a6290db` has a fresh Linux 45-process matrix and one bounded
108-process follow-up: the unchanged controlled audit passes **76/78**, leaving
B1 resident block and BERT checked load open. The initial audit was 71/78.
The prior 72/78 controlled result discussed below belongs to the earlier
candidate. [Evidence and reproduction](finite-scan-v1.md#fresh-linux-performance-gate).

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
with complete read-only telemetry and qualified outputs. No acceptance failure was waived. Those 72/78 Linux results are historical;
current platform gate statuses are stated at the top of this document.

Contract: [approved five-step plan](../../PHASE2-OPTIMIZATION-REVIEW.md),
[locked limits](acceptance-v1.json), [experiment ledger](experiments.json).

| Requirement / verification surface | Inspected evidence | Remaining before completion |
| --- | --- | --- |
| Step 1: accepted Rust `3307dff` / production `083c1e5`, unchanged qualified Python; source/binary/artifact/workload identity | [Baseline](baseline.json); current [Linux](tiny-borrow-linux-gate-v1.md) and [Apple](tiny-borrow-apple-gate-v1.md) gate inventories, clean exports, binaries and supplemental trait/module sources are bound | Repeat source binding if an execution-affecting candidate changes |
| Step 1: locked 78-cell matrix; median/memory 1.05 and p95 1.10 limits; process-level uncertainty | [Limits](acceptance-v1.json); current [Linux gate](tiny-borrow-linux-gate-v1.md) is 78/78 in both matrices; [Apple initial/control](tiny-borrow-apple-gate-v1.md) is 78/78 and 76/78, with earlier failures retained | Final Apple 78/78 with every comparator and unchanged limits |
| Step 2: reconstruction reuse; independent arbitrary-code decode and immutable snapshots; all named traces | [lower-risk report](../2026-11-11-phase2-low-risk/README.md); extended tiny/trained tests; exact untraced/traced output checks | Final candidate Apple validation; keep decision after both gates |
| Step 2: HF ID-only API and TinyStories in-place merges, independently measured | Lower-risk report, independent patches, 3-process allocation/timing reports; 2,047 exhaustive overlap/rank strings and long cases | Final Apple evidence; no general tokenizer latency claim from small warm-fixture gains |
| Step 2: position table, codebook norms, prepared resampling metadata | [prepared-state report](../2026-11-11-phase2-prepared-state/README.md); two incremental matrices/profiles; retry/concurrency/isolation/drop tests | Full model resource/lifecycle evidence; final controls; Apple shared-state checks |
| Step 2: hot weight-name / trace-label work | [Three native GDB captures](../2026-11-11-phase2-linux-controls/README.md): 168 direct model formatting calls / 7,992 requested allocator bytes per reconstruction; shim-only coverage failure preserved | Further typed-name/trace plumbing explicitly deferred; no isolated latency/RSS benefit is claimed |
| Step 2: checked loading copies/raw upload | Prepared-state report: 58 tensors / 59,586,048 bytes, bit-exact native readback, three processes, 5 warmups/30 trials; guarded alignment/ownership/overflow tests | Public API change has not been approved and is not implemented. Final disposition must not imply full-load/RSS gains from upload-only timing |
| Step 2: vectorized MLX resampling | [Linear gather](mlx-linear-poc-v1.md) passes expanded Metal tests and 642 trained traces; isolated and matched codec diagnostics show meaningful reconstruction improvements with higher MLX peaks. [Area hybrid](mlx-area-poc-v1.md) passes trained parity but its full-model timings are confounded and memory rises | Final linear candidate resource and acceptance checks; area remains deferred without demonstrated causal full-model benefit |
| Step 3: Linux correctness, `tools/check.sh --cuda`, fresh 642 trained traces / 2,500 decisions, tokenizer probes | [Candidate `5200924`](tiny-borrow-linux-gate-v1.md) passes the full CUDA check, 18 real-GPU operator cases and four tokenizer probes; its codec binaries match the prior fresh 642-trace qualification and cross-stream-tested candidate | Preserve source binding and requalify after any execution-affecting change |
| Step 3: Linux performance versus accepted Rust AND Python | [Candidate `5200924`](tiny-borrow-linux-gate-v1.md) passes 78/78 initially and after fixed all-workload control; earlier failures remain retained | Rerun if a later execution-affecting change requires it |
| Conditional clock diagnostic: owned restoration, absolute deadline, original policy, permissions, telemetry and fixed 18-job binding | [Clock proposal and runner audit](linux-clock-diagnostic-v3-proposal.md): fake-command restoration/deadline/external-cancellation tests pass; latest default checks pass 397 reporting tests (prior lifecycle-fixture and subsequent handoff-fixture failures retained; causes unestablished, assertion diagnostics expanded). Four handoff test methods cover default-budget restoration plus 18 real CPU-child observations, and nineteen rejected role/command/budget/private-obligation cases that still reset without workloads. The handoff validates live parent/self bindings and returns the original deadline; schema-2 obligations add supervisor/parent PIDs. Frozen preparation/entry and full final qualification remain missing. Five new CPU-only process-factory tests cover actual `/proc` observations for all 18 fake jobs, frozen command/env drift, image/injection mismatches and missing required libraries, without construction-time IO. Python command symlink spelling stays separate from its ELF identity; frozen hardware entry/postflight remain pending. Nine CPU-only coordinator tests cover readiness-gated restoration, 18 fake jobs, shortened sampler lifetime after ordinary failure, observer loss and coordinator TERM/SIGKILL. A separate observer supervisor reaps session-escaped samplers after telemetry-worker SIGKILL while the sibling restorer resets fake policy; unrelated child survives. Readiness binds the sampled sole live worker to the observer supervisor. A retained broken-warning-pipe red/green test preserves final failed restoration evidence/role cleanup. Read-only lifecycle regressions also pass. An opt-in 660/60-second outer capture passes a real 12-second fake cleanup-retry test; grace stdout backpressure and selector-error ownership handling have retained red/green regressions. Four composed CPU-only tests exercise actual supervisor/worker deadline handoff, failure exits and owned cleanup with fake policy/workloads. Observer loss/errors latch failure without suppressing reset; the new coordinator monitors its owned telemetry process, but the tested samples/commands are synthetic. A bounded 18-job worker passes nine fake-workload CPU tests, not frozen hardware/output qualification. Runner audit at `c82d847` matched v2; subsequent pure-plan extraction preserves exact commands/cwd/order/copies in before/after fake-CLI tests. Offline order/affinity probe confirms 18 jobs and CPU0-only startup rejection | Not launch-ready: bind frozen preparation and production roles/deadlines; bind the CPU-tested sibling-supervisor topology into a frozen production entry (no generic adoption-and-KILL ancestor around the restorer); validate actual per-job identities, full telemetry/outputs and private/intervention metadata. The user confirmed default policy and no external controller; required noninteractive permissions and launch readiness remain unproven. Request a fresh window only after readiness. No clock command or benchmark ran; no acceptance cell closes |
| Clock preparation: historical Python pins versus current runtime/startup identities | [Offline audit](linux-clock-python-preparation-audit.md): all 110 archived members verify; six Python metadata sets agree; current three helper sources and 36 distribution versions match. Initial five current-only snapshots, extra pip and startup shim documented. New file-only preparation has seven synthetic tests, including retained red/green same-byte inode replacement; three actual calls bind three historical sources and 43 current-only files/37 distribution versions, with all launch/runtime-provenance gates false. Reproducible without Torch import/GPU commands | No historical interpreter/library byte equivalence established. Finish native/Python loader inventory, current upstream check and frozen full-entry binding; bind actual runtime observations and postflight. Preserve intrinsic harness hash/Git work; add no collector probes/early rehash. No acceptance or launch gate closes |
| Clock preparation: standalone native/Python/Torch loader inventory | [Loader preparation](linux-clock-loader-preparation.md): six complete bounded utility receipts, 35 current-only libraries/3,547,116,642 bytes, native accepted/candidate resolution agrees; seven new tests and 22 focused tests pass. Torch source review identifies explicit ctypes preloads and two cuBLAS path differences across native/Python standalone listings | Standalone resolution is not Python runtime selection or import closure. Complete preload-candidate inventory/required-map review, current upstream verification, frozen full entry and live observations/postflight. No historical library-byte equivalence, new GPU check, benchmark or acceptance closure |
| Clock preparation: file-only preload candidates | [Preload follow-up](linux-clock-loader-preparation.md#file-only-preload-candidate-follow-up): six new tests, 20 focused/331 reporting tests pass. Actual reviewed-site inventory has 16 source entries, 15 current ELF candidates/1,956,767,857 bytes, optional NVTX absent here, no multiple matches; three new candidates beyond standalone listing | Candidate inventory is not actual runtime selection, complete sys.path or dependency closure. Proposed mandatory core-map subset and known-candidate validation still need implementation/binding into the frozen entry and final postflight. No package/policy change, Torch import, GPU query, benchmark or acceptance closure |
| Clock runtime: known candidate versus mandatory mapping | [Sample-time guard](linux-clock-loader-preparation.md#sample-time-known-candidate-guard): clock factory requires an explicit copied known inventory containing mandatory identities; first-sample optional libm mismatch fails child one and latches failure, absent optional surrogate allows eighteen jobs. No-IO independence/invalid inventory/read-only regressions pass; 22 focused/333 reporting tests | Actual inventory/subset compilation, frozen role binding and independent clock postflight remain missing. Pathname device/inode only, not same-inode byte mutation, arbitrary alias detection, continuous coverage or mapped-byte authentication. No GPU/benchmark run or acceptance closure |
| Clock preparation: compiled required/known mapping declarations | [Pure compiler](linux-clock-loader-preparation.md#compilation-of-the-observation-library-contract): five new tests, 12 focused/338 reporting tests pass. Actual retained inputs compile to native required 11/11, Python 10, known 40/3,825,790,348 bytes, 50 path bindings. All eighteen unchanged commands construct independent observers with no children/samples; normal Python startup preserved | Not fresh preflight or producer authentication. Independent clock postflight, upstream verification, fresh complete preparation, frozen entry/roles and hardware observations remain missing. No GPU/benchmark run or acceptance closure |
| Clock process-record consistency | [Independent validator](linux-clock-process-validation.md): all eighteen supplied jobs/receipts, recomputed map digests, known/required identities, declared injection absence, counters/limits and original-deadline/sequence-local clocks are checked. Six new tests include 40 negative variants and persisted evidence from eighteen real CPU fake jobs; 23 focused/344 reporting tests pass | Supplied-record consistency, not producer authentication or immutable mapped bytes. Restoration/observer lifecycle, full telemetry, actual codec outputs/capture files and frozen rehash remain unqualified. Fresh upstream/full preparation/entry and live policy gates remain missing; no acceptance closure |
| Clock final current-file component | [File checks](linux-clock-process-validation.md#final-current-file-component): supplied frozen files, known optional libraries, images, aliases and mapped paths rechecked; unknown libraries remain post-only. Nine new tests include eighteen actual CPU fake jobs into process/file validation; final 31 focused/353 reporting tests pass. Initial full check failed an existing lifecycle sequence; telemetry/cleanup completed, cause unestablished, assertion now includes sequence details; red log retained | Component only: ordering after restoration/telemetry/output/capture gates is not enforced yet. No complete frozen preparation, symlink-chain identity proof, immutable mapped bytes, producer authenticity or acceptance closure. Full finalizer and fresh preparation/entry still required |
| Clock restoration-record consistency | [Independent validator](linux-clock-restoration-validation.md): obligation, fixed budgets, original deadlines, command/clock order, first-reset spawn and cleanups checked; six new tests include 52 negatives and persisted fake-policy/default-budget/eighteen-CPU-job transport. Final 18 focused/359 reporting tests pass; initial duplicate discovery corrected without disabling tests | Not hardware enforcement/restoration, outer reaping, sibling topology/coverage or complete job/output validation. Bind same records into full lifecycle/telemetry/capture/output finalization before rehash last; fresh preparation/entry and all live gates remain pending |
| Clock sibling lifecycle consistency | [Composed validator](linux-clock-restoration-validation.md#sibling-lifecycle-follow-up): parent/PID/nested receipt chains, readiness digest/clocks, outer restoration 660/60 capture, ordered eighteen jobs and CPU-sample brackets checked with the restoration validator. Five new tests include 40 negatives and real sibling fake-policy transport, default restoration budgets/five-second telemetry/stub observations; 26 focused/364 reporting tests pass | Not live ancestry/reaping or observer/outer role authentication. Full 901-sample/900-second CPU/GPU/readiness content, raw captures and output quality remain separate; same-record finalizer must enforce rehash last. Current upstream/fresh preparation/roles/live gates remain pending; no acceptance closure |
| Clock telemetry contents/readiness | [Full-stream validator](linux-clock-telemetry-validation.md): unchanged 901-sample CPU/GPU content parsers, raw receipt/source/UUID/timezone binding, exact readiness metadata/first GPU row and complete restorer/eighteen-job bracketing. Six new tests include 43 negatives and retained readiness-time red/green; 21 focused/370 reporting tests pass | Virtual-clock/synthetic evidence, not actual 900-second GPU collection or enforcement. Gaps/unavailable data retained; no quietness/per-call clocks/hardware-restoration claim. Report/output/raw-capture gates and same-record rehash-last finalizer, fresh preparation/roles/live gates remain pending |
| Clock reported codec outputs | [Report qualifier](linux-clock-output-qualification.md): all six reports/runtime in fixed halves, raw receipt/source/reference-metadata binding, unchanged twelve-scope/stage/load/FP32 checker; current Python directly versus frozen Python. Three new tests include eighteen archived members/synthetic receipts and 43 altered cases; 7 focused/373 reporting tests pass. Initial no-op order-mutation fixture failure retained/corrected | Historical bytes with synthetic receipts are not new collection or live provenance. No raw-capture/report-file identity, unreported floating outputs, 642-trace parity or performance acceptance claim. Finalizer/role/fresh preparation/live gates remain pending |
| Clock sequence evidence files | [Private bounded reader](linux-clock-evidence-binding.md): 74 expected files, frozen plan/deadline, embedded/individual records, capture hash/length and report device/inode/size/hash binding; final visible metadata check. Eight new tests include actual eighteen-CPU-job transport and archived report qualification; 17 focused/381 reporting tests pass | No capture producer-inode evidence, atomic filesystem snapshot, producer authentication or complete process/lifecycle/output proof. Outer-capture/role binding, same-evidence rehash-last finalizer, fresh preparation/entry and live gates remain pending; no acceptance closure |
| Clock control-side evidence files | [Control-side binding](linux-clock-evidence-binding.md#control-side-follow-up): 21 additional files, six declared role receipts/capture digests, embedded/individual equality, CPU report inode and GPU digest checks; one 95-file inventory with sequence files. Six new tests include predeclared actual sibling fake-policy transport through the lifecycle validator; 25 focused/387 reporting tests pass | Observer-supervisor stdout explicitly post-only; no producer capture-inode/authentication or full telemetry claim. Coordinator/observer-supervisor/GPU live role binding remains for frozen entry. Same-evidence rehash-last finalizer, upstream/fresh preparation/live gates remain pending; no acceptance closure |
| Clock same-evidence finalization | [Import-only finalizer](linux-clock-finalization.md): one trusted runner call, 95-file binding, full telemetry/lifecycle/restoration/process/output gates, same receipt digests and intersecting clocks, visible metadata recheck then frozen rehash last; private exclusive completion publication. Four tests use actual gates over archived outputs/synthetic collection, 15 gate faults/two publication faults; 18 focused/391 reporting tests pass | Not complete authenticated preparation, live role binding/reaping, hardware policy restoration/enforcement or performance acceptance. Frozen restoration-aware production entry, upstream/fresh preparation and live gates remain missing; no acceptance closure |
| Current upstream/native worktree checkpoint | [Actual metadata snapshot](linux-clock-upstream-verification.md) at `f536960`: raw `c702742` commit hash, replacement objects disabled, 39 tracked files/877,235 bytes match HEAD/index/blob bytes, zero untracked/ignored entries; native accepted/candidate guards pass and production/manifests unchanged. Ten upstream/six native probes complete, 25-member archive consistency/privacy reviewed, 15 focused/391 reporting tests pass | Snapshot must be rechecked in fresh complete preparation; not atomic, Python import closure or historical interpreter/library identity. Frozen entry and policy/permission/enforcement/restoration/fresh-window gates remain pending. Session filenames stay private; no GPU/benchmark or acceptance closure |
| Complete clock preparation admission | [Path-count audit and approved increase](linux-clock-preparation-admission.md#approved-count-increase): 303 planned unique paths; clock-only limit raised to 512, with 261/303 admitted and 513 rejected in synthetic declaration tests | Fresh preparation must enforce the same limit and establish complete current identities/inventory; entry and hardware permission/restoration gates remain pending. Byte and all collection/telemetry/deadline/acceptance limits unchanged |
| Early clock declaration admission | [Shared pure guard](linux-clock-preparation-admission.md#early-fail-closed-admission-follow-up): retained historical red test reached a callback with 257 files; current limit is 512. File/byte/size/alias bounds and identity conflicts reject before filesystem inspection, directories or callback; 17 focused tests pass at the new boundary | Does not authenticate or prove a complete/fresh inventory, reserve capacity for unknown maps or authorize launch. Frozen rehash remains last; complete preparation/entry and hardware gates remain pending |
| Clock coordinator local admission | [Local-subreaper guard](linux-clock-diagnostic-v3-proposal.md#coordinator-local-subreaper-admission-follow-up): real process-local flag test retains red lifecycle-sentinel failure; query failure and unsupported platforms fail closed before outputs/branches. Two new tests; 32 focused/397 reporting tests pass. First full run failed separate handoff fixture: worker exit 1, fake reset/cleanup complete; cause unestablished, bounded diagnostics expanded. A fixed six-fresh-process CPU-only reproduction at `1ec6aaa` retained all 108 jobs/six successful fake resets and passed an independent 566-file audit; all attempts passed, so the historical failure remains unreproduced/unresolved, not repaired | Not ancestor inspection or proof of safe restoration-aware topology, frozen role binding or launch readiness. Full preparation/entry, file-count approval, policy facts and live gates still pending; no changed limits or acceptance closure |
| Step 3: Apple availability/state, reviewed pushed commits, Linux-first ordering | Linux gate passes first. The Mac checkout was clean, the baseline was separate, SSH was restored, and the reviewed candidate was fast-forwarded | Recheck competing activity before further Apple work |
| Step 3: Apple `tools/check.sh --mlx`, fresh trained qualification and cleared-environment tokenizer probes | [Candidate `3555c67`](tiny-borrow-apple-gate-v1.md) passes the full Mac check, 642 trained traces / 2,500 decisions and four tokenizer profiles | Requalify after any execution-affecting changes |
| Step 3: Apple 78-cell matrix against accepted Rust and qualified Python MLX | Current candidate `3555c67` has [78/78 initial and 76/78 fixed control](tiny-borrow-apple-gate-v1.md); TinyStories host memory passes, but batch-4 block p95 remains open against both comparators | Resolve the two block p95 cells without widening limits; retain all results |
| Step 3: Apple supplemental source binding | The unchanged `mlx_codec_bench` inventory lists model blocks but omits `crates/ncm-core/src/lib.rs`, which now contains the default projection-bias hook; Linux used a separate binding for its analogous omission | The [current gate](tiny-borrow-apple-gate-v1.md) binds the compiler dependency, clean source hash and measured binary; repeat for a later candidate |
| Step 4: median/p95/throughput, comparable host RSS/HWM, first load/execution, all codec batches/scopes and tokenizer profiles | Full raw baseline/lower-risk/prepared-state reports retain these fields | Final summaries on both platforms; allocator counters are not host RSS substitutes |
| Step 4: CPU/allocation attribution | Baseline and [fresh 45-process retained-candidate diagnostics](../2026-11-11-phase2-linux-controls/README.md); all raw outputs qualified; System/GDB scope distinctions explicit | Apple comparisons pending; CPU totals are not per-request CPU and no broad native CPU improvement is established |
| Step 4: startup/teardown, repeated model load/drop, bounded long-lived cache growth | The current [fused projection diagnostic](fused-projection-review-v1.md) covers 144 requests and ten drops with zero live CUDA pool bytes after each; its `codec_memory` binary is byte-identical to the integrated clean build. The Linux gate also includes nine lifecycle processes. Prior Apple reports record zero active MLX bytes and unchanged cache after 35 load trials | Current model runtime [Apple memory observation](mlx-memory-addmm-v1.md) covers 384 requests and each of three drops; active MLX bytes return to zero and sampled cache stabilizes, while host RSS grows. The subsequent tokenizer-only candidate retains an identical codec benchmark binary. This finite observation is not a general bound; reassess after model-runtime changes |
| Step 4: reproduce from clean checkout; raw favorable/negative evidence, hashes, exact commands and limitations | Current [Linux](tiny-borrow-linux-gate-v1.md) and [Apple](tiny-borrow-apple-gate-v1.md) controlled audits reproduce byte-for-byte from clean exports; each binds 120 native inventories, 180 shared Python copies and supplemental sources | Reproduce and bind any eventual passing final candidate; current Apple failure remains retained |
| Step 5: remaining CUDA layout/selection/resampling/convolution profile; CUDA graphs | [Fused strict-FP32 projection](fused-integration-v1.md) is user-approved, qualified and passes the Linux gate. Earlier [profile-backed deferrals](../2026-11-11-phase2-linux-controls/README.md) retain no-go decisions for unproven layouts and graphs | Current Apple control remains 76/78; do not claim unimplemented graph or custom-kernel benefits |
| Step 5: compact merge lookup / heap BPE; BERT per-token decoder cleanup | Ledger records bounded deferrals using measured warm latency/allocation work and semantic/maintenance costs; no new structure or decoder shipped | No general unseen/long-input benefit claim is justified; such future claims require separately labeled workloads |
| Step 5: MLX fast norms, compilation, attention | Global fast norms were deferred; targeted decoder fast normalization and attention are integrated, and later layer-norm affine and projection-bias fusions pass trained parity. The separate [RMS fusion](mlx-rms-affine-diagnostic-v1.md) did not show a target improvement; [pure compilation](mlx-compile-poc-v1.md) remains deferred for startup/cache tradeoffs. Current block p95 control remains open | Final acceptance remains open; preserve explicit array-input and bounded-specialization constraints |
| Step 5: level-zero tables and new loader ownership | [No-go for this iteration](../2026-11-11-phase2-linux-controls/README.md): 2,916,352-byte tables without established break-even; ownership/trust change not justified by upload-only PoC | Do not describe unimplemented tables or ownership changes as accepted optimizations |
| Contracts: pins, schemas, goldens, FP32 tolerances, first-index ties, complete authenticated preflight, Python-free production, retained output semantics | [Dependency scope audit](dependency-scope-audit-v1.md) verifies vendored patch scope, package inventory, pins and tracked fixtures against the baseline; current trained checks retain existing tolerances. Raw upload remains isolated | Complete final semantic/source/test audit and document reporting-tool extensions without changing model/artifact/workload schemas |
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
