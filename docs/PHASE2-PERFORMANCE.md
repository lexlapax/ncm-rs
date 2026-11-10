# Phase 2 extension — performance research and remediation

Status: [production CUDA remediation](results/2026-11-10-phase2-cuda-remediation/README.md)
passes unchanged correctness and brings all codec latency cells within the Python
bands. [Extended Linux acceptance now passes](results/2026-11-10-phase2-linux-acceptance/README.md),
including controlled uncertainty follow-ups and separate startup/teardown evidence.
Apple remediation is next. The
[initial research](results/2026-11-10-phase2-performance-research/README.md) and
[predeclared thresholds](results/2026-11-10-phase2-performance-research/acceptance-v1.json)
remain immutable evidence. The user reopened
Phase 2 after its correctness/measurement baseline was published. **Phase 3 is
blocked until this extension passes Linux/CUDA first, then Apple/MLX.** Publishing
negative results was necessary, but is no longer sufficient for Phase 2 closure.
Historical reports remain unchanged evidence, not acceptance of these new gates.

## Problem statement and scope

Compare implementations and their native libraries, not languages in isolation.
Python dispatches optimized native kernels; Rust syntax alone guarantees neither
fewer launches nor lower memory. Optimize actual checked loading and codec APIs,
not just a microbenchmark, while retaining shared model semantics.

Published observations (Rust / Python; historical, not fresh measurements):

| Workload | Observation | Investigation priority |
| --- | --- | --- |
| CUDA B1/B2/B4 reconstruction | 18.890/4.668, 26.625/5.095, 28.879/5.881 ms | Critical: roughly 4–5× slower |
| CUDA B4 resident decoder block | 0.836/0.314 ms | Investigate independently of resampling |
| MLX warm checked codec load | 943.60/289.97 ms | Critical: roughly 3.25× slower |
| MLX final process host HWM | 3.40–4.42/0.287–0.288 GiB | Critical: roughly 12–15× higher |
| MLX B1 reconstruction | 24.791/28.417 ms | Preserve existing advantage; block timings are near parity |
| MLX Qwen tokenizer load | 183.64/144.38 ms | Material loading deficit |
| GPT-2 MLX / Qwen CUDA load | Small pooled native deficits | Remeasure before calling them regressions |

Sources: [Linux report](results/2026-11-10-phase2-linux/README.md),
[Apple report](results/2026-11-10-phase2-mlx/README.md). Tokenizer execution is
currently faster; CUDA loading and host memory are favorable. Do not trade these
advantages away to improve one headline.

In scope: pinned FP32 codec/primitives, checked artifact loading, tokenizer loading
and execution, benchmark/report tooling, and narrowly justified backend changes.
No generator, CLI vertical slice, precision relaxation, oracle replacement,
prefix cache, speculative framework rewrite, or inference release. Phase 5 still
owns broad optimization; measured Phase 2 deficits must be addressed now.

## Ordered work and deliverables

### 1. Establish trustworthy Linux measurements and a reproducible comparison

- Freeze the published baseline (`5dc9626`), source/binary hashes, artifacts,
  workload and oracle. Re-run old Rust, candidate Rust and pinned eager Python on
  the same GPU and host under controlled conditions, with separate build caches.
  Do not compare fresh candidate timings only against historical workstation runs.
- Audit equivalent output work: snapshots, unused reference outputs, tensor layout,
  input/output transfers, validation, drop/destruction, synchronization and cache
  state. Keep the existing public workload unchanged. Any additional diagnostic
  scope is separately versioned and must be applied fairly to both runtimes.
- Inspect timing boundaries for pending work or implicit synchronization. Distinguish
  host API wall time, CUDA stream spans and actual kernel durations. Instrumented
  barriers are diagnostic, not the acceptance timing path.
- Separate process/import/context startup, first load, first execution, warm load,
  steady execution and teardown. Existing first-load results are filesystem-warm,
  not disk-cold. Add a separately labeled startup measurement; do not flush global
  caches or change machine configuration without authorization.
- Test report regeneration from a clean checkout without ignored local reports.
  The initial audit found `phase2_mlx_summary.py` read baseline qualification
  from ignored `reports/` despite advertising a committed-only recheck. This gap
  is now fixed with checksum-pinned committed evidence and clean-checkout/mutation
  tests, not a permissive fallback. Continue auditing schema/source/runtime/output
  checks and timing equivalence before accepting measurements.
- Establish noise from independent paired process runs; at least three processes,
  five warmups and thirty trials each, more processes when uncertain. Record
  median, p95, dispersion and per-process results. Avoid treating 90 trials as
  90 independent processes. No competing GPU work; ask rather than stop others'
  processes. Missing profiling permissions/evidence is pending, not passing.

Deliverable: audited harness, reproducible baseline matrix, predeclared comparison
thresholds, and a ranked list of observed deficits before optimization.

### 2. Research and diagnose on Linux/CUDA

Research is a required work product, not a synonym for trial-and-error edits.
Read the actual pinned implementation alongside official documentation and relevant
upstream issues/patches. Record URLs, versions/commits, access dates, licensing,
applicability, and which claims have been experimentally verified.

Research sources and questions (to be investigated, not claimed as already read):

- Pinned NCM `models/quant.py`, `vae.py` and the qualified Python harness: which
  native operators execute, and what work does each API actually retain?
- Candle 0.11.1 source, CUDA backend/kernels, cuBLAS/cuDNN dispatch, allocation and
  synchronization behavior: exact supported native resampling/reduction paths,
  layout copies and batching. Review upstream fixes without silently upgrading.
- PyTorch operator source at the installed reference version: adaptive pooling,
  half-pixel interpolation, reduction tie order and FP32 accumulation. Translate
  semantics, not merely API names. Respect source licenses for any derived code.
- NVIDIA Nsight Systems/Compute and CUDA documentation: interpret kernel launches,
  CPU submission gaps, transfers, allocation calls, occupancy/bandwidth and sync.
  Begin with timeline profiling; use kernel profiling only on demonstrated hotspots.
- Rust allocation/I/O/serde/safetensors source and documentation: bounded reads,
  buffer growth, JSON allocation, checksum passes and tensor upload ownership.
  Use CPU sampling/allocation traces and phase counters, not RSS guesses.

Initial hypotheses grounded in local source, **not confirmed root causes**:

- `crates/ncm-backend-cuda/src/ops.rs::area1d/linear1d` loop over output positions
  and concatenate separate tensor results. Count launches, copies and allocations
  across actual level sizes; estimate their share of end-to-end latency.
- `select_last` uses several reductions/masks and creates index/sentinel tensors.
  Quantify its cost while preserving first-index and signed-zero tie behavior.
- Decoder attention/norm/GELU compositions, layout conversions and matmul dispatch
  may explain block scaling; a resampling fix alone may leave this deficit intact.
- `Artifact::load` reads/verifies every declared payload and parses JSON;
  `Codec::from_safetensors` preflights and copies runtime weights. Attribute bytes,
  allocations and CPU time to each phase, including files not used by the codec.

Deliverable: source-cited research note and profiler evidence tying each proposed
fix to a reproducible mechanism, expected gain, numerical risk and test plan.

### 3. Fix and accept Linux/CUDA before Apple work

- Try existing native primitives and fewer dispatches/layout conversions first.
  Evaluate vectorized resampling and reusable immutable shape metadata. Preserve
  overlapping area bins, half-pixel boundaries and declared reduction budgets.
- If profiling proves the remaining bottleneck needs a dedicated resampling or
  deterministic selection kernel, a small audited FP32 backend kernel is in scope.
  This supersedes the original restriction against optimization only for evidenced
  hotspots; it does not authorize speculative fusion or custom-kernel proliferation.
- Profile and correct decoder bottlenecks separately. Keep the simple composition
  as a test/reference path where useful; preserve CPU diagnostic behavior.
- Investigate shared loader/tokenizer allocation and parsing inefficiencies on
  Linux first. Prefer bounded allocation, fewer temporary copies and shorter
  lifetimes. Never remove checksums, preflight, symlink/path checks or validation
  to win a load benchmark. A codec-only importer or selective I/O path requires an
  explicit reviewed trust/output contract and a separately labeled comparison.
- Change one mechanism at a time; retain rejected experiments and before/after
  profiles. A custom kernel, dependency update, mmap or allocator replacement
  needs a documented safety/maintenance tradeoff, not an automatic escalation.
- Run unchanged operator/component/tiny/trained/tokenizer checks and `--cuda`;
  publish old-Rust/new-Rust/Python quality, latency, throughput and memory results.
  Do not start Apple optimization until the extended Linux gate is accepted.

### 4. Research, fix and accept Apple/MLX

After Linux acceptance, recheck Mac availability/worktree and safely pull pushed
reviewed changes. Revalidate all shared changes before Apple-specific optimization.

- Research the pinned MLX 0.32.2, mlx-c and mlx-rs array ownership/evaluation and
  memory APIs; Apple Instruments Allocations/VM Tracker, `vmmap`, Darwin malloc
  and process-memory accounting documentation. Distinguish live allocations,
  retained allocator pages, mapped bytes, RSS, high-water RSS and physical footprint.
- Extend retained `mlx_memory_probe` diagnostics: fresh-process host-only artifact
  load, array upload and codec construction; sample before/read/parse/hash/preflight/
  upload/drop/reload. Compare one-shot and repeated loads. Host-only reproduction
  implicates host loading but does not establish a leak or blame MLX.
- Use allocation stacks and byte accounting to identify the cause of multi-GiB HWM.
  Test read capacity growth, JSON parsing, temporary weight buffers and allocator
  retention independently. Do not mask the issue by resetting counters, dropping
  required validation, spawning a new process per load or changing the global allocator.
- Profile the 3.25× load gap and Qwen/GPT-2 tokenizer loading separately. Preserve
  exact tokenization and wrapper/postprocessing semantics; report source/canonical
  format differences and isolate comparable parsing work without hiding full loads.
- Preserve native execution latency while reducing load time and memory. Force lazy
  outputs, keep independent snapshots and run long repeated requests/load-unload
  tests to detect growing live memory. Record final RSS as well as monotonic HWM.
- If Apple diagnosis requires another shared change, validate it on Linux first,
  re-establish its gate, then retest Apple. The platform order still applies.

Deliverable: explained allocation/latency causes, tested fixes, unchanged real-Metal
parity, `--mlx` checks and new three-way measurements against the qualified adapted
Python MLX reference. No unmodified-upstream-Apple or pure-language speedup claim.

## Acceptance and stop rules

Before implementation, lock workload-specific thresholds from the refreshed
baseline in a reviewed measurement manifest. The Linux
[manifest](results/2026-11-10-phase2-performance-research/acceptance-v1.json) now
records these bands before production optimization. Candidate medians
no more than 5% above the matched Python baseline for codec scopes B1/B2/B4 and
codec/tokenizer loading/execution; p95 no more than 10% above. These are maximum
non-inferiority bands, **not measured noise or a license to regress**. Target parity
or better, not merely a reduction of the current 4–5× deficit. If uncertainty
crosses a gate, collect more independent processes; never widen a band afterward.

Apply the same locked 5% ceiling to comparable whole-process peak host memory
and steady-state live memory versus Python, and prevent material regressions
versus old Rust in metrics where Rust already leads. CUDA pool versus Torch
allocator figures are not comparable whole-process peaks; report them separately
and use matched process/device observations where available. On Apple, the multi-GiB
host-loading amplification must be removed and its cause demonstrated, not just
renamed allocator retention. Persistent growth after warmup must be investigated.

All cells in the agreed workload matrix must pass; do not average away a slower
batch/profile or offset poor loading with fast execution. Publish first-call,
startup and memory results even where a separate matched baseline is needed.
Unchanged quality gates remain mandatory: 18 operator cases, tiny/component codec
fixtures, all 642 trained traces, stable IDs/ties, independent snapshots, tokenizer
fixtures and rejection tests. No relaxed tolerances, pins, goldens or output scope.

If native API limits, numerical constraints, unavoidable validation cost or hardware
access prevent parity, retain Phase 2 as blocked and bring the evidence plus
alternatives to the user. Only the user may accept a residual deficit or change a
contract. A dependency/backend change is a separate reviewed decision, not a silent
substitution. Improvement alone, explanation alone, or unavailable hardware does
not close this extension.

## Completion checklist

- [x] Measurement audit, clean-checkout reproduction and baseline/noise thresholds.
- [x] Source-cited Linux research, profiles and ranked causal hypotheses.
- [x] Linux fixes, unchanged correctness, three-way measurements and extended gate.
- [ ] Apple research, allocation diagnosis, fixes and extended gate after Linux.
- [ ] Regression/performance checks and reproducible raw evidence for both devices;
      update plan, acceptance tracker, context and results index before closure.

The original planning change performed no experiments. Subsequent CUDA profiling,
production changes, correctness and three-way measurements are linked above.
They establish measured codec improvements, not full extended acceptance or MLX
qualification. Remaining deliverables must not be inferred as complete.
