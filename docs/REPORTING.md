# Performance and quality publication policy

Required deliverable: reproducible comparisons when inference is completed, for
each subsequent release/upstream upgrade, and for code changes affecting execution
or outputs. Benchmark and quality evidence is part of the change, not a final
marketing task. No results exist yet; see [results index](results/README.md).

## When reports are required

| Change | Required evidence |
| --- | --- |
| Initial end-to-end implementation | Python vs Rust on both devices, quality + performance + memory |
| Model, operator, kernel, dtype, cache, sampler, decoding or tokenizer change | Affected fixtures/quality plus previous Rust vs candidate vs Python on affected workloads; both devices if shared logic changes |
| Loader/CLI/batching change | Relevant cold-load/end-to-end/memory comparisons; output identity where applicable |
| Runtime/compiler/dependency upgrade | Affected device matrix and numerical/quality regressions; full suite for release qualification |
| Upstream pin/artifact/evaluator upgrade | Full supported-profile matrix, old/new Python and old/new Rust; distinguish model from runtime changes |
| Release | Full published matrix, reproducible commands, known gaps and regression disposition |
| Documentation-only/scaffold change | Explicit no-runtime-impact note; no fabricated or meaningless performance run |

Every PR classifies its impact and links results, or explains why measurements are
not applicable. A relevant unmeasured change remains unqualified: record missing
hardware/workloads as pending, not as passing. Experimental branches may proceed
without release qualification, but must not update the supported baseline.

## Honest baselines

**CUDA/Linux:** run pinned upstream Python NCM on the same GPU as Rust. Compare
upstream eager evaluation and, when feasible, a separately validated compiled
Python profile. Record precision and math modes; do not use an unoptimized FP32
Python run to imply a language speedup over BF16/fused Rust. Count codec decode,
initial sampling and final penalties consistently. Keep reference scoring outside
generation timing for both implementations.

**MLX/Apple Silicon:** upstream evaluation is CUDA-specific. An unmodified upstream
Apple GPU timing is not currently available. First establish a minimal PyTorch
MPS adaptation if its operator coverage permits it, explicitly flagging any CPU
fallbacks. Alternatively use an independently parity-validated **Python MLX port**
on the same MLX version as Rust. Label either as an adapted reference, never the
unmodified upstream implementation. Quality can still be compared against pinned
upstream CUDA outputs/fixtures. Do not claim a pure language speedup when the tensor
backend differs. If no defensible Apple baseline exists, publish absolute Rust
metrics and the limitation, not an invented Python comparison.

Keep Python/Rust CUDA comparisons, Python/Rust MLX comparisons and cross-hardware
comparisons separate. Cross-device speed ratios reflect hardware as well as software.
The paper's diffusion-baseline 10–31x claims are not this project's speedup target.
A CPU-to-GPU comparison is not a same-device runtime comparison.

## Measurement protocol

1. Pin Git commits, artifacts and evaluator versions. Fixed sample IDs, batch sizes,
   prefixes, generation lengths, sampling knobs, CFG and dtype; publish memory-fit
   exceptions explicitly. Compare identical work, including warmup and penalties.
2. Fixed hardware/power mode, no competing GPU workload, record thermals/clocks
   where available. Synchronize before/after timed device execution; force MLX
   graph materialization. Do not time dispatch alone. Use appropriate CUDA events
   for device-only timing and wall-clock for the complete pipeline.
3. Separate download (excluded), artifact load, compile/capture, first generation,
   warm generation and output postprocessing. Report both device-only and
   end-to-end scope. Include CPU transfers/f64 penalty fallback in end-to-end cost.
4. Baseline initial harness: at least 5 warmup batches and 30 measured batches,
   across 3 fresh-process runs; increase if noisy and justify exceptions for costly
   full evaluations. Alternate baseline/candidate order to reduce thermal bias.
   Report median, p95, dispersion/confidence intervals and sample counts, not the
   fastest single run. Report the actual synchronization and timing method.
5. Measure batch 1 latency and several common throughput batches that fit **both**
   implementations; separately report maximum-capacity runs. Measure prefix
   preparation, each hierarchy level, final codec decoder and token selection.
6. Report ms/sample, samples/sec, generated output token positions/sec and actual
   non-special text tokens/sec with definitions; level code counts are not text
   tokens. Include level count and actual generator function evaluations. NCM's
   “time to first draft” is not autoregressive time to first committed token.
7. Report peak device allocation/reservation and host RSS; on MLX identify unified
   memory counters and avoid comparing them as if identical to CUDA allocated VRAM.
   Include steady-state memory, CPU work, transfers and OOM/failure cases.
8. Quality uses [TESTING.md](TESTING.md) protocols with fixed evaluators and raw
   output hashes. Report deterministic disagreement, stochastic seed variation,
   confidence intervals and predeclared non-inferiority margins. Improvements in
   speed never excuse an undisclosed quality drop.

Lock workload-specific regression thresholds after Phase 1 establishes noise.
A suggested starting triage threshold is >5% median latency/memory deterioration
outside the measurement uncertainty; it is not a validated budget yet. Quality
margins must be set per task/profile before evaluating held-out results. Blocking
failures, waived regressions and changed precision must all be visible in reports.

## Result bundle and publication

Planned harness emits:

```text
reports/<run-id>/
  manifest.json       # revisions, commands, schema, licenses, hardware, environment
  metrics.json        # structured quality/performance; missing fields are null, not zero
  timings.jsonl      # raw timing trials and scopes
  outputs.jsonl      # sample IDs, seeds, codes/IDs/text when redistribution permits
  comparison.md      # readable tables, conclusions, regressions and limitations
```

Run IDs identify immutable source commits and UTC execution time. Manifest includes
Rust/Python/compiler/library/SDK/driver versions, code/artifact/data/tokenizer/evaluator
hashes, native build flags, precision, RNG/TF32/determinism settings, fixture version,
warmup/trial counts, command line and benchmark harness commit. Never include secrets.

- Commit concise reviewed summaries and compact machine-readable aggregates under
  `docs/results/<run-id>/`; update the results index without overwriting history.
- Store large raw bundles as GitHub release assets or versioned external artifacts
  with checksums and durable links. Actions artifacts alone expire and are not a
  permanent publication strategy. Respect dataset/output redistribution licenses.
- PR summaries link candidate results. Tagged releases link the full comparison
  and environments. The user approved public repository visibility; committed
  summaries and release assets will be public. A later GitHub Pages site should
  render the same reviewed summaries, not separate unverifiable charts.
- A later CI workflow may publish approved reports with narrowly scoped credentials.
  Never auto-publish untrusted PR outputs or run fork code on persistent GPU hosts.

## Required summary columns

Task/profile, hardware, runtime and Git SHA, precision, batch/length/prefix, quality
metric(s) and uncertainty, disagreement rate, median/p95 ms, samples/sec, peak
memory, speedup vs named baseline, measurement scope, artifact/evaluator hashes,
and status (`measured`, `pending`, `blocked`, `not-applicable`). Include failures and
negative results; do not selectively report only faster tasks.
