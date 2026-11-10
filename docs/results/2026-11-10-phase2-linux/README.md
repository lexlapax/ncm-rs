# Phase 2 Linux gate: codec and tokenizer measurements

**Linux/CUDA Phase 2 acceptance is complete. Apple/MLX remains pending.** This
establishes a correctness-qualified measured baseline, not an efficiency claim,
full text-model support, generation benchmark or inference release. Native codec
execution is substantially slower than eager PyTorch. No numerical budget, pin or
golden was changed, and no speed threshold is being waived.

## Evidence and protocol

- [Full tiny/trained codec qualification](../2026-11-10-phase2-codec/README.md):
  all 236 tiny and 642 trained captures pass, with stable IDs exact; two near ties
  and the worst internal error at 94.2% of budget remain disclosed. Reconstruction
  agrees with Python on 445/445 positions and 5/5 sequences, but recovers only
  437/445 original input positions. This is not Sudoku-solving quality.
- [Tokenizer conversion/parity](../2026-11-10-phase2-tokenizers/README.md): all
  four profiles, 356 encode and 2,314 decode comparisons; Sudoku unchanged.
- [Machine-readable summary](summary.json), [raw reports](raw/),
  [environment and exact commands](raw/environment.json). The raw bundle is
  approximately 3.2 MiB of JSON/text, including timings and first logits, not
  weights. All 30 fresh-process reports are committed for reproducibility.
- Three fresh processes per runtime/workload; alternating Python/Rust order,
  five warmups and thirty measured trials, plus separate first calls. Tables pool
  90 samples; p95 is nearest rank. Summary also includes standard deviation, IQR,
  extrema and individual process medians. These are descriptive statistics, not
  confidence bounds from 90 independent processes.
- Source base `0396b56` plus the measured benchmark changes in this milestone;
  exact native/reference source SHA-256 and binary fingerprints are in raw data.
  Runtime changes are only `CandleBackend: Clone` to reuse one device context
  across repeated constructions; no model, operator or tokenizer math changed.
  There was no previous Rust full-codec timing baseline to compare against.
- RTX 3060 Ti 8 GiB, driver 615.78.08, CUDA compiler 13.4; i7-12700K, performance
  CPU governor; Rust 1.99.1. Torch 2.15.0+cu132, FP32, TF32 off, math SDPA,
  deterministic cuDNN, one CPU thread. All installed reference lock versions are
  checked. Tokenizers use the separate established reference environment.
- This is a workstation, **not an exclusive GPU reservation**. Initial device
  use was 4,809 MiB and reported utilization 0%; existing model-server/desktop
  processes were not stopped. [Process observations](raw/gpu-process-utilization.txt)
  reported numeric SM activity only for the benchmark processes; `-` is not
  proof of inactivity. [Telemetry](raw/gpu-telemetry.csv) records 41–54°C,
  SM clocks 210–1,950 MHz and 4,809–5,126 MiB device-global usage. No power/clock
  controls changed. Short unobserved interference remains possible. Batch-one
  Rust reconstruction process medians ranged 18.18–19.51 ms; do not overinterpret
  small differences. These limitations qualify the measurements, not parity.

## Matched codec execution

[Workload](../../../tests/fixtures/codec-bench-v1.json): solved-board patterns,
length 89, width 256, twelve levels / 322 codes, batches 1/2/4. Encode retains all
12 cumulative latents; decode retains all 12, logits and selected IDs;
reconstruct retains both sets. Pipeline timing starts with host input IDs and
ends with host code/output IDs; logits/snapshots stay on GPU. Destruction,
correctness checks and hashing are excluded. The resident middle-block case has
no host transfers. CUDA events measure **stream spans including submission gaps**,
not summed kernel time; synchronized wall time is the headline. The full decoder
and selection are separately profiled below. Generator prefixes/NFE/generated
text throughput are not applicable to this codec-only phase.

Python encode uses the pinned original helper, including its unused `z_levels`
resampling. Decode explicitly fixes the upstream helper's aliased snapshots via
out-of-place accumulation, without extra unused level outputs. Its final logits
are checked bit-exact against the unmodified upstream decoder at every batch.
This small documented output-contract adaptation is not an upstream math change.
Native snapshots were independently qualified in the complete codec report.

All measured input/code/output hashes and retained floating-output counts match
between runtimes and across processes. The benchmark adds **44,856 native logit
comparisons** at the unchanged `5e-4 + 2e-4*abs(reference)` budget, all passing;
worst budget fraction 0.212. The summary's 89,712 comparisons additionally include
reference self/repeat checks. Benchmarks do not replace the internal trace gate.

Wall median / p95 milliseconds (90 trials per cell):

| Batch | Scope | Rust Candle | Python eager |
| ---: | --- | ---: | ---: |
| 1 | encode | 10.518 / 10.996 | 3.574 / 3.621 |
| 1 | decode | 8.401 / 8.659 | 1.654 / 1.668 |
| 1 | reconstruct | 18.890 / 19.549 | 4.668 / 4.696 |
| 1 | resident decoder middle block | 0.308 / 0.310 | 0.257 / 0.283 |
| 2 | encode | 14.454 / 14.537 | 3.484 / 3.534 |
| 2 | decode | 12.106 / 12.288 | 1.742 / 1.753 |
| 2 | reconstruct | 26.625 / 26.753 | 5.095 / 5.126 |
| 2 | resident decoder middle block | 0.401 / 0.404 | 0.284 / 0.306 |
| 4 | encode | 15.671 / 15.754 | 3.899 / 3.922 |
| 4 | decode | 13.234 / 13.268 | 2.140 / 2.156 |
| 4 | reconstruct | 28.879 / 29.107 | 5.881 / 5.931 |
| 4 | resident decoder middle block | 0.836 / 0.841 | 0.314 / 0.394 |

Reconstruction throughput Rust/Python: batch 1 **52.94/214.24 boards/s**, batch 2
**75.12/392.57**, batch 4 **138.51/680.19**. Position throughput is respectively
4,711/19,067; 6,686/34,938; 12,327/60,537 positions/s. These are 89-position Sudoku
sequences including separators, not generated or non-special text tokens. The
summary gives per-sample milliseconds and throughput for every scope. Batch 4 is
a tested common batch, **not a maximum-capacity claim**; capacity/OOM sweeps were
not performed on the occupied workstation.

A separate instrumented reconstruction pass measures encoder/pre-quant, all
12 quantization levels, full decoder logits and argmax. Batch-one median stream
spans in ms:

| Stage | Rust | Python |
| --- | ---: | ---: |
| encoder/pre-quant | 0.058 | 0.096 |
| level 0, learned cascade | 0.267 | 1.919 |
| intermediate levels 1–10, range | 0.769–1.412 | 0.095–0.102 |
| terminal level 11 | 0.101 | 0.091 |
| full decoder logits | 0.343 | 0.145 |
| output selection | 0.022 | 0.006 |

Per-level/per-batch raw trials are retained. The intermediate composed resampling
path is the main observed native cost; first-index tie selection also costs more
than Torch's argmax. Instrumentation perturbs dispatch and excludes accumulation,
residual subtraction and unused reference `z_levels` between spans: do **not** sum
these figures into a headline or call them kernel-only time. No speculative
kernel/precision optimization was introduced to hide the negative result.

## Loading, first execution and memory

Both codec loaders authenticate the canonical artifact inventory and validate
payload/tensor hashes, then construct only the codec. Python retains upstream
training buffers on GPU; native uploads 58 runtime weights (59,586,048 bytes).
Python's constructor also initializes weights before replacement. This compares
actual checked loading paths, not identical parsers or a pure-language effect.
Imports and CUDA context initialization are excluded. Filesystem cache is warm;
"first load" is process-first, **not disk-cold**.

| Metric | Rust | Python |
| --- | ---: | ---: |
| warm checked-load median | 184.49 ms | 267.61 ms |
| first checked load, 3-process range | 191.57–201.55 ms | 279.52–290.21 ms |
| first B1 reconstruction, range | 68.04–72.37 ms | 222.74–233.47 ms |
| final process host HWM, range | 559.37–561.77 MiB | 1530.27–1540.22 MiB |
| B1 reconstruction allocation peak | 60.83 MiB | 102.92 MiB |
| B2 reconstruction allocation peak | 64.82 MiB | 104.32 MiB |
| B4 reconstruction allocation peak | 72.82 MiB | 107.11 MiB |
| B1/B2/B4 reserved peaks | 64/96/96 MiB | 134/138/144 MiB |

Native allocation/reservation figures are CUDA **default async-pool** current and
high-water counters, not all process VRAM: context, modules and cuBLAS allocations
outside the pool are excluded. Torch uses its caching allocator's counters.
Their scopes differ; the raw data retain both and device-global snapshots, which
must not be described as process peaks. Peaks reset after preparation before each
scope, and include first/warm calls. Steady post-output-drop pool/allocated usage
is roughly 56.9–57.2 MiB native / 57.1–57.4 MiB Python, including the resident
middle-block input. RSS/HWM is whole-process Linux `/proc/self/status`, including
load loops, preflight and instrumented diagnostics, not just tensors. Byte bounds
are not RSS bounds. Lower observed memory does not make the slower codec efficient.

## Text tokenizers

Exactly the same 89 fixture strings/profile are encoded and the resulting fixed
IDs decoded in both skip-special modes. Every first/warm output equals the pinned
fixture and all cross-process/runtime framed output hashes match. Native execution
is Python-free. HF comparisons include Python wrapper overhead around the same
Tokenizers version; this is not a universal Rust-vs-Python language comparison.
These short synthetic Unicode/special/whitespace cases are not a corpus benchmark.

Median / p95 ms for one complete 89-string pass:

| Profile | Encode Rust / Python | Decode Rust / Python | Skip-special decode Rust / Python |
| --- | --- | --- | --- |
| BERT | 0.999/1.054 · 2.129/2.285 | 0.465/0.482 · 0.816/0.863 | 0.392/0.406 · 0.740/0.764 |
| GPT-2 | 0.878/0.913 · 2.151/2.191 | 0.113/0.116 · 0.474/0.484 | 0.121/0.126 · 0.483/0.490 |
| Qwen | 0.998/1.098 · 2.207/2.297 | 0.104/0.110 · 0.463/0.489 | 0.110/0.113 · 0.478/0.499 |
| TinyStories | 0.240/0.250 · 2.649/2.772 | 0.016/0.017 · 0.290/0.298 | 0.015/0.017 · 0.516/0.536 |

The summary reports strings/s, total positions/s and **non-special positions/s**.
Per pass total/non-special positions: BERT 1522/1261, GPT-2 2280/2274, Qwen
1980/1974, TinyStories 2988/2982. [Throughput definitions](../../../tests/fixtures/tokenizer-bench-v1.json)
list the canonical special IDs; this metadata does not alter timing or goldens.

| Profile | Warm load median Rust/Python ms | Process HWM range Rust/Python MiB |
| --- | ---: | --- |
| BERT | 9.31 / 19.74 | 21.73–21.90 / 90.11–90.50 |
| GPT-2 | 42.49 / 54.83 | 57.59–57.67 / 113.88–114.83 |
| Qwen | 170.65 / 169.52 | 184.11–185.02 / 224.70–231.11 |
| TinyStories | 3.80 / 28.73 | 11.68–11.73 / 75.00–75.18 |

Each tokenizer profile gets its own fresh process. Loading authenticates files
inside timing, but **formats differ**: Rust canonical JSON vs original pinned
source files through the upstream Python wrapper. TinyStories Python uses the
bounded data-opcode interpreter, never executable pickle. Qwen native load is
slightly slower in pooled medians; no load-speedup claim is made. Python imports
are excluded from load timing but included in host HWM. First-call/load trials and
all distributions remain in raw reports.

## Optional compilation and validation

The [compiled FP32 probe](raw/compiled.json) fails before producing outputs:
fullgraph Inductor cannot specialize the data-dependent NumPy `argmin` used by
upstream `PhiPartiallyShared.__getitem__`. No upstream patch or silent graph-break
fallback was used. This profile has **no qualified timing baseline** and no full
internal-trace acceptance. Even a passing output preflight would not establish
that gate. The earlier compiled FP16 failure remains unchanged.

Validation: `tools/check.sh --cuda` passed (30 base Rust tests, real-CUDA codec
suite, all 18 unchanged operators, formatting/Clippy/docs/syntax); all 18 optional
reference tests passed. Report tests now total 11, including incomplete matrices,
changed hashes, scope mismatch, output errors, NaNs and fixed tolerance failures.
Benchmarks/diagnostics are offline examples; the only production addition is a
cloneable device wrapper. No new dependency or production Python path was added.

Reproduce from the repository root (fresh output directories):

```sh
NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 \
  cargo build --release -p ncm-backend-cuda --features cuda --example codec_bench
cargo build --release -p ncm-tokenizers --example asset_bench
python3 tools/bench/run_phase2_linux.py reports/phase2-linux-repeat
python3 tools/bench/phase2_summary.py reports/phase2-linux-repeat reports/phase2-linux-repeat/summary.json
# Recheck committed evidence without GPU/assets:
python3 tools/bench/phase2_summary.py docs/results/2026-11-10-phase2-linux/raw /tmp/phase2-linux-summary.json
TORCHINDUCTOR_CACHE_DIR="$PWD/.reference/phase2-inductor" TORCHINDUCTOR_COMPILE_THREADS=2 \
  .reference/venv/bin/python tools/reference/codec_compile_probe.py \
  artifacts/canonical/sudoku-easy-qualified .reference/upstream-ncm reports/phase2-linux-repeat/compiled.json
NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 ./tools/check.sh --cuda
.reference/venv/bin/python -m unittest discover -s tools/reference -p 'test_*.py' -v
```

Next: recheck Mac availability/worktree, safely pull this reviewed pushed milestone,
implement/qualify the same shared codec on MLX, investigate its historical load/RSS
weakness, and publish matched Apple evidence. Phase 2 overall remains open; large
oracle/native-trace/tokenizer archive publication is still pending.
