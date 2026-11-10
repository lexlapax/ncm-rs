# Phase 1 Apple Silicon / MLX qualification

**Phase 1 runtime gate passed**, following the Linux gate. Not a Rust model or
inference release. Measured on Apple M1 Ultra / 64 GiB, macOS 26.7.3, Xcode 27.2,
Rust 1.99.1, mlx-rs 0.34.0 pinned to `adf439bf5361d8a02955de7aaf651b0c45900945`,
native MLX 0.32.2. Python 3.14.10 / MLX 0.32.2 is an **adapted operator/loader
baseline**, not unmodified upstream NCM. Environment, wheel tags and native
library hashes are in [raw/environment.json](raw/environment.json).

## Correctness and local checks

- All **18 unchanged PyTorch goldens** pass Rust MLX and Python MLX on real Metal,
  at the existing `atol=1e-5, rtol=1e-4`. Largest Rust absolute discrepancy is
  `4.77e-7`; neither goldens nor tolerances were changed.
- All **238 canonical tensors** match byte-for-byte after checked copied MLX load
  and readback, preserving F32/I64. A Metal kernel explicitly consumes a loaded
  weight. Unified-memory construction is not misrepresented as a CUDA upload.
- The five pinned Sudoku text-tokenizer cases pass on this architecture too.
- `tools/check.sh --mlx` passes: default CPU/loader/tokenizer checks (15 Rust tests),
  two real-Metal tests, CPU/MLX Clippy, docs, seven Python report-tool tests and
  the release operator probe. Full Linux `--cuda` validation still passes.
- A Rust probe also passes with a cleared environment and no reference venv on
  PATH. [Native dependencies](raw/native-dependencies.txt) contain no Python library.

## Operator baseline

Three fresh processes/runtime, five warmups then thirty measured trials/case/run.
Times include graph construction and synchronous `eval`, not merely lazy dispatch.
Inputs are prepared/evaluated before timing; output readback is excluded. The
FP16 intermediate is explicitly evaluated in both implementations. No compilation
or fast-attention override is enabled.

| Case | Rust median ms | Python MLX median ms |
| --- | ---: | ---: |
| matmul | 0.2951 | 0.2975 |
| conv1d | 0.3068 | 0.3044 |
| transposed conv1d | 0.3097 | 0.2993 |
| transposed conv, stride 89 | 0.3111 | 0.3122 |
| GroupNorm | 0.3444 | 0.3626 |
| LayerNorm | 0.3419 | 0.3612 |
| RMSNorm | 0.3202 | 0.3348 |
| area 7→4 | 0.3295 | 0.3496 |
| area 89→13 | 0.3723 | 0.4031 |
| area 1→1 | 0.2480 | 0.2580 |
| area 3→7 | 0.3248 | 0.3381 |
| linear 3→7 | 0.3390 | 0.3583 |
| linear 1→9 | 0.3389 | 0.3705 |
| linear 7→3 | 0.2861 | 0.3068 |
| linear 7→7 | 0.3354 | 0.3507 |
| padded prefix attention | 0.3015 | 0.3163 |
| split-half, reset RoPE | 0.2910 | 0.2977 |
| FP16 roundtrip | 0.4884 | 0.4949 |

P95, min/max, all raw samples and numerical errors are in
[summary.json](summary.json) and `raw/`. These tiny, synchronization-dominated
cases do not predict full-model speed. Do not compare these numbers to CUDA as
a language/runtime speed ratio; the machines differ.

## Loading, tokenizer and memory — including negative findings

| Workload | Rust median / p95 ms | Python median / p95 ms |
| --- | ---: | ---: |
| Checked copied unified-memory load | 938.66 / 961.08 | 281.87 / 289.36 |
| 1024 × five Sudoku encode+decode pairs | 0.747 / 1.903 | 9.615 / 9.717 |

Loading verifies every file/tensor hash and retains all 162,300,212 logical tensor
bytes, including training state. The Python loader is a trusted-input adaptation,
not the hardened Rust manifest parser. Both preflights check exact bytes and execute
a Metal kernel; initial Python-only leaf-evaluation observations were discarded
and both load baselines rerun with matching GPU execution preflights. Three fresh
processes/runtime, five warmups and thirty trials; operator/load ordering alternates,
tokenizer ordering is Rust then Python. First timed loads are recorded separately
and are diagnostic, not comparable startup figures (Python readback preflight
precedes its first timed load; Rust readback follows it). Filesystem caches are
warm; disk caches were not flushed. No prior Rust MLX implementation exists.

**Negative result:** this Rust copied-load path is about 3.3× slower than the
adapted Python path. Host peak RSS is also much higher: approximately **4.40 GiB
Rust vs 0.41 GiB Python**, measured over the full repeated-load process. This
needs profiling before any memory-efficiency or production-loader claim; do not
interpret the manifest's payload-byte bounds as a process-RSS bound.

MLX's own active allocation is 162,321,424 bytes with all weights loaded and zero
after all trials in both runtimes. Peak MLX allocations are approximately
154.8 MiB; caches retain approximately that amount after drop. Tiny Rust
post-kernel observations sometimes retain 1–2 KiB transiently before later evals.
These allocator counters do not explain away the much larger host RSS. The
operator suite peaks at at most 0.141 MiB of MLX-managed memory. No CUDA VRAM /
Apple unified-memory equivalence, model-memory budget, or leak-free server claim
is made. Raw `/usr/bin/time -l` output is retained for every Rust load process.

Background inference services were left intact and were idle in sampled process
listings; this is a workstation baseline, not an isolated benchmark host. Tokenizer
p95 shows a visible tail. No performance threshold was waived or fabricated;
Phase 1 establishes capability/baselines, not optimization or inference release.

## Reproduce and limitations

See [Apple reproduction commands](../../../tools/reference/README.md#apple-silicon--mlx-phase-1).
The initial crates.io source build failed because Xcode's optional Metal compiler
component is absent. Rather than modify global tooling, the tested immutable
bindings pin supports a project-local native prefix copied from the exact MLX-metal
wheel. No Python executable/module is needed by the Rust runtime. The tested wheel
requires macOS 26.2+; older OS tuples and source-built MLX remain unqualified.

Operator/Rust loader sources were built at `2d262e1`; the Python load preflight
correction is `2b19c87`. Raw reports embed exact source, fixture and artifact hashes;
Cargo.lock and native binaries did not change between those commits. The checked
in aggregator validates all three-process inputs before computing the summary:

```sh
python3 tools/bench/phase1_summary.py docs/results/2026-11-10-mlx-phase1/raw reports/new-mlx-summary.json
```

Full codec/generator parity, task quality, caches, mixed-precision model execution,
large shapes, production load budgets and redistributable packaging remain future
phase work. See the [Phase 1 acceptance checklist](../../PHASE1.md).
