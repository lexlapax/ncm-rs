# Phase 1 Linux load/memory and tokenizer baseline

Linux portion of Phase 1 accepted for the scoped Sudoku/runtime spike; Apple
qualification is separate. This is not a general model importer or inference release.
Environment: RTX 3060 Ti 8 GiB, driver 615.78.08, CUDA toolkit 13.4, Candle 0.11.1,
Rust 1.99.1, Python 3.14.9, Torch 2.15.0+cu132. Artifact manifest and 238 tensor
identities are those of the [artifact report](../2026-11-10-sudoku-artifacts/README.md).

| Workload | Rust median / p95 ms | Python median / p95 ms |
| --- | --- | --- |
| Checked copied canonical load + CUDA upload | 201.61 / 205.42 | 241.20 / 248.25 |
| 1024 × five Sudoku encode+decode fixture pairs | 0.471 / 0.488 | 9.925 / 10.072 |

Each row/runtime pools 30 measured trials after five warmups in three fresh
processes. Raw trials and per-process medians are in [summary.json](summary.json)
and `raw/`. Load-runtime ordering alternates; tokenizer ordering is Rust then
Python. Download, process/import/context initialization, tensor destruction, and
correctness readback are outside the timed load scope. First-load times are
separately recorded, **not cold-disk measurements**: filesystem caches were not
evicted. The Python canonical loader is an adapted trusted-input benchmark, not
the upstream pickle loader and not a hardened production validator. Both hash all
files/tensors, copy-load all three bundles including training state, and synchronize
uploads. No previous Rust implementation exists; these are initial baselines.

Memory behavior (all tensors resident together, 162,300,212 logical bytes):
- Rust host VmHWM: 538–541 MiB; Python process max RSS: 1007–1009 MiB.
- Rust global CUDA used-memory delta on load: 160 MiB; returns to the initial
  snapshot after drop and after 35 repeated loads.
- Python global used-memory delta: 172 MiB; its allocator retains reservations
  after drop. Torch allocation/reservation peaks are recorded in raw reports.
- CUDA driver snapshots are device-global, **not process allocation peaks**.
  Host RSS includes different language runtimes. Other desktop GPU allocations
  (~4.4 GiB) were resident; a later utilization sample was 14%. No other service
  was stopped. Treat these as workstation baselines, not isolated release numbers.

Quality checks are exact tokenizer goldens and the prior all-tensor readbacks;
full `tools/check.sh --cuda` still passes all 18 operator cases and 15 Rust tests.
No task-quality, model-generation, or language-level speedup claim follows.

## Compile feasibility and capability disposition

`torch.compile(fullgraph=True)` succeeds numerically for **17/18** operator cases
under the unchanged `1e-5/1e-4` tolerance. The FP32→FP16→FP32 case loses rounding
(3/6 elements disagree). See [compile report](raw/compile.json). Compiled mixed-
precision behavior is **unsupported**, not waived. Eager FP32 plus explicitly
materialized casts remains the baseline. This single-process compile experiment
is a capability check, not a compiled timing comparison. Shared backend traits
remain deliberately unfrozen; phase 2 will build only on tested operations.

The 56 model traces now have a durable [research prerelease asset](https://github.com/lexlapax/ncm-rs/releases/tag/phase1-fixtures-v1).
Its SHA-256 is in the original oracle report. This is not an inference release.

## Reproduce

Build with `NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4`:

```sh
cargo build -p ncm-backend-cuda --features cuda --example load_probe --release --locked
cargo build -p ncm-tokenizers --example tokenizer_probe --release --locked
# Create fresh report paths; run each pair in three fresh processes.
target/release/examples/load_probe artifacts/canonical/sudoku-easy-qualified reports/new-rust-load.json
.reference/venv/bin/python tools/reference/load_probe.py artifacts/canonical/sudoku-easy-qualified reports/new-python-load.json
target/release/examples/tokenizer_probe tests/fixtures/sudoku-tokenizer-v1.json reports/new-rust-tokenizer.json
.reference/venv/bin/python tools/reference/tokenizer_probe.py .reference/upstream-ncm tests/fixtures/sudoku-tokenizer-v1.json reports/new-python-tokenizer.json
TORCHINDUCTOR_COMPILE_THREADS=4 .reference/venv/bin/python tools/reference/compile_probe.py tests/fixtures/operators-v1.json reports/new-compile.json
```

Measured source digests are recorded in `summary.json`; loader raw reports also
embed source/manifest hashes. Sources were a candidate atop `bbdf422`. TinyStories
unigrams/byte-BPE and generalized runtime config validation remain later-profile
work, not capabilities established by this Sudoku-specific spike.
