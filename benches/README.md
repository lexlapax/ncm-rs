# Benchmarks

Phase 1's operator microprobe lives in
`crates/ncm-backend-cuda/examples/probe.rs`; the matching eager PyTorch oracle is
`tools/reference/operators.py`. `tools/bench/compare.py` validates and aggregates
three fresh-process reports per runtime.

Phase 2 adds the shared Sudoku `codec_bench` and text-tokenizer `asset_bench`
examples, paired with pinned Python tools. `tools/bench/run_phase2_linux.py` runs
three alternating fresh-process comparisons; `phase2_summary.py` validates and
aggregates them. See the [Linux Phase 2 report](../docs/results/2026-11-10-phase2-linux/README.md)
for retained raw data, throughput/memory definitions and the slower native codec
result. There is still no generator/end-to-end inference benchmark.

See the [first measured report](../docs/results/2026-11-09-cuda-operators/README.md)
for exact reproduction commands, warmups/trials, source hashes and limitations.
Tiny operator timings are not NCM throughput or language speedup claims.

Future benchmarks must follow [REPORTING.md](../docs/REPORTING.md), pairing
performance with quality for pinned Python, previous Rust and candidate Rust.
