# Benchmarks

Phase 1's operator microprobe lives in
`crates/ncm-backend-cuda/examples/probe.rs`; the matching eager PyTorch oracle is
`tools/reference/operators.py`. `tools/bench/compare.py` validates and aggregates
three fresh-process reports per runtime. No model benchmark exists yet.

See the [first measured report](../docs/results/2026-11-09-cuda-operators/README.md)
for exact reproduction commands, warmups/trials, source hashes and limitations.
Tiny operator timings are not NCM throughput or language speedup claims.

Future benchmarks must follow [REPORTING.md](../docs/REPORTING.md), pairing
performance with quality for pinned Python, previous Rust and candidate Rust.
