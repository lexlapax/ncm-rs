# LM1B-small CUDA multi-batch development screen

The explicit CUDA sampling and CUDA FP64 penalty candidate was compared with
pinned upstream Python at batch sizes 1, 4 and 16 using non-held-out seed 11.
Both paths timed initial unigram sampling, generation, codec decoding, final
penalties and code/ID host readback, excluding model loading. Three fresh
processes per runtime ran in alternating order, with two warmups and eight
synchronized timed batches each. Each process recorded signatures of its eight
generated code/ID pairs; signatures matched across all three processes within
each runtime and batch size. Python and Rust use independent random streams:
these signatures establish repeatability, not cross-runtime output parity or
quality at the new batch sizes. The separately scored batch-eight development
results are [recorded here](small-rust-development-quality-v1.md).

| Batch | Runtime | Median | p95 | Samples/s at median | Median process VmHWM |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Python | 0.061500 s | 0.062375 s | 16.26 | 2,666,476 KiB |
| 1 | Rust | 0.047244 s | 0.048036 s | 21.17 | 1,637,596 KiB |
| 4 | Python | 0.133519 s | 0.135166 s | 29.96 | 2,659,864 KiB |
| 4 | Rust | 0.111512 s | 0.112622 s | 35.87 | 1,626,308 KiB |
| 16 | Python | 0.427107 s | 0.428623 s | 37.46 | 2,635,024 KiB |
| 16 | Rust | 0.367945 s | 0.370921 s | 43.48 | 1,676,032 KiB |

Each timing summary pools 24 batches. The supervisor records per-process
timings, PIDs, source and binary hashes, and `nvidia-smi` activity at roughly
one-second intervals. Observed per-process GPU memory maxima in MiB were:

| Batch | Python observations | Rust observations | Samples per process |
| ---: | --- | --- | --- |
| 1 | 1370, 1370, 1370 | 1032, 1032, 1032 | Python 2–3; Rust 1 |
| 4 | 1612, 1612, 1612 | 1192, 1192, 1192 | Python 3; Rust 1–2 |
| 16 | 2992, 2992, 2992 | 1864, 1832, 1864 | Python 5–6; Rust 4 |

These are sparse samples, not measured GPU peaks; a single batch-one Rust
sample per process cannot establish peak memory. Four other GPU
processes were resident before the run. The report SHA-256 is
`92f68b1df52610b9d7dbb5a14302505901e69e331d453706f6e55b964abafed0`.
The first screen is retained as `phase4-lm1b-small-capacity-screen-v1`
(result SHA-256 `2e23cae4a4ed616f83687ca664f4ac4f5033855c65f34404feb346812dc56e78`);
output signatures also match between the two screens within each runtime.
The earlier [batch-eight screen](small-cuda-performance-screen-v1.md) is
separate. The source and binary hashes in the run manifest identify the
candidate as tested while these new probe files were uncommitted.

Commands:

```sh
setpriv --inh-caps=-all env RUSTC_WRAPPER= LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo build --locked -p ncm-backend-cuda --release --features lm1b-experimental,cuda-cudnn --example lm1b_small_capacity_probe
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/bench/run_lm1b_small_capacity_screen.py --output reports/phase4-lm1b-small-capacity-screen-v2
python -m unittest discover -s tools/bench -p test_lm1b_capacity_screen.py
setpriv --inh-caps=-all env RUSTC_WRAPPER= LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo clippy --locked -p ncm-backend-cuda --all-features --all-targets -- -D warnings
RUSTC_WRAPPER= ./tools/check.sh > reports/phase4-lm1b-capacity-check-v1.log 2>&1
```

This is a development screen, not the release performance or memory gate.
It does not establish official-seed quality, cross-runtime stochastic parity,
exclusive-GPU performance, Metal, or Phase 4 completion. The known controlled
draw failure remains visible. Evidence-validator tests, all-feature CUDA lint
and the standard repository check passed. Raw measurements and captures are
ignored local artifacts.
