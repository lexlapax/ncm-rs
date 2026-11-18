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

## Repeated CUDA timing and allocator measurements

The longer run compares the LM1B-small reference rotary table and
1 MiB QKV candidate with pinned Python CUDA at development seed 11
and batch sizes 1, 4 and 16. Three fresh processes per runtime and
batch each make five warmup requests and 30 synchronized timed
requests. The Rust and Python seeds generate independent
streams. Repeat signatures are checked within each runtime and
shape. Model load is measured separately from request latency.

Run from the repository root using the same CUDA libraries and
pinned inputs as the short screen.

```sh
lib="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib"
link="$PWD/.reference/phase4-cudnn-probe-lib"
setpriv --inh-caps=-all env CUDA_COMPUTE_CAP=86 RUSTC_WRAPPER= \
  LIBRARY_PATH="$link" LD_LIBRARY_PATH="$lib" \
  cargo build --locked --release -p ncm-backend-cuda \
  --features cuda-cudnn,lm1b-experimental --example lm1b_small_capacity_probe
setpriv --inh-caps=-all env -u LD_LIBRARY_PATH OMP_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 \
  .reference/venv/bin/python tools/bench/run_lm1b_small_capacity_screen.py \
  --repeated --reference-rope-diagnostic \
  reports/phase4-lm1b-level7-replay-v1/reference-rope-tables.safetensors \
  --output reports/lm1b-small-repeated-v1
```

The process records include `VmHWM`, load time, and native
allocator high-water and current counters. Rust reports CUDA's
default async pool. Python reports the PyTorch caching allocator.
The counters cover their respective allocators from initialization
through the measured batch sequence. The sampled process GPU
observations are retained separately.

| Batch | Runtime | Median (ms) | p95 (ms) | Peak RSS (MiB) | Model load (s) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Rust | 45.64 | 48.86 | 1542.16 | 1.303 |
| 1 | Pinned Python | 61.89 | 65.57 | 2551.47 | 1.187 |
| 4 | Rust | 111.08 | 116.91 | 1533.97 | 1.295 |
| 4 | Pinned Python | 134.21 | 138.92 | 2542.63 | 1.200 |
| 16 | Rust | 374.27 | 379.09 | 1582.43 | 1.256 |
| 16 | Pinned Python | 434.54 | 438.47 | 2517.54 | 1.205 |

Median and p95 pool 90 measured requests per runtime and batch.
RSS and model load are medians of three process observations.
Rust's median request latency is 26.3%, 17.2% and 13.9% lower at
batches 1, 4 and 16. Model load is longer in each Rust cell.
All 18 processes completed, and all trial signatures repeat
within each runtime and batch. The first eight Rust batch-one
signatures also match the short-mode control.

The native allocator observations are in MiB and use the median
of three processes. Final figures are collected after synchronized
readback. Peak figures include model load and warmups.

| Batch | Runtime and allocator | Final used | Peak used | Final reserved | Peak reserved |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Rust CUDA default pool | 800.77 | 834.62 | 832.00 | 864.00 |
| 1 | PyTorch caching allocator | 993.50 | 1296.45 | 1382.00 | 1382.00 |
| 4 | Rust CUDA default pool | 800.77 | 935.83 | 832.00 | 1056.00 |
| 4 | PyTorch caching allocator | 1096.39 | 1799.87 | 1914.00 | 1914.00 |
| 16 | Rust CUDA default pool | 800.77 | 1340.67 | 864.00 | 1664.00 |
| 16 | PyTorch caching allocator | 1605.52 | 4121.64 | 4792.00 | 4792.00 |

Desktop processes stayed resident on the GPU. The observer
captured 165 snapshots with no query failures. Median sampled
process maxima for Rust and Python are 1,032/1,550 MiB at batch
one, 1,224/2,082 MiB at batch four, and 1,832/4,960 MiB at batch
16. These observations use roughly one-second sampling, with
only one or two batch-one Rust observations per process.
The timed workload ran one process at a time.

The raw evidence is in
`reports/phase4-lm1b-small-repeated-cuda-v1.yQCKWM/timing/`.
The `result.json` SHA-256 is
`382a29048a1b76eafbfb0c2d1aef544e2770bc57ea6e687eddf0512b62e5360b`.
The development benchmark leaves the original
[frozen LM1B quality failure](small-heldout-cuda-v1.md) in effect.

The manifest SHA-256 is
`7f02664adcc8ca5002688455ab87ac39f72ae165d9424ebdb983924a1375832b`.
The unchanged quality summary tool independently replays the
original held-out receipts and exits with the same failing entropy
interval. The reproduced summary SHA-256 is
`2e11e2b0aac10d152e58b888a23cda46b0df1b45285019d0ccf847e1f2f93002`.
The original quality protocol and its outcome remain authoritative.

The Linux host suite passed all 532 report tests. Focused release
build and Clippy passed for the CUDA capacity probes. The shared
default-pool helper also reproduces the pinned WMT14 batch-four
codes and IDs. The host log and WMT14 probe hashes are
`621ce44f07ecfef2193a7483588425d882605811ec58c49a0673aa13313f9851`
and `94b3ec921547878f8d3a3ddfe8a930c80d4d497427c055a326a05b13ddbb7eff`,
respectively.
