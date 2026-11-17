# WMT14 CUDA final-selection candidate

The uncached WMT14 reference backend's generic last-axis first-maximum
selection constructs full-logit equality and index intermediates. On the
batch-32 generated logits (`[32, 64, 151680]`), an isolated, synchronized
20-sample diagnostic measured median 94.050 ms for this selection alone. The
existing level-stage diagnostic measured a 125.370 ms decoder-plus-selection
checkpoint. A separately instrumented pinned Python decoder measured 24.957 ms;
its decoder hook excludes the subsequent argmax, so these are not directly
comparable complete decoder timings. Both profilers reproduced the retained
batch-32 codes and IDs; they run in separate processes and include different
synchronization, and desktop GPU activity was not excluded. They are
bottleneck diagnostics, not headline performance evidence.

A bounded opt-in CUDA operator reads each 151,680-value row once and returns
the first maximum without allocating full-logit temporary arrays. Its explicit
backend constructor is separate from the existing reference backend. The
one-pass kernel ignores NaNs in the same way as Candle's FP32 `fmax` reduction;
all-NaN rows retain the generic `u32::MAX` sentinel. Real-CUDA tests compare
the candidate and generic operator on 64 full-width rows including ties across
threads, infinities, all-negative-infinity, all-NaN, mixed NaN/negative
infinity, signed zero, and deterministic unique maxima. Unsupported shapes
are rejected by the explicit operator. The opt-in backend selects this operator
only for largest-index selection at `[1..=32, 64, 151680]`; other shapes and
default constructors retain the existing path.

On retained generated logits, the explicit candidate and generic selection
returned identical IDs. Separate one-process synchronized selection screens
measured candidate median 2.927 ms versus the generic 94.050 ms. A preliminary
30-trial full-generation screen measured candidate batch-32 median/p95
1.183275/1.187579 s and matched retained codes and IDs; its process and
conditions differ from earlier Python measurements, so it is not a paired
performance comparison or gate pass. The first CUDA integration test used a
tensor from a separate device context and correctly failed; the corrected
context test, targeted CUDA tests and feature-specific Clippy passed. The full `tools/check.sh --cuda` rerun passed including 18 device cases
(`reports/phase4-wmt14-fast-argmax-cuda-check-v2.log`). Its first run failed
the process-lifecycle fixture `test_real_writers_and_owned_process_samples_compose`;
the isolated retry and unchanged full rerun passed, and the failed log is
retained. Targeted real-CUDA integration and feature-specific Clippy/build
pass. Source-bound full-corpus comparisons, frozen quality scoring, and
repeated same-condition performance remain pending.

Evidence retained under ignored `reports/`:

- Generic selection result SHA-256 `d244eff38c00db6a86360d1d225c3adcb4f9a4d12cf2530b53edca614e04bf04`.
- Candidate selection result SHA-256 `127bbb9277c0b8545eb41503ec6e9057b1266a4f0eace5d9a815f24326f02aee`.
- Candidate full-generation screen SHA-256 `d8faf686394a5f77bd2bdc898e3de89eab1a51857a5407faa161113d66837cfc`.
- Rust stage profile base/CFG result SHA-256 `d49c9dfee4b84a1876479ec0757bc61b98a383256911d6731904244dff8bf352` / `94a7533185c6ecc1c6dc7b56d983176ed7e124c2783b53606e3852fa3adae177`.
- Pinned Python all-level and decoder profile SHA-256 `5375815c347292694c9dc4247d1d98f6315feadda7131952daec3ec21c5c962b`.

The CUDA activity and timing limitations above remain; no cross-platform or
full Phase 4 acceptance is claimed.

## Source-bound corpus and repeated timing

At clean commit `0e7fede1ab47254335f4558d7f3a77e91dea73e1`, both
35-row source-bound smoke comparisons passed. Full base and CFG=1 captures
then independently matched all 939,939 retained Python codes and 192,192 raw
IDs in each mode. The base and CFG outputs were respectively byte-identical
to the previously scored reference-candidate captures, with SHA-256 values
`d26b8a3b16a4ad07565f52d3e61f22b30e2512ea10348e674a24e9ed52fecc79` and
`ff04571bfc72be02aee45ef7f64b6b4226431dca2264a7fd02d30be44294ef0f`.
The unchanged frozen full and held-out scorer replays both pass, with hashes
`84a87dcf08a7ca8e9498aab4cc198fa3e53bca341637d8124d83ea5e1e8e7eda` and
`915cae3bf2351db1f90bd290ebbe79353a4b9bdb91a67cb494d30791403f89bc`.
These are retests on previously inspected data, not new blind held-out studies.

The candidate and pinned Python were then compared in alternating fresh
processes, three per runtime per cell, five warmups and 30 synchronized trials
per process. All output codes and raw IDs matched within and between runtimes.
Median/p95 pool 90 trials. RSS is peak process VmHWM; GPU memory is the maximum
observed process allocation from approximately one-second samples, not an
allocator peak. All durations are milliseconds per batch.

| CFG | Batch | Python median / p95 | Rust median / p95 | Python / Rust RSS MiB | Python / Rust sampled GPU MiB |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 64.458 / 64.922 | 58.174 / 58.759 | 2093.2 / 1849.0 | 1360 / 1222 |
| 0 | 4 | 178.781 / 179.688 | 166.098 / 167.117 | 2093.3 / 1843.3 | 1492 / 1414 |
| 0 | 16 | 653.012 / 654.774 | 616.087 / 619.095 | 2093.7 / 1839.1 | 2300 / 1574 |
| 0 | 32 | 1276.834 / 1278.932 | 1206.310 / 1208.483 | 2093.3 / 1874.8 | 3726 / 2792 |
| 1 | 1 | 102.535 / 103.460 | 97.359 / 98.140 | 2093.2 / 1843.6 | 1364 / 1254 |
| 1 | 4 | 331.508 / 332.432 | 310.853 / 311.784 | 2093.2 / 1849.2 | 1538 / 1320 |
| 1 | 16 | 1239.487 / 1241.437 | 1182.031 / 1183.560 | 2093.2 / 1875.0 | 3072 / 2184 |

At pooled median the Rust/Python samples per second are respectively
17.19/15.51, 24.08/22.37, 25.97/24.50 and 26.53/25.06 for base batches
1/4/16/32; for CFG batches 1/4/16 they are 10.27/9.75, 12.87/12.07 and
13.54/12.91. Each sample has 64 decoder output positions, including those
past EOS. Process-median ranges were 58.072–58.264 ms Rust versus
64.297–64.606 ms Python at base batch 1 and 1193.159–1207.746 ms Rust versus
1274.496–1278.137 ms Python at base batch 32; complete per-process data and
CUDA-event spans are in the measurement files. Python's deliberately timed
separate garbage collection had pooled medians of roughly 49–51 ms per batch;
it is excluded from the table and may not represent other Python deployments.
Events exclude final host readback and can include host-induced stream idle.

The batch-16 CFG cell was also run separately before the combined small-batch
matrix: Rust median/p95 1179.604/1181.432 ms versus Python
1238.261/1239.745 ms, with matching outputs. The small, batch-32 and separate
CFG runs had 644, 277 and 269 telemetry samples, zero query errors and sampled
temperature ranges of 47.41–72.03, 49.59–71.94 and 52.88–71.91 °C. Desktop
GPU processes were present throughout; sampled driver memory does not prove
allocator peaks or exclusivity. The normal backend remains unchanged; full
Phase 4, WMT14 Metal and other profile gates remain open.

Source-bound commands, with the corresponding checked binary SHA-256, were:

```sh
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/bench/capture_wmt14_fast_argmax.py --mode base --head 0e7fede1ab47254335f4558d7f3a77e91dea73e1 --binary-sha256 e0e7d5fb304c5afb32b8943727cb42f99fbbdf4a541e6997d3157be920e64a07 --protocol-sha256 26f5f13aeb440da2f95f56091a482e221fa0c311934494a334566d9bc5ff8de1 --rows 3003 --output reports/phase4-wmt14-fast-argmax-base-full-v1
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/bench/capture_wmt14_fast_argmax.py --mode cfg1 --head 0e7fede1ab47254335f4558d7f3a77e91dea73e1 --binary-sha256 451c7b4201ea74038eb8a11457862279e9de7873905bff3af485c26a0136fe0c --protocol-sha256 26f5f13aeb440da2f95f56091a482e221fa0c311934494a334566d9bc5ff8de1 --rows 3003 --output reports/phase4-wmt14-fast-argmax-cfg-full-v1
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/bench/run_wmt14_fast_argmax_bench.py --output reports/phase4-wmt14-fast-argmax-bench-small-v1 --repetitions 3 --batches 1 4 16 --modes 0 1
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/bench/run_wmt14_fast_argmax_bench.py --output reports/phase4-wmt14-fast-argmax-bench-b32-v1 --repetitions 3 --batches 32 --modes 0
```

The small-matrix manifest/measurements/telemetry SHA-256 values are
`506a16da5b23eeb32d68699116d648376231bdc380be3535ec5d7c09fd74a2c5` /
`ba624dd92f5a3a8eac34d4c6214598d8b98ccf955d568e0113dc5d6a0f2b26bf` /
`bccd82d2abe4e89ab383617e31bde09eab2a1d2e095f71ea56081575991a5571`.
The batch-32 values are
`9f10f16ee352f7aa6fac3fd955d67f350d54bfb4c2fe83a00b3a91163f298e90` /
`4e4229b5a73282f5dea21a51a2a0822bd445984f3a1ab2d8ce4e236e9fb6d018` /
`a8ce89d65ead5037e12691610dcc198f135fabe94be86a588ac9415ff6c7c5cd`.
The separate CFG batch-16 values are
`74c394872edcf277559845fce39d06a1269f9c7c1c1e91e569f7c9b24d9e26b6` /
`013064b8814af2d497d67cbd5a19a514e944ea411cc284035aef474cf697fc11` /
`4428863196151686396362580490c78cb5144b0de371b407a0ac1755043feb53`.

## Allocator observation preparation

A separate batch-32 one-process smoke produced matching Rust/Python codes and
IDs. The Rust default CUDA async pool reported live-output used/reserved high
water 2323704992/2751463424 bytes and, after dropping the output and
synchronizing, current used/reserved 1062175648/1140850688 bytes. The Python
PyTorch allocator reported live-output allocated/reserved high water
2475093504/3730833408 bytes and, after dropping outputs and explicit garbage
collection, current allocated/reserved 1128726016/3730833408 bytes. Both high
water counts include model loading; current counts retain the model. These are
different allocators with different coverage and cleanup policies, not a
direct comparison of total GPU usage. The earlier sampled process memory is
reported separately. One observed load from artifact read through synchronized
model construction took 1.587 s Rust and 1.197 s Python; the exact startup
boundaries and CUDA initialization coverage differ, so these are diagnostic
observations rather than a controlled startup performance claim.

The Rust/Python smoke reports hash to
`8ae9b843efd4704c9c3e657b7b470b3b42c2065da1b369539b58669682a61722` /
`928b88fa545bd27253c413566da99bb54498472d0fa9482b0b42d8dc21c5c962b`.
The standalone Rust example passed feature-specific Clippy/build and Python
syntax validation. A source-bound per-batch allocator observation across all
seven scoped cells is prepared but pending; no additional gate is claimed.

## Source-bound native allocator and load observations

At clean commit `0a2e288eb98a5295af718ad2f277568c2e1e87e2`, one fresh
process per runtime and cell ran three warmups and one observed generation.
All codes and IDs matched the corresponding previously qualified benchmark
output. The table reports MiB in the order **used or allocated peak / reserved
peak / used or allocated current / reserved current**. Rust current values
are after dropping the generation output and synchronizing; Python current
values are after dropping outputs and explicit garbage collection. Both keep
the model loaded. High-water counters include loading. The CUDA default async
pool does not account for all CUDA or library allocations; PyTorch's allocator
has different coverage. These figures must not be treated as interchangeable
total-device peaks or combined with sampled driver memory.

| CFG | Batch | Rust CUDA default pool MiB | Python PyTorch allocator MiB |
|---:|---:|:---|:---|
| 0 | 1 | 1050.6 / 1088.0 / 1013.0 / 1056.0 | 1126.2 / 1192.0 / 1076.4 / 1192.0 |
| 0 | 4 | 1163.4 / 1248.0 / 1013.0 / 1056.0 | 1244.7 / 1324.0 / 1076.4 / 1324.0 |
| 0 | 16 | 1614.5 / 1824.0 / 1013.0 / 1056.0 | 1724.3 / 2132.0 / 1076.4 / 2132.0 |
| 0 | 32 | 2216.1 / 2624.0 / 1013.0 / 1088.0 | 2360.4 / 3558.0 / 1076.4 / 3558.0 |
| 1 | 1 | 1050.9 / 1088.0 / 1013.0 / 1024.0 | 1128.3 / 1196.0 / 1076.4 / 1196.0 |
| 1 | 4 | 1164.9 / 1280.0 / 1013.0 / 1024.0 | 1253.7 / 1370.0 / 1076.4 / 1370.0 |
| 1 | 16 | 1620.6 / 2016.0 / 1013.0 / 1056.0 | 1757.2 / 2904.0 / 1076.4 / 2904.0 |

The one-process load observations ranged 1.494–1.521 s Rust and
1.191–1.229 s Python. Rust's interval includes artifact loading and backend
construction through synchronization; Python's includes checkpoint loading
and model construction through synchronization, but its CUDA initialization
can begin earlier. These are not equivalent startup boundaries or a repeated
startup comparison. The observer recorded 70 telemetry samples with zero query
errors; desktop GPU activity remained present. The independent measured
process RSS and sampled GPU memory remain in the timing table above.

```sh
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/bench/observe_wmt14_fast_argmax_memory.py --head 0a2e288eb98a5295af718ad2f277568c2e1e87e2 --binary-sha256 13291d4b00e6d1367078f68e0ed95405f2c70d37e0c4c6c1c3d42f0de0011c3b --output reports/phase4-wmt14-fast-argmax-memory-matrix-v1
```

Manifest, completion receipt and telemetry SHA-256 values are
`d3fa5383fc625c34df77ee5f0a5d674835026913cb0648948d16a68e99e1d631` /
`fc56425e9d5d51b0db7f3331b3dc9ec7f9ee44149a63b44818aaf5efb78dbfa8` /
`2a8f58e25f02fabfada09ccb72f33b8bcda7946bfd6c7b96eb53714cddeb2a26`.
The individual snapshots and their hashes are in the completion receipt.
This remains an opt-in CUDA candidate; WMT14 Metal and the Phase 4 gates
remain pending.
