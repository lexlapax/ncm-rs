# WMT14 CUDA synchronized timing comparison

The comparison on `52f3374` completed on the RTX 3060 Ti (8 GiB), driver
620.70.06, with the [pinned plan](cuda-performance-plan-v1.md). Each cell has
three alternating fresh processes, five warmups and 30 measured batches per
process. All processes and all trials matched Python's codes and raw IDs at
each mode/batch. These are the first 1, 4 or 16 fixed prefixes, not an
independent quality evaluation. CFG=0 is the released setting; CFG=1 is an
explicit diagnostic. No tested workload ran out of memory.

Times are milliseconds per batch, synchronized wall-clock including input
upload, generation, decoding and output readback; model loading and Python's
explicit garbage collection are excluded. Median and nearest-rank p95 pool
90 trials; the process-median range shows between-process dispersion. GPU
memory is the largest approximately one-second sampled process allocation,
not an allocator peak. Host memory is the largest process peak RSS.

| CFG | Batch | Runtime | Median / p95 ms | Process medians range ms | Host peak MiB | Sampled GPU peak MiB |
|---:|---:|:---|---:|---:|---:|---:|
| 0 | 1 | Python | 64.09 / 64.44 | 64.07–64.12 | 2093 | 1360 |
| 0 | 1 | Rust | 59.78 / 60.46 | 59.62–59.88 | 1850 | 1222 |
| 0 | 4 | Python | 178.46 / 179.38 | 178.26–178.84 | 2093 | 1492 |
| 0 | 4 | Rust | 175.60 / 176.51 | 175.04–176.08 | 1843 | 1574 |
| 0 | 16 | Python | 652.17 / 654.08 | 651.52–653.42 | 2093 | 2300 |
| 0 | 16 | Rust | 661.71 / 663.57 | 656.51–662.31 | 1839 | 2566 |
| 1 | 1 | Python | 102.22 / 103.12 | 101.78–102.76 | 2093 | 1364 |
| 1 | 1 | Rust | 98.80 / 99.53 | 98.69–99.20 | 1844 | 1222 |
| 1 | 4 | Python | 331.40 / 332.36 | 330.41–331.62 | 2094 | 1538 |
| 1 | 4 | Rust | 322.45 / 323.60 | 321.80–322.86 | 1849 | 1544 |
| 1 | 16 | Python | 1238.95 / 1240.75 | 1236.83–1240.03 | 2093 | 3072 |
| 1 | 16 | Rust | 1223.69 / 1225.43 | 1223.10–1224.07 | 1875 | 2600 |

The corresponding pooled median CUDA-event spans (Python/Rust, ms) were:
CFG=0, batches 1/4/16: 64.04/59.71, 178.42/172.12, 652.11/643.56;
CFG=1: 102.18/98.75, 331.33/318.02, 1238.85/1205.54. Events bracket
upload through greedy selection and exclude final code/ID readback. They may
include stream idle caused by intermediate host work and are not kernel sums.
Python's explicit post-batch garbage collection plus synchronization, excluded
from the headline, raises its pooled median to 114.16/229.01/703.21 ms for
CFG=0 and 152.39/382.08/1290.32 ms for CFG=1. This deliberate collection
policy should not be assumed to represent every Python application.

The largest PyTorch allocated/reserved peaks were respectively 1126/1192,
1245/1324 and 1724/2132 MiB for CFG=0 and 1128/1196, 1254/1370 and
1757/2904 MiB for CFG=1. Rust reports sampled driver process memory, not
comparable allocator counters. The roughly one-second telemetry produced 657
samples with no query errors; four desktop GPU processes persisted throughout,
so the run was not GPU-exclusive. Sampled GPU utilization ranged 0–100%,
temperature 44.12–72.53 °C. Clock policy was not changed. The device-only
spans above are not pure kernel time; stage breakdowns, maximum capacity,
steady-state allocator accounting, actual non-special token throughput and
strictly exclusive-device measurements remain pending. Thus this report alone
does not satisfy every reporting requirement or establish the WMT CUDA gate.

The source-bound run command was:

```sh
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/bench/run_phase4_wmt14_bench.py --output reports/phase4-wmt14-perf-full-v1 --repetitions 3 --batches 1 4 16 --modes 0 1
```

The manifest SHA-256 is
`a05c7cf076e78a8d686e55953e118093aa47116389f183be6403460dc340ba2b`,
measurements SHA-256 is
`5a59cfdf7ef0ea4085717d2beda86147db187a54f433230104ce28d23b8c0614`,
and telemetry SHA-256 is
`e156af6d93d7827e06e50bb9feee31ea532cb6b4187a5258f65199e272b1756d`.
Full raw records remain under ignored `reports/`. The separate original CFG
failure, candidate retest limitation and unresolved XSum Metal raw-ID failure
remain documented; these timings do not waive them.
