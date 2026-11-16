# XSum CUDA batches 1 and 4 with Python cleanup

Following the [batch-32 comparison](cuda-b32-gc-comparison-v1.md), the same
source commit, model, device, synchronized generation/readback scope and
cleanup-corrected Python reference were measured at batches 1 and 4. Each cell
uses three alternating fresh processes, five warmups and 30 measured batches
per process. Runtime results use the median of process medians; p95 pools the
90 trials. Whole-process peak RSS is the median of three process peaks.

| Batch | Rust median | Python median | Rust p95 | Python p95 | Rust RSS | Python RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 160.49 ms | 190.45 ms | 161.35 ms | 192.04 ms | 1,062.34 MiB | 1,860.45 MiB |
| 4 | 580.50 ms | 691.66 ms | 582.15 ms | 692.81 ms | 1,063.93 MiB | 1,860.61 MiB |

Rust is about 15.7% and 16.1% faster by these median latencies, with about 43%
lower median RSS at each size. All six processes per size returned identical
codes and raw IDs. Python's median cleanup was 50.20 and 50.27 ms respectively;
the pooled median including cleanup was 240.84 and 741.72 ms. Its cleanup is
reported separately because the generation-only rows time equivalent model
work; the inclusive rows show the practical cost of reclaiming the accumulated
GPU-backed objects. The earlier no-cleanup allocator peaks remain provisional.

Rust CUDA default pool high-water used/reserved was 632.73/672 MiB at batch 1
and 785.61/896 MiB at batch 4; Python allocator peak allocated/reserved was
748.21/822 MiB and 904.95/1,456 MiB respectively. Their scopes differ and do
not establish equivalent total device memory. Rust pool end used/reserved was
583.37/608 MiB for both sizes. Allocation high-water marks include model load.

All 201 activity samples returned both GPU queries. Four other GPU processes
collectively reported 307 MiB throughout; global utilization ranged 0–100%,
temperature 54–71 °C, and SM clock 210–1,950 MHz. These global samples do not
attribute activity, so interference cannot be ruled out. Clock policy was
unchanged. The earlier batch-32 OOM and all other limitations in the linked
report remain visible. These measurements do not establish the full XSum Linux
gate: quality, full selection semantics, maximum-capacity behavior and broader
reporting are pending; XSum MLX is also pending.

The ignored raw bundle is `reports/phase4-xsum-cuda-performance-b1-b4-gc-v1/`.
The manifest SHA-256 is `7533007940e8e71fa8080a49cf4cdeb240715de0f81f6ac16e77e26bf4323126`,
activity SHA-256 `ef9e4b2321e78c073d7d3ffb9c8386f74bfad623d3e940eef4b87d2f5c8455f9`,
and process-events SHA-256 `03e66cf46e9a3041acc494a999f19cf85a077926979c8cf0793eeb1a52c3fb3b`.
Reproduce with a new output directory; the collector refuses an existing one:

```sh
.reference/venv/bin/python tools/bench/run_phase4_xsum_bench.py --output reports/phase4-xsum-cuda-performance-b1-b4-gc-v1 --repetitions 3 --batches 1 4 --cudnn-dir .reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib > reports/phase4-xsum-cuda-performance-b1-b4-gc-v1.log 2>&1
.reference/venv/bin/python tools/bench/summarize_phase4_xsum_b32.py --directory reports/phase4-xsum-cuda-performance-b1-b4-gc-v1 --batches 1 4
```
