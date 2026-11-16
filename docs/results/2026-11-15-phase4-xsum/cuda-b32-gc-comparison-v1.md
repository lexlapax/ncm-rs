# XSum CUDA batch-32 comparison with Python cleanup

At source commit `b1d07b9800e20231ead95d82f9f2c208996c199f`, the restricted
temperature-1, top-k-1, top-p-0, CFG-0 generation path was measured on an RTX
3060 Ti, using strict cuDNN for Rust and the pinned original Python model on
the same inputs. Each runtime used three alternating fresh processes with five
warmups and 30 measured batches per process. Timing covers generation, the
FP64 final selector and synchronized code and ID readback, excluding model load.
The Python reference explicitly collects unreachable batch objects after each
batch; the separately timed collection and synchronization are excluded from
the generation-only row and included in the additional row below. See the
[OOM investigation](b32-python-memory-v1.md) for why this reference harness
change was necessary. No upstream model math was changed.

| Runtime | Median of process medians | Pooled p95 | Median process peak RSS |
| --- | ---: | ---: | ---: |
| Rust generation | 4,500.39 ms | 4,508.48 ms | 1,064.03 MiB |
| Python generation | 5,336.60 ms | 5,343.38 ms | 1,865.16 MiB |
| Python generation plus cleanup | 5,387.03 ms pooled median | 5,394.50 ms | same process |

The Rust median is 15.7% lower, and its median whole-process peak RSS is 43.0%
lower. Rust's individual process medians were 4,466.54, 4,502.70 and
4,500.39 ms; Python's were 5,336.60, 5,337.18 and 5,335.06 ms. Median Python
cleanup alone was 50.55 ms. The median-of-medians row and pooled p95 are
different aggregation methods as labeled; the cleanup-inclusive median is
pooled across the 90 paired generation-and-cleanup samples. At the
median-of-medians, throughput is approximately 7.11 and 6.00 samples/s.
All six processes returned the same 6,400 codes and 1,536 raw output IDs.

The CUDA default asynchronous pool reported Rust high-water used/reserved of
2,201.24/2,912 MiB and end used/reserved of 583.37/672 MiB in each process.
Python's PyTorch allocator reported peak allocated/reserved of 2,367.25/5,504
MiB. These counters cover different allocations and must not be interpreted as
equivalent total device memory. High-water figures include model loading.

The 1,018 approximately one-second activity samples returned both GPU queries
successfully. Four other GPU processes collectively reported 307 MiB in all
samples; global GPU utilization ranged from 0–100%, temperature from 37–73 °C,
and SM clock from 210–1,950 MHz. The counters do not attribute GPU activity to
individual processes, so interference cannot be ruled out. The clock policy was
unchanged. The later [batch-1 and batch-4 repetition](cuda-small-batch-gc-comparison-v1.md)
uses the same cleanup-corrected reference. These are scoped results, not the
full XSum Linux gate: real-device quality, full selection semantics,
maximum-capacity behavior and broader reporting remain open. MLX has not been
tested for XSum.

The raw ignored bundle is `reports/phase4-xsum-cuda-performance-b32-v2/`.
Its manifest SHA-256 is `7375c3de8d4fef623b53d22d28de675f1bf51aa1efedc69221b46d4c0c31fee0`,
activity SHA-256 is `1c8b0bda91888e269cb65be75b2b4c033a39c0f20bfd931b43dd38c4c53eaf8c`,
and process-events SHA-256 is `25e1b890e6f69679a15dd9ea01806bf5e9b05d2962629f2572bef1e79394b0a7`.
The retained first attempt and its OOM are not replaced by this run. Reproduce
the collection and compact summary from the repository root with:

```sh
.reference/venv/bin/python tools/bench/run_phase4_xsum_bench.py --output reports/phase4-xsum-cuda-performance-b32-v2 --repetitions 3 --batches 32 --cudnn-dir .reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib > reports/phase4-xsum-cuda-performance-b32-v2.log 2>&1
.reference/venv/bin/python tools/bench/summarize_phase4_xsum_b32.py
```

The collector refuses an existing output directory; choose a new directory for
a fresh measurement and pass that directory to the summary script with
`--directory`.
