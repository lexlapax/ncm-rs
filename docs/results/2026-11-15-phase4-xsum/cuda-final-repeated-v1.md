# XSum CUDA repeated performance after stochastic support

The pinned official CFG-zero, temperature-1, top-k-1, top-p-0 configuration
was remeasured on the current generator source. Each batch has three fresh
processes per runtime, five warmups and 30 synchronized measured generations
per process, with alternating runtime order. Every checked code and final ID
matched within each paired batch and across processes.

| Batch | Runtime | Median ms | Pooled p95 ms | Peak RSS MiB | Sampled driver GPU peak MiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Rust | 156.05 | 156.96 | 1060.44 | 838 |
| 1 | Python | 189.66 | 190.51 | 1861.05 | 990 |
| 4 | Rust | 566.26 | 568.95 | 1062.68 | 1062 |
| 4 | Python | 689.88 | 690.88 | 1860.40 | 1624 |
| 32 | Rust | 4404.64 | 4411.38 | 1059.95 | 3080 |
| 32 | Python | 5336.48 | 5358.92 | 1864.29 | 5672 |

Latency is the median of process medians; p95 pools 90 trials. RSS is the
median of `/usr/bin/time` whole-process peaks, including load; generation
timing excludes load. Python's post-trial garbage collection and synchronization
are timed separately and excluded from the table. Driver GPU memory is the
median of each process's **sampled** peak, attributed through the collector's
process-parent records; sampling approximately once a second can miss brief
peaks. It reports NVIDIA driver per-process memory for both runtimes, not a
comparison between the differently scoped CUDA pool and PyTorch allocator
counters. The raw timing summary retains those allocator counters separately.

The collector recorded 1,208 activity samples and no failed memory queries;
the driver memory summary had 6–9, 20–26 and 151–183 samples per process for
batches 1, 4 and 32 respectively. Desktop GPU processes were present, so the
run was not isolated. The first GPU-memory summary attempt rejected the
unit-free `nvidia-smi` format; the corrected parser and its attribution test
pass. Rust has lower measured latency, RSS and sampled driver GPU memory in
these cells. The separate [stochastic](cuda-stochastic-investigation-v1.md)
and [tie](cuda-tie-order-screen-v1.md) limitations remain; this is not the
complete XSum or MLX gate.

The raw bundle is `reports/phase4-xsum-final-repeated-v1/`; manifest SHA-256
`7d9c782658277c8015cd6f0ad8037fd19dd971e76a3d1bc95b97904c1e3ef036`. The
timing summary hashes to
`54422fdca6a3db8cfff2529746c668c66d54420243ab97e41ff3ad6ee840f04a`, and
`reports/phase4-xsum-final-repeated-v1-gpu-memory.json` hashes to
`f3537f5bc8f185cbb25a491e78cbe327a7ed0b03660429bebd70eea690a4d30d`.
Source, benchmark binary, artifact and linked CUDA library hashes were checked
before and after collection.

Use the [prior report's collection command](cuda-repeated-v2.md) with a new
output directory, then run `tools/bench/summarize_phase4_xsum_b32.py` and
`tools/bench/summarize_xsum_gpu_activity.py --directory NEW_REPORT`. The latter
requires successful, complete process-event records and at least one attributed
driver sample per process; its CPU attribution regression test passed.
