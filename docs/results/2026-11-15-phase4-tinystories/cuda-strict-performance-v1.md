# TinyStories strict-cuDNN CUDA performance

Three alternating fresh Python and Rust processes per batch each ran five warmups and 30 synchronized generation trials on an RTX 3060 Ti. Timing covers generation and host readback of codes and final IDs, excluding model load. Python uses the pinned upstream generator and codec, default PyTorch SDPA and disabled TF32; Rust uses the opt-in strict-cuDNN backend. Rust's SplitMix64/inverse-CDF and Python's multinomial use different random streams, so samples differ. The separately scored [quality retest](heldout-quality-v1.md) uses the same previously frozen limits but is not a fresh blinded holdout.

| Batch | Python median ms | Rust median ms | Python p95 ms | Rust p95 ms | Python peak RSS MiB | Rust peak RSS MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 47.284 | 41.005 | 48.005 | 41.806 | 2335.6 | 971.6 |
| 4 | 101.085 | 96.818 | 102.033 | 97.706 | 2318.8 | 971.9 |
| 32 | 628.076 | 604.712 | 633.375 | 607.715 | 2331.9 | 971.7 |
| 128 | 2450.117 | 2358.878 | 2454.004 | 2394.529 | 2280.1 | 966.2 |

The median is the median of three process medians, and p95 is the nearest rank over the 90 pooled trials. RSS is the median GNU `time` process high-water mark including model load. Rust's median latency is 3.7–13.3% lower and its process peak RSS is approximately 58% lower in these cells. The 3.7% margins at batches 32 and 128 should be interpreted with the activity caveat below. This establishes the measured timing/RSS direction only, not complete TinyStories acceptance or Apple performance.

Python's live GPU allocations grew during repeated batch-128 trials and caused an OOM, including attempts with alternative allocator and cache settings. Explicit `gc.collect()` after every request kept allocations stable; it was timed separately and excluded from the generation times above. Its process medians were approximately 51–52 ms per request. This is a conservative compute-only comparison, not a claim that every application must collect every request. Process RSS includes startup and cleanup. The failed attempts and allocation diagnostics are retained alongside the [earlier default-CUDA comparison](cuda-performance-v1.md).

GPU-process memory was sampled approximately once per second. The Python/Rust sampled peak ranges across three processes were 844/678 MiB (batch 1), 928/712 MiB (batch 4), 2386/1192 MiB (batch 32), and 5532–5964/2760 MiB (batch 128). These process-level values can miss transient peaks and include context and allocator effects. They are not exact native allocation/retained-cache measurements and cannot establish that separate device-memory gate. Chrome, 1Password, walker and t3code appeared in all 712 activity samples; observed GPU clocks ranged from 210 MHz at idle to 1965 MHz. No clock policy was changed. Sampling and desktop activity limit attribution, particularly for the small timing differences.

The source-bound manifest, executable and native cuDNN library hashes are in ignored `reports/phase4-tinystories-linux-bench-strict-v1/manifest.json` (SHA-256 `001dba5884923cf9cb8777d0a596fae280c7c8d16e44d4bb5f7e3e0410bf0ede`). The independently generated `summary.json` has SHA-256 `087151d1115e30e596df18a165f8f06b8662ce629667e20d865a064b1ac6debb`; raw timings, process events and activity are retained alongside it outside Git. With the pinned local artifacts and release binary built as described in the [numerical report](cuda-numerical-investigation-v1.md), reproduce with:

```sh
PYTHONUNBUFFERED=1 .reference/venv/bin/python -B tools/bench/run_phase4_tinystories_bench.py \
  --output reports/NEW-RUN --repetitions 3 \
  --cudnn-dir .reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib
.reference/venv/bin/python -B tools/bench/summarize_phase4_tinystories.py reports/NEW-RUN
```

## Native allocator observations

A separate short observation used one warmup and one measured request per fresh process and synchronized before reading Rust's default CUDA async pool. Three Rust processes per batch produced identical rounded pool values below; the peak includes model loading and both requests, and the current values are after request outputs were dropped. The process also retains the loaded model. The corresponding PyTorch values come from one fresh process per batch after two requests and explicit garbage collection, with peak statistics reset after model loading. They use the same generation parameters. In addition, the three-process timing run independently recorded the same Python allocated/reserved peak values for each batch. All values are MiB.

| Batch | Rust pool used peak / current | Rust pool reserved peak / current | PyTorch allocated peak / current | PyTorch reserved peak / current |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 492.9 / 482.6 | 512 / 512 | 648.9 / 629.5 | 676 / 676 |
| 4 | 524.0 / 482.6 | 544 / 512 | 700.9 / 629.5 | 760 / 760 |
| 32 | 814.0 / 482.6 | 1024 / 512 | 1183.6 / 629.5 | 2218 / 2218 |
| 128 | 1808.3 / 482.6 | 2592 / 576 | 2836.9 / 629.5 | 5796 / 4054 |

These counters are exact within their own allocator's scope but are **not equivalent measures**: the CUDA pool excludes context, cuDNN and other allocations outside the pool, while PyTorch's allocator reports only its own managed blocks. The sampled process memory above includes broader process allocations but can miss peaks. Their direction supports the observed memory improvement without proving equal-coverage instantaneous device peaks. One Python retained observation per batch is not a three-process retained-memory study. The pool observation source is `crates/ncm-backend-cuda/examples/tinystories_memory.rs` (SHA-256 `2bcdfbccc17f9f76416a96999e69e7cbbd59d4bdc6482dcc55a2d36e85e1b43e`); the Python observation source is `tools/reference/tinystories_memory.py` (SHA-256 `8c8486e3496a003eb9c74b620a4b3d9c132a1e2404ba854c13b0139ae8e5f89b`). Raw outputs are under ignored `reports/phase4-tinystories-{strict-pool,python-memory}-v1/`.
