# TinyStories CUDA/Python repeated performance comparison

2026-11-20. The existing synchronized TinyStories benchmark was rerun at commit
`d294019741799fac18f26b6314b50129bcb4e173` on an NVIDIA RTX 3060 Ti, CUDA 13.4 and driver
620.70.06. The strict-cuDNN Rust model probe and pinned upstream Python ran three alternating fresh
processes per runtime and batch, five warmups and 30 trials each. Generation plus code/ID host
readback was timed; model load and Python post-trial garbage collection were outside the latency
scope. Whole-process peak RSS includes model loading. Rust SplitMix64 and Python multinomial use
different random streams; this is a fixed-workload comparison, not sample-by-sample equality.
The [separate nonblind quality retest](cuda-tinystories-quality-retest-v1.md) covers the stochastic
quality criterion.

| Batch | Runtime | Median of process medians (ms) | Pooled p95 (ms) | Median peak RSS (MiB) | Sampled GPU process peaks (MiB) |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Python | 46.497 | 47.022 | 2187.184 | 844 / 844 / 844 |
| 1 | Rust | 40.949 | 41.210 | 969.621 | 678 / 678 / 678 |
| 4 | Python | 100.765 | 101.170 | 2173.539 | 928 / 928 / 928 |
| 4 | Rust | 96.413 | 96.942 | 975.176 | 712 / 712 / 712 |
| 32 | Python | 625.794 | 629.233 | 2203.301 | 2386 / 2386 / 2386 |
| 32 | Rust | 602.691 | 604.304 | 974.961 | 1032 / 1032 / 1032 |
| 128 | Python | 2439.729 | 2442.106 | 2160.156 | 6180 / 5532 / 6180 |
| 128 | Rust | 2346.815 | 2349.435 | 975.164 | 2120 / 2120 / 2120 |

The [compact summary](tinystories-cuda-perf-summary-v1.json) includes per-process medians,
RSS and sampled peaks; it is SHA-256 `0c0b360b4fb4129955762a0351c4d15f71874ec29c39e646c88457bbaca35ac3`.
The run manifest SHA-256 is `3aeb0f7d1ae9002fc15c959dd50417b14834629399715a4e6d77688ce09a375d`;
the telemetry JSONL is `3d4d0f1e8d86c7294576338827bafd4f785e8f031f0df7942615328a8d0e3b4a` and
process events are `208b59ac2431dd408d2c77447750d4c60dc8045707a20bcb1f1ae33e0fd9f86a`.
All 707 telemetry samples had successful queries, but desktop GPU processes were present in every
sample; no exclusive GPU reservation was established. Sampled driver peaks are approximately
1 Hz lower bounds, **not native allocation peaks**, and Python and Rust allocator scopes are not
interchangeable. These margins, especially 3–4% at larger batches, are observations under shared
activity, not a validated non-regression threshold.

Reproduce with a fresh trusted output path after building the strict candidate:

```sh
setpriv --inh-caps=-all env NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 \
  LIBRARY_PATH="$PWD/.reference/phase4-cudnn-probe-lib" cargo build --release --locked \
  -p ncm-backend-cuda --features cuda-cudnn --example tinystories_sample_bench
setpriv --inh-caps=-all env -u LD_LIBRARY_PATH -u PYTHONPATH OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  .reference/venv/bin/python tools/bench/run_phase4_tinystories_bench.py --output reports/NEW-DIRECTORY \
  --repetitions 3 --cudnn-dir .reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib
setpriv --inh-caps=-all python3 tools/bench/summarize_phase4_tinystories.py reports/NEW-DIRECTORY
```

The Rust probe SHA-256 was `5778b05b13dc6dab6e1f5f7496d69d149702242f7377d1736913383e76b3c9f5`;
the collector checked source, artifact and native cuDNN hashes before and after. Raw trial and
activity captures remain ignored under `reports/phase5-tinystories-cuda-perf-v1/`, not durably
published. The probe does **not** invoke the changed CLI, measure cold load, report all required
native peaks, establish an isolated GPU or provide previous-Rust-vs-candidate evidence. This is a
partial current Python/Rust model comparison, not the Phase 5 performance gate or Metal evidence.

## Sanitized per-trial publication

The [sanitized trial series](tinystories-cuda-perf-trials-v1.json) publishes all 720 measured generation timings (four batches, two runtimes, three processes, 30 trials), with per-source-file hashes but no inputs, generated outputs, local paths or GPU-process telemetry. SHA-256 `5f02b50af7ed8114cefa77e581cc0ea3e97d576946c5daf4894349edbab37a6c`. The exporter checked the original manifest and published aggregate hashes, checked each process median against that aggregate, and rechecked input files after export. An independent replay of the published series reproduced all process medians and pooled p95s; a second export was byte-identical. Two negative-control tests and `setpriv --inh-caps=-all tools/check.sh` passed.

The original manifest does not pin individual timing-result files: matching aggregate medians and p95s is a consistency check, not independent authentication of every trial. Source hashes identify the files read during this export, not a precommitted record of them. The series does not include warmups, process RSS, device peaks, output identities or full telemetry; raw bundles remain local. Shared-GPU and scope limitations above are unchanged; no release gate is implied.

```sh
setpriv --inh-caps=-all python3 tools/bench/export_phase5_tinystories_trials.py \
  --run reports/phase5-tinystories-cuda-perf-v1 \
  --summary docs/results/2026-11-19-phase5-cli/tinystories-cuda-perf-summary-v1.json \
  --output reports/NEW-tinystories-trials.json
```
