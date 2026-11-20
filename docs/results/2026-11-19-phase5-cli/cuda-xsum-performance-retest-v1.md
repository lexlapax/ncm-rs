# XSum CUDA/Python repeated model performance comparison

2026-11-20. At commit `6a5c2cb1d4bdc7e6ff557dfbc2119dfc0d6bd68b`, the strict-cuDNN Rust model probe and pinned upstream Python ran on an NVIDIA RTX 3060 Ti with top-1, CFG zero and the first 32 pinned corpus prefixes. Three fresh processes per runtime and batch alternated order; each had five warmups and 30 synchronized trials. Generation plus code/ID host readback was timed, excluding model load and Python's separately measured garbage collection. Whole-process RSS includes load. The collector checked every reported code and ID across runtimes and repetitions for each batch; this is not a full-corpus quality result. See the [separate quality retest](cuda-xsum-quality-retest-v1.md) for the known intermediate-code tie outside these prefixes.

| Batch | Runtime | Median of process medians (ms) | Pooled p95 (ms) | Median peak RSS (MiB) | Median sampled GPU process peak (MiB) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Rust | 153.678 | 154.770 | 994.258 | 838 |
| 1 | Python | 189.219 | 190.814 | 1779.027 | 990 |
| 4 | Rust | 556.275 | 558.430 | 994.336 | 1062 |
| 4 | Python | 689.640 | 691.108 | 1779.508 | 1624 |
| 32 | Rust | 4322.778 | 4330.690 | 991.742 | 3016 |
| 32 | Python | 5327.429 | 5339.055 | 1783.160 | 6214 |

Rust had lower observed median latency and process RSS in these cells. The [timing summary](xsum-cuda-perf-summary-v1.json) includes per-process medians, RSS and allocator counters (SHA-256 `1223fba4ca6ccff50c20f1243d293be10652f75d297f8dc01de8528bc7229725`); the [driver memory summary](xsum-cuda-perf-memory-v1.json) includes per-process sample counts (SHA-256 `2cf7a7ae69a2fc5d6e93a8584a7b6da52d7eaf3004c23b70f488bbc6dc8cc88a`). Allocator counters have different scopes between runtimes. All 1,198 GPU activity samples had successful queries; four other GPU processes were observed throughout, so no exclusive reservation was established. Approximately 1 Hz sampled driver peaks can miss brief peaks and are not native allocation peaks.

The raw bundle remains ignored at `reports/phase5-xsum-cuda-perf-v1/`, not durably published. Its manifest SHA-256 is `4395af5fcf84d629ff3a5eb7e785b8ebc4aa93e819d5583340471775d6ce61fe`; the Rust binary SHA-256 is `a74f8ef3306704f380b7e5849109058fd08b7b9c1f2ed14024c56e6f160e2fa3`. The collector checked sources, inputs, artifact manifest and linked libraries before and after collection.

Reproduce after preparing the pinned XSum artifact, upstream source and corpus documented in the Phase 4 evidence. The corpus file must hash to `cba412c100c1264a7a107b2df3bacd20bff5846325ce48fd5aa353b4726daa6a`; derive the first 32 prefixes as in the [capacity preparation](../2026-11-15-phase4-xsum/cuda-capacity-preparation-v1.md), using compact JSON plus newline. The prefix file must hash to `a9c0da895804e5146191993ae27db4c921d1b08b21594cb6eaa63777748019cf`.

```sh
setpriv --inh-caps=-all env LIBRARY_PATH="$PWD/.reference/phase4-cudnn-probe-lib" \
  cargo build --release --locked -p ncm-backend-cuda --features cuda-cudnn --example xsum_perf_screen
setpriv --inh-caps=-all .reference/venv/bin/python tools/bench/run_phase4_xsum_bench.py \
  --output reports/NEW-DIRECTORY --profile v2 --batches 1 4 32 --repetitions 3 \
  --cudnn-dir .reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib
setpriv --inh-caps=-all python3 tools/bench/summarize_phase4_xsum_b32.py \
  --directory reports/NEW-DIRECTORY --batches 1 4 32
setpriv --inh-caps=-all python3 tools/bench/summarize_xsum_gpu_activity.py --directory reports/NEW-DIRECTORY
```

This model probe does not invoke the changed CLI, measure cold load, prove GPU isolation or compare previous Rust with candidate. It does not satisfy the Phase 5 release performance gate or provide Metal evidence.

## Sanitized timing trials

The [sanitized trial series](xsum-cuda-perf-trials-v1.json), SHA-256 `049da4c0f9da2b05468b05a3823b9275dbdd0e2683e0def12853d703d84293c1`, publishes 540 measured timings (three batches, two runtimes, three processes, 30 trials) without source prefixes, generated IDs, local paths or telemetry. Exporter replay was byte-identical; independent published-only replay reproduced per-process medians and pooled p95s. The manifest and summary identities are pinned, but individual raw timing files were not precommitted: aggregate agreement is a consistency check, not authentication of every trial. Source file hashes were recorded at export time. Negative tests and `setpriv --inh-caps=-all tools/check.sh` passed. Raw data remain local; shared GPU, omitted device/host memory and release-gate limitations remain unchanged.

```sh
setpriv --inh-caps=-all python3 tools/bench/export_phase5_text_model_trials.py --profile xsum \
  --run reports/phase5-xsum-cuda-perf-v1 --summary docs/results/2026-11-19-phase5-cli/xsum-cuda-perf-summary-v1.json \
  --output reports/NEW-xsum-timing-trials.json
```
