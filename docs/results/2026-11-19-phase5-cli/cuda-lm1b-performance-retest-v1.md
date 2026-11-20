# LM1B CUDA/Python repeated model performance comparison

2026-11-20. At commit `415fdef`, the existing LM1B model probe and pinned upstream Python were rerun on an NVIDIA RTX 3060 Ti. The Rust probe used reference rotary tables, cuDNN graph and efficient attention; both implementations used FP32 and development seed 11 with **independent random streams**. For each size/batch cell, three fresh processes per runtime alternated order; each ran five warmups and 30 synchronized measured generations. Rust and Python output signatures repeated within each runtime, not across runtimes. The separate 80-seed nonblind quality retests for [small](cuda-lm1b-small-quality-retest-v1.md), [medium](cuda-lm1b-medium-quality-retest-v1.md) and [large](cuda-lm1b-large-quality-retest-v1.md) remain the quality evidence.

Timing covers initial sampling, generation, codec decode, final penalties and code/ID host readback; model load is excluded. Median and p95 pool 90 trials. RSS is median whole-process VmHWM including load. Model-load intervals exclude process startup and are implementation-specific: Rust starts at artifact loading and includes backend/rotary setup; Python starts at checkpoint loading after imports/configuration. They are not CLI cold-load timings or identical boundaries.

| Size | Batch | Rust median / p95 ms | Python median / p95 ms | Rust / Python load s | Rust / Python peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small | 1 | 41.938 / 42.294 | 61.796 / 62.973 | 1.257 / 1.154 | 1536.4 / 2551.1 |
| Small | 4 | 105.504 / 105.974 | 134.003 / 134.652 | 1.251 / 1.154 | 1537.5 / 2544.7 |
| Small | 16 | 357.637 / 359.243 | 430.957 / 432.051 | 1.249 / 1.181 | 1536.0 / 2517.3 |
| Medium | 1 | 118.671 / 119.029 | 141.870 / 142.551 | 2.341 / 2.146 | 2374.2 / 3843.8 |
| Medium | 4 | 302.381 / 303.142 | 332.724 / 333.678 | 2.349 / 2.138 | 2375.2 / 3845.7 |
| Medium | 16 | 1047.409 / 1049.777 | 1097.795 / 1100.058 | 2.354 / 2.135 | 2362.9 / 3843.9 |
| Large | 1 | 247.502 / 248.511 | 276.600 / 277.522 | 4.315 / 3.925 | 3931.1 / 6961.8 |
| Large | 4 | 638.135 / 639.920 | 681.807 / 684.060 | 4.318 / 3.922 | 3932.1 / 6960.0 |
| Large | 8 | 1167.049 / 1169.494 | 1225.893 / 1228.164 | 4.312 / 3.936 | 3933.5 / 6961.7 |

Rust had lower observed generation median and RSS in these cells; its separately scoped model-load interval was longer. Large batch 16 was not rerun: the prior pinned Python attempt OOMed, as recorded in the [Phase 4 report](../2026-11-16-phase4-lm1b/medium-large-cuda-timing-v1.md); batch 8 is the largest common batch in this matrix, not a proven physical maximum.

The [compact aggregate](lm1b-cuda-perf-summary-v1.json) (SHA-256 `257ecb87e33299188270110574f666a7d363f58dc4b5cb8a175e803902f21545`) includes per-process latency, load, RSS, native allocator counters and sampled driver memory. Rust CUDA async-pool and PyTorch caching-allocator counters have different scopes and are not total device peaks. Across the five runs, 1,092 GPU telemetry samples succeeded; four other GPU processes were observed throughout, so GPU exclusivity was not established. Roughly one-second driver samples can miss peaks. The raw bundles remain ignored locally under `reports/phase5-lm1b-*-cuda-perf-v1/`, not durably published; input hashes are in the aggregate. Probe binary SHA-256 is `bba40b724b20bfff983f879e50568cac065ecfff7b51b83022ecf504ce09b1ca`. Aggregate replayed byte-for-byte; offline negative controls and `setpriv --inh-caps=-all tools/check.sh` passed.

Reproduce after preparing the pinned artifacts, Python upstream and rotary reference using fresh output directories:

```sh
setpriv --inh-caps=-all env CUDA_COMPUTE_CAP=86 LIBRARY_PATH="$PWD/.reference/phase4-cudnn-probe-lib" \
  cargo build --locked --release -p ncm-backend-cuda --features cuda-cudnn,lm1b-experimental --example lm1b_small_capacity_probe
for size in small medium; do
  setpriv --inh-caps=-all env -u LD_LIBRARY_PATH OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 \
    .reference/venv/bin/python tools/bench/run_lm1b_small_capacity_screen.py --size "$size" --repeated \
    --reference-rope-diagnostic reports/phase4-lm1b-level7-replay-v1/reference-rope-tables.safetensors \
    --rust-cudnn-graph-efficient --output "reports/NEW-$size"
done
for batch in 1 4 8; do
  setpriv --inh-caps=-all env -u LD_LIBRARY_PATH OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 \
    .reference/venv/bin/python tools/bench/run_lm1b_small_capacity_screen.py --size large --batch "$batch" --repeated \
    --reference-rope-diagnostic reports/phase4-lm1b-level7-replay-v1/reference-rope-tables.safetensors \
    --rust-cudnn-graph-efficient --output "reports/NEW-large-b$batch"
done
setpriv --inh-caps=-all python3 tools/bench/summarize_phase5_lm1b.py \
  reports/NEW-small reports/NEW-medium reports/NEW-large-b{1,4,8}
```

These are model probes, not changed-CLI cold-load/end-to-end, previous-Rust-vs-candidate, isolated-GPU or Metal evidence. They do not complete the Phase 5 release performance gate.
