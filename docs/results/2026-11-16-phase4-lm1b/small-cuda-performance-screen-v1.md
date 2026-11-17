# LM1B-small CUDA development performance screen

The existing explicit device sampler was compared with pinned upstream Python
on non-held-out seed 11 at batch eight. Both probes timed initial-code sampling,
generation, codec decode, final penalties and code/ID host readback, excluding
model loading. Each fresh process used two warmups and eight synchronized
trials; three processes per runtime ran in alternating order. Every Python
trial matched its previously recorded Python development capture exactly, and
every Rust trial matched its own retained Rust capture exactly. The runtimes
use independent random streams; exact Python/Rust outputs are not asserted.

The first screen used the CPU FP64 final-penalty path after parallelizing its
independent rows. Pooled median/p95 were 0.228985/0.229470 seconds for Python
and 0.249707/0.255184 seconds for Rust. That screen showed the candidate was
slower and is retained in `reports/phase4-lm1b-small-perf-screen-v1/`.

An explicitly selected CUDA FP64 penalty candidate was then tested. It preserves
the existing CPU path, uses the pinned frequency/deficit/adjacency arithmetic,
and rejects invalid device results. The retained-logit probe matched all 256
CPU-selected IDs and passed tie, repeated-token, nonfinite, negative-allowance
and invalid-shape checks. A separate parallel CPU implementation matched its
sequential version in a focused five-row test. The full 1,280-sample CUDA
device-penalty development capture matched the previous CPU-penalty candidate
on all 820,480 codes and 163,840 final IDs, so the earlier [development FLM
scores](small-rust-development-quality-v1.md) apply. Its hash-checked
comparison repeated byte-for-byte, SHA-256
`ef832dadceb24392738ebccfb9e74644fea76e8c63cf4436c35de73a256fef99`.

The repeated candidate screen produced the following pooled timings:

| Runtime | Median | p95 | Samples/s at median | Median process VmHWM |
| --- | ---: | ---: | ---: | ---: |
| Python | 0.228202 s | 0.229602 s | 35.06 | 2,661,584 KiB |
| Rust device penalty | 0.195581 s | 0.196627 s | 40.90 | 1,629,668 KiB |

The pooled result has 24 timed batches per runtime. Python per-process medians
were 0.228259, 0.227874 and 0.228521 seconds; Rust medians were 0.195236,
0.195399 and 0.196222 seconds. Roughly one-second `nvidia-smi` samples observed
at most 2,282 MiB for each Python process and 1,416 MiB for each Rust process,
with only 2–4 samples per process; these are sampled observations, not measured
peaks. Four other GPU processes were resident. The result and manifest retain
PIDs, times, source and binary hashes, per-process host high-water marks and
activity telemetry. Result SHA-256:
`a0f6d9c757a4b6eb8686703ac139879c299f0e5822ab4585b87a80c89cb8f9f1`.

This is a batch-eight development screen, **not** a release performance gate:
it does not cover other batch sizes, a full memory envelope, an exclusive GPU,
official held-out quality, or Metal. The Python/Rust controlled stochastic
draw failure remains documented separately; no tolerance was changed.

Commands:

```sh
setpriv --inh-caps=-all env RUSTC_WRAPPER= cargo run --locked -p ncm-backend-cuda --release --features lm1b-experimental --example lm1b_device_penalty_probe -- artifacts/canonical/lm1b-small-inference-v1 reports/phase4-lm1b-small-controlled-rust-v1/traces.safetensors
setpriv --inh-caps=-all env RUSTC_WRAPPER= LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo run --locked -p ncm-backend-cuda --release --features lm1b-experimental,cuda-cudnn --example lm1b_small_device_development_capture -- artifacts/canonical/lm1b-small-inference-v1 SEED reports/phase4-lm1b-small-full-device-dev-seed-SEED-v1.json device-penalty
.reference/venv/bin/python tools/reference/lm1b_full_device_development_compare.py --output reports/phase4-lm1b-full-device-dev-comparison-v1.json
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/bench/run_lm1b_small_performance_screen.py --device-penalty --output reports/phase4-lm1b-small-perf-screen-v2
RUSTC_WRAPPER= cargo test --locked -p ncm-model --all-features
setpriv --inh-caps=-all env RUSTC_WRAPPER= LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo clippy --locked -p ncm-core -p ncm-model -p ncm-backend-cuda --all-features --all-targets -- -D warnings
setpriv --inh-caps=-all env RUSTC_WRAPPER= ./tools/check.sh --cuda > reports/phase4-lm1b-device-penalty-cuda-check-v1.log 2>&1
```

Feature tests and all-feature lint passed. The full CUDA regression passed,
including 18 device operator cases and the existing XSum FP64 penalty test.
Raw captures, logs and telemetry remain in ignored `reports/`. The benchmarking
source and binary hashes in each manifest capture the tested candidate even
though local changes had not yet been committed at measurement time.
