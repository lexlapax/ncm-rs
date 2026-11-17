# Experimental LM1B-small CUDA device sampling

An explicitly selected experimental CUDA path samples each 8,192-way head on
device with FP64 inverse-CDF arithmetic and the same midpoint-u32 SplitMix64
words used by the existing CPU sampler. Initial unigram sampling and final
FP64 penalty selection remain on the CPU. Unsupported backends fail explicitly;
nonfinite device rows return a sentinel that is checked before embedding. The
CPU sampler remains the default. Different reduction orders can change draws at
very close boundaries, so exact agreement is not assumed for unseen inputs.

On retained Rust controlled logits, the device operation matched all 1,280
CPU head choices. Separate all-equal, concentrated, midpoint-boundary, NaN and
infinity cases passed. A full two-row controlled trace matched the existing
CPU trace byte-for-byte (SHA-256
`8d5184bc9ffeb0600c724d2454682cf3acf79e6488de4079fd8d4c162bb364e5`),
including traced/untraced output agreement. This does not resolve the separate
[Python/Rust controlled comparison failure](small-controlled-cuda-v1.md).

For all five previously captured non-held-out development seeds 11–15, each
with 256 batch-eight samples, device and CPU outputs matched exactly: 820,480
codes and 163,840 final IDs. The hash-checked comparison repeated byte-for-byte
with SHA-256 `d5a22d90b1225b291ce6592ff64d0d4de6af44a3457db83f24b4acf0e3b1996e`.
Because the raw IDs match, the existing [development FLM scores](small-rust-development-quality-v1.md)
apply to those outputs. This is not held-out quality evidence.

A small alternating batch-eight timing screen, one warmup and three measured
rounds, matched complete CPU/device outputs in each round. Median end-to-end
elapsed time was 0.2952 seconds with device sampling and 0.5112 seconds with
CPU sampling. This includes explicit CPU final penalties, has no Python timing
comparison, and ran with four other GPU processes resident; it is a screen,
not a performance or memory gate. The screen log SHA-256 is
`cebb9ca8fd88c40b176b2bb99310bd4d693fa71f94299b436581ce5f21cef876`.

Commands:

```sh
setpriv --inh-caps=-all env RUSTC_WRAPPER= cargo run --locked -p ncm-backend-cuda --release --features lm1b-experimental --example lm1b_categorical_probe -- reports/phase4-lm1b-small-controlled-rust-v1/traces.safetensors
setpriv --inh-caps=-all env RUSTC_WRAPPER= LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo run --locked -p ncm-backend-cuda --release --features lm1b-experimental,cuda-cudnn --example lm1b_small_device_trace -- artifacts/canonical/lm1b-small-inference-v1 reports/phase4-lm1b-small-controlled-cuda-reference-v1 reports/phase4-lm1b-small-device-trace-v1
setpriv --inh-caps=-all env RUSTC_WRAPPER= LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo run --locked -p ncm-backend-cuda --release --features lm1b-experimental,cuda-cudnn --example lm1b_device_sampling_screen -- artifacts/canonical/lm1b-small-inference-v1
setpriv --inh-caps=-all env RUSTC_WRAPPER= LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo run --locked -p ncm-backend-cuda --release --features lm1b-experimental,cuda-cudnn --example lm1b_small_device_development_capture -- artifacts/canonical/lm1b-small-inference-v1 SEED reports/phase4-lm1b-small-device-dev-seed-SEED-v1.json
.reference/venv/bin/python tools/reference/lm1b_device_development_compare.py --output reports/phase4-lm1b-device-dev-comparison-v3.json
RUSTC_WRAPPER= cargo test --locked -p ncm-model --all-features
setpriv --inh-caps=-all env RUSTC_WRAPPER= LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo clippy --locked -p ncm-core -p ncm-model -p ncm-backend-cuda --all-features --all-targets -- -D warnings
setpriv --inh-caps=-all env RUSTC_WRAPPER= ./tools/check.sh --cuda > reports/phase4-lm1b-device-sampling-cuda-check-v1.log 2>&1
```

All-feature model tests/lint and the full CUDA check passed, including 18
real-device operator cases. The first lint found an argument-count warning;
it was annotated before the passing run. Logs and captures are ignored local
artifacts. No Metal, official quality, release performance or Phase 4 gate is
claimed.
