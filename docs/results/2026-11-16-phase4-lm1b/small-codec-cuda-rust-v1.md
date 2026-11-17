# Experimental LM1B-small CUDA codec

The opt-in `lm1b-experimental` codec uses the pinned checked artifact and the
shared codec implementation. The explicit profile covers an encoder without
positions, a four-block decoder with positional embeddings, and one learned
128-wide initial up/down convolution. Normal LM1B artifact construction still
rejects execution; this experimental loader is not model or release support.
The strict cuDNN convolution accepts the bounded 128-wide, stride-128,
zero-padding case required by this codec.

Two previously inspected development rows were compared against the pinned
upstream CUDA trace documented in [the reference record](small-codec-cuda-reference-v1.md).
Strict FP32 cuDNN passes all 362 tensor comparisons under the existing
`codec-fp32-v1` policy, including exact encoded codes and both decoded ID sets.
The candidate trace SHA-256 is
`26d5b1d0a3ced2b4e4a8f634f8ba54743cfda46407f36e5a499611fd4b13959d`;
the comparison report SHA-256 is
`9ff1fa60050a7d7a293f6589229a6746562907c8fa62893e16557ef6e8f0b277`.
An initial Candle CUDA capture had 120 policy failures, despite exact recorded
codes and IDs; it is retained. Enabling the cuDNN feature alone did not select
the strict backend and reproduced that same result. A first strict attempt
explicitly rejected the previously unsupported 128-wide convolution; the
bounded extension enabled the passing capture. No tolerance was changed.

Commands:

```sh
setpriv --inh-caps=-all env RUSTC_WRAPPER= LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo run --locked -p ncm-backend-cuda --release --features lm1b-experimental,cuda-cudnn --example lm1b_small_codec_trace -- artifacts/canonical/lm1b-small-inference-v1 reports/phase4-lm1b-small-codec-cuda-reference-v1 reports/phase4-lm1b-small-codec-rust-v5
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 .reference/venv/bin/python tools/reference/compare_lm1b_small_codec_trace.py reports/phase4-lm1b-small-codec-cuda-reference-v1/traces.safetensors reports/phase4-lm1b-small-codec-rust-v5/traces.safetensors reports/phase4-lm1b-small-codec-comparison-v2.json
RUSTC_WRAPPER= cargo test --locked -p ncm-core --test lm1b_small_config
RUSTC_WRAPPER= cargo test --locked -p ncm-model --all-features
setpriv --inh-caps=-all env RUSTC_WRAPPER= LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo clippy --locked -p ncm-core -p ncm-model -p ncm-backend-cuda --all-features --all-targets -- -D warnings
setpriv --inh-caps=-all env RUSTC_WRAPPER= ./tools/check.sh --cuda
```

The comparison covers two inputs only, not generation, task quality,
performance, Metal, or the Phase 4 gate. All-feature model tests and all-target
lint for core, model and CUDA passed. The full CUDA check passed, including 18
device operator cases; the initial check stopped at formatting, corrected before
the passing run. Logs are in ignored `reports/`.
