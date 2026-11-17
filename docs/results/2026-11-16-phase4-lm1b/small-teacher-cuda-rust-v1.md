# Experimental LM1B-small fixed-history CUDA teacher

The opt-in `lm1b-experimental` teacher validates the pinned artifact, uses the
shared transformer and codec, and accepts the explicit first code and forced
subsequent codes. It implements the shared 8,192-way head with per-level bias,
the residual post-upsampling convolutions, and the evaluator's final frequency,
deficit and adjacency penalties as an explicit CPU FP64 selection. It does not
perform free generation or sampling; normal LM1B artifact loading remains
unsupported.

The two retained development histories in the [pinned CUDA reference](small-teacher-cuda-reference-v1.md)
pass all 51 tensor comparisons under the existing `codec-fp32-v1` policy.
All 1,280 recorded head argmax choices match, as do forced codes, greedy IDs
and penalty-adjusted IDs. Candidate SHA-256:
`b1f901bb5f505ced27a334fae45d064b556e8105c8dde806470150c7f394ad62`.
Comparison report SHA-256:
`18e5a31821d961ed4e8cdc51ade8fd3fe24ab7665eba2f04a946f1cf06987922`.
Strict FP32 cuDNN was explicitly selected, following the preceding codec
comparison. No tolerance or reference was changed.

Commands:

```sh
setpriv --inh-caps=-all env RUSTC_WRAPPER= LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo run --locked -p ncm-backend-cuda --release --features lm1b-experimental,cuda-cudnn --example lm1b_small_teacher_trace -- artifacts/canonical/lm1b-small-inference-v1 reports/phase4-lm1b-small-teacher-cuda-reference-v3 reports/phase4-lm1b-small-teacher-rust-v1
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 .reference/venv/bin/python tools/reference/compare_lm1b_small_teacher_trace.py reports/phase4-lm1b-small-teacher-cuda-reference-v3/trace.safetensors reports/phase4-lm1b-small-teacher-rust-v1/traces.safetensors reports/phase4-lm1b-small-teacher-comparison-v1.json
RUSTC_WRAPPER= cargo test --locked -p ncm-model --all-features
setpriv --inh-caps=-all env RUSTC_WRAPPER= LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo clippy --locked -p ncm-model -p ncm-backend-cuda --all-features --all-targets -- -D warnings
setpriv --inh-caps=-all env RUSTC_WRAPPER= ./tools/check.sh --cuda > reports/phase4-lm1b-teacher-cuda-check-v1.log 2>&1
```

The feature tests include deterministic tie/repetition and invalid-input checks
for the CPU penalty selector; all-feature model tests/lint and the full CUDA
check passed, including 18 device operator cases. This two-history result does not establish free
generation, stochastic sampling, quality, performance, Metal, or Phase 4
completion. Raw traces and logs are ignored local artifacts.
