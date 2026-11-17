# Experimental LM1B-small greedy CUDA continuation

The experimental Rust path now continues from explicit level-zero codes with
temperature-zero greedy choices at each later level. It uses the same shared
execution as the fixed-history teacher, rejects nonfinite head logits, and
applies the evaluator's final penalties on the CPU in FP64. It is unoptimized
and reads head logits back to check finiteness. The first code is supplied, not
sampled. Normal LM1B loading remains unsupported.

Two fresh pinned upstream CUDA captures use level-zero codes from the first
two inspected development rows and CLS prefixes. Their reports and 51-tensor
traces are byte-identical: report SHA-256
`874772e2a12028ab2432e5e5a0a6edff6cb8d30c43d06564e53338b68a16e8b8`,
trace SHA-256
`03b598e97609d01f46671523709a55836c4894050bbd7055c857ea9e7d6cea78`.
The upstream temperature-zero branch is argmax; hooks record logits and do not
force later codes. The capture source SHA-256 is
`65796db59d0ee056081751203d4fb03afe8ab8faef991deae400b7bf855f55b2`.

Strict FP32 cuDNN Rust passes all 51 comparisons under the existing policy,
including 1,280 head choices, generated codes, greedy IDs and penalty-adjusted
IDs. The Rust trace SHA-256 is
`d3ccff390fa1d9befd20134862f2d7b2d0339d2a67d4f86dfd1cb9a1cdb19253`;
the comparison SHA-256 is
`59b8ea1a302b5391fe140079c6f63003e39da264919f83e29d2ee554607d30b1`.
The Rust greedy trace repeated byte-for-byte after the final input-validation
change; the earlier fixed-history Rust trace also repeated after the shared
loop changed. Input-bound tests, all-feature model tests/lint and the full CUDA
regression passed, including 18 device operator cases.

Commands (repeat the first with a fresh `v2` output directory):

```sh
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/reference/lm1b_small_greedy_trace.py --upstream .reference/upstream-ncm --source artifacts/upstream/lm1b-phase4 --prefixes reports/phase4-lm1b-small-cuda-development-v1.json --output reports/phase4-lm1b-small-greedy-cuda-reference-v1
setpriv --inh-caps=-all env RUSTC_WRAPPER= LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo run --locked -p ncm-backend-cuda --release --features lm1b-experimental,cuda-cudnn --example lm1b_small_greedy_trace -- artifacts/canonical/lm1b-small-inference-v1 reports/phase4-lm1b-small-greedy-cuda-reference-v1 reports/phase4-lm1b-small-greedy-rust-v1
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 .reference/venv/bin/python tools/reference/compare_lm1b_small_greedy_trace.py reports/phase4-lm1b-small-greedy-cuda-reference-v1/trace.safetensors reports/phase4-lm1b-small-greedy-rust-v1/traces.safetensors reports/phase4-lm1b-small-greedy-comparison-v1.json
RUSTC_WRAPPER= cargo test --locked -p ncm-model --all-features
setpriv --inh-caps=-all env RUSTC_WRAPPER= LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo clippy --locked -p ncm-model -p ncm-backend-cuda --all-features --all-targets -- -D warnings
setpriv --inh-caps=-all env RUSTC_WRAPPER= ./tools/check.sh --cuda > reports/phase4-lm1b-greedy-cuda-check-v1.log 2>&1
```

This two-row development diagnostic is not stochastic generation, held-out
quality, performance, Metal, or completion of the Phase 4 gate. Captures and
logs remain in ignored local `reports/`.
