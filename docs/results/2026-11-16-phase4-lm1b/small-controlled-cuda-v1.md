# Experimental LM1B-small controlled stochastic CUDA sampling

The experimental sampler draws initial codes from the pinned level-zero
unigram and subsequent codes from the full 8,192-way distribution with the
configured temperature 1, top-k 0, top-p 1 and CFG 0. It uses explicit CPU
FP64 inverse-CDF selection and a request-local SplitMix64 stream; it does not
reproduce PyTorch's RNG or its floating-point sampler. The sampled-output API
returns codes and CPU penalty-selected IDs. Traced and untraced outputs match
on the two inspected rows. This CPU implementation is not a performance claim
or production qualification.

Two fresh pinned upstream CUDA captures use seed 11 and the same independently
implemented controlled stream and inverse-CDF hook. They reproduce byte for
byte: report SHA-256
`aa2bd1f8db4da63cfba04bbb857997a31855c79a63d55ce51f86a46b89b1b643`,
trace SHA-256
`173f1be79510d9fabad465652e748a43d5d71d447c20acaa581e4c1476689c64`.
The capture source SHA-256 is
`c38c6b2474141f722d089de5870c520c804735beafcebef9486ca4531c9877f3`.
The Rust CUDA trace reproduced byte-for-byte across four captures, SHA-256
`8d5184bc9ffeb0600c724d2454682cf3acf79e6488de4079fd8d4c162bb364e5`.

**The exact cross-runtime comparison fails:** 21 of 51 tensors fail the
unchanged policy after 116 generated-code differences and 42 penalty-selected
ID differences. The first difference is level 7, row 1, position 43. Its
uniform is 0.7007409624056891; the cumulative boundary after code 3212 is
0.700740463967027 on Python logits and 0.700741351477724 on Rust logits.
The upstream selects 3215 and Rust selects 3212, and later choices cascade.
An independent replay on the captured Rust logits reproduces all 1,282 Rust
choices, including the two initial codes. This supports a near-boundary
explanation for this first difference, not a general cross-runtime parity
guarantee. The failure report SHA-256 is
`74df032c70b805bcd780167b27f4cca4f3bf89bf7504a2fea12922f69077c7c7`;
the replay audit SHA-256 is
`557a5bffcc1e61492f38240303e3961027ae7a92e3de7f43e89ad787c724270d`.
No tolerance was changed.

Commands (repeat the first with a new v2 directory):

```sh
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/reference/lm1b_small_controlled_trace.py --upstream .reference/upstream-ncm --source artifacts/upstream/lm1b-phase4 --prefixes reports/phase4-lm1b-small-cuda-development-v1.json --output reports/phase4-lm1b-small-controlled-cuda-reference-v1
setpriv --inh-caps=-all env RUSTC_WRAPPER= LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo run --locked -p ncm-backend-cuda --release --features lm1b-experimental,cuda-cudnn --example lm1b_small_controlled_trace -- artifacts/canonical/lm1b-small-inference-v1 reports/phase4-lm1b-small-controlled-cuda-reference-v1 reports/phase4-lm1b-small-controlled-rust-v4
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 .reference/venv/bin/python tools/reference/compare_lm1b_small_controlled_trace.py reports/phase4-lm1b-small-controlled-cuda-reference-v1/trace.safetensors reports/phase4-lm1b-small-controlled-rust-v1/traces.safetensors reports/phase4-lm1b-small-controlled-comparison-v1.json
.reference/venv/bin/python tools/reference/lm1b_small_controlled_audit.py reports/phase4-lm1b-small-controlled-cuda-reference-v1/trace.safetensors reports/phase4-lm1b-small-controlled-rust-v1/traces.safetensors reports/phase4-lm1b-small-controlled-audit-v1.json
RUSTC_WRAPPER= cargo test --locked -p ncm-model --all-features
setpriv --inh-caps=-all env RUSTC_WRAPPER= LIBRARY_PATH="$PWD/reports/phase4-wmt14-uncached-cfg-link-v1" LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" cargo clippy --locked -p ncm-model -p ncm-backend-cuda --all-features --all-targets -- -D warnings
setpriv --inh-caps=-all env RUSTC_WRAPPER= ./tools/check.sh --cuda > reports/phase4-lm1b-controlled-cuda-check-v1.log 2>&1
```

Focused RNG/invalid-input tests, all-feature model tests/lint and full CUDA
regression (18 device cases) pass. The initial lint found a style issue, which
was corrected. This two-row diagnostic
does not establish PyTorch-seed equivalence, stochastic quality, held-out
metrics, performance, Metal, or Phase 4 completion. Logs and traces are ignored
local artifacts.
