# WMT14 top-1 CUDA reference

The pinned upstream generator was run on the first two verified WMT14 source
prefixes using the released temperature 1, top-k 1, top-p 0 and CFG 0 settings.
Its original sampler runs unchanged; hooks only record the eleven head logits.
The 26-tensor trace also includes generated codes, eleven reconstruction levels,
prefix IDs, final logits and greedy decoded IDs. TF32 is disabled, math SDPA is
selected, and pinned inputs and source files are checked before and after.

```sh
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/reference/wmt14_top1_trace.py --upstream .reference/upstream-ncm --source artifacts/upstream/wmt14-phase4 --prefixes reports/phase4-wmt14-tokenizer-reference-v1.json --output reports/phase4-wmt14-top1-cuda-reference-v1
```

A second run using output suffix `v2` produced identical trace SHA-256
`68e56edf9d8efedbd7fa2ec41bc6ea1b25aba8d24978ba6d057a585926260ecb` and
report SHA-256 `8115734c26df556d991320bbd3459d59ec20bfb56ea8676d407de47748f539b7`.
The script SHA-256 is `e418da9228b22d4129cc3ea3af9c4e10686a0dc2791aa4fb8f56a876358155fa`.
All 626 sampled code choices equal their respective head argmax; no exact head
ties occurred and the minimum top-two gap was 0.00031185150146484375. Desktop
GPU processes were present. These two rows are a development fixture, not
full task quality, performance, Rust generation or Metal evidence.

## Experimental Rust comparison

The non-default `wmt14-experimental` feature provides a traced CFG-zero
top-1 development path. It rejects tied or nonfinite head maxima explicitly;
it does not claim PyTorch tie ordering or implement CFG. Using the exact
artifact and reference identities embedded in the example:

```sh
setpriv --inh-caps=-all env RUSTC_WRAPPER= CUDA_COMPUTE_CAP=86 cargo build --release -p ncm-backend-cuda --features wmt14-experimental --example wmt14_top1_trace
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 target/release/examples/wmt14_top1_trace artifacts/canonical/wmt14-inference-v1-final reports/phase4-wmt14-top1-cuda-reference-v1 reports/phase4-wmt14-top1-rust-cuda-v1
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 .reference/venv/bin/python tools/reference/compare_wmt14_top1_trace.py reports/phase4-wmt14-top1-cuda-reference-v1/trace.safetensors reports/phase4-wmt14-top1-rust-cuda-v1/traces.safetensors reports/phase4-wmt14-top1-cuda-comparison-v1.json
```

All 26 trace tensors pass the existing bands, all 626 head argmax choices and
generated codes match, and recorded greedy IDs match exactly. The native trace
SHA-256 is `d1f87ce3c53dcbae6f7965fb643ce53e5d8aa73a737eb604254c7ea5cab4feb2`,
comparison SHA-256 `f6ba3897942c305c0c09045c430625252922e3de0f83e1688318dcafd5d252a3`,
and executable SHA-256 `a7ed291ca19dc060f20a15a2074f17afab3ba97316f48c43cac3553ec87777e7`.
The fixed-history teacher capture after the shared-path refactor exactly
matches its previous trace SHA-256
`cff57cecc1ba1e486fdad62ed1050754266cdfff126e94111063d3dd9db718ee`.
Feature-enabled model tests, comparator regression and strict feature Clippy
pass. Full `tools/check.sh --cuda` passed, including 18 real-device cases (log
SHA-256 `753235ebbaf2f660f19d1d46395ce9a0d221932d2e4d9723848881bb17c1d38a`).
The capture checks artifact and reference hashes but does not perform
source postflight. Two inputs do not qualify end-to-end task quality,
performance, CFG or Metal; normal WMT14 inference remains unsupported.
