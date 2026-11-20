# WMT14 CUDA CFG=1 quality retest

2026-11-20. This repeats the frozen [WMT14 protocol](../2026-11-16-phase4-wmt14/heldout-protocol-v1.md) on previously inspected rows; it is a **nonblind retest**, not a new held-out experiment. CFG=1 remains unflagged under the user's Phase 5 decision. The earlier CFG=1 diagnostic failure and its remediation remain recorded; this does not erase them.

At clean commit `c040301`, pinned upstream Python regenerated all 3,003 rows in batches of four. Its capture SHA-256 `78f24bba0fecc8ce7c06abf22508516142a5331cc6df5572efd533f9211e9256` reproduces the previously pinned CFG=1 reference. The current fast-argmax CUDA example produced SHA-256 `ff04571bfc72be02aee45ef7f64b6b4226431dca2264a7fd02d30be44294ef0f`, identical to the earlier scored candidate capture. The checked comparator found zero differences in 939,939 codes and 192,192 raw IDs; comparison SHA-256 `36aab40106bf6054264c833beaa636b4d5dc6c605822186e720915915abffe65`. The binary SHA-256 was `5daf22097b129db6587e38d68af0dc2132bbf06b028a85a0f143dfa854084ea9`. This example exercises the backend, not the unified CLI.

The pinned evaluator decoded both captures. Python decoded-reference SHA-256 `1dc421ee6bdd115e050c7d85069b7f7076c177881d35752181b0ee45ca7bf1df` matches the frozen identity; Rust decoded-capture SHA-256 `43392bd5c8011956f514a5f6bf3f8090d72a09dd3cd08b3027fc02d5c3e3b0d2`. Both independent frozen scorer environments emitted byte-identical, passing full and held-out receipts, SHA-256 `915cae3bf2351db1f90bd290ebbe79353a4b9bdb91a67cb494d30791403f89bc`, matching the previous CFG=1 receipt. Full BLEU/lowercase BLEU/chrF/length ratio were 22.9464/23.8358/51.5160/0.9587; held-out values were 23.1470/24.0310/51.6912/0.9597. The scorer used its identical-hypotheses bootstrap shortcut.

Reproduction, using fresh output paths and the existing pinned environments and inputs:

```sh
out=reports/NEW-wmt14-cfg1-retest
mkdir "$out"
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  .reference/venv/bin/python tools/reference/wmt14_cfg_capture.py \
  --upstream .reference/upstream-ncm --source artifacts/upstream/wmt14-phase4 \
  --prefixes reports/phase4-wmt14-tokenizer-reference-v1.json \
  --output "$out/python.json" --start 0 --count 3003
export LIBRARY_PATH="$PWD/.reference/phase4-cudnn-probe-lib"
export LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib"
export NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 OMP_NUM_THREADS=1
setpriv --inh-caps=-all cargo build -p ncm-backend-cuda --example wmt14_fast_argmax_cfg_capture --features wmt14-experimental,cuda-cudnn --release --locked
setpriv --inh-caps=-all target/release/examples/wmt14_fast_argmax_cfg_capture \
  artifacts/canonical/wmt14-inference-v1-final reports/phase4-wmt14-tokenizer-reference-v1.json "$out/rust.json" 0 3003
setpriv --inh-caps=-all .reference/venv/bin/python tools/reference/compare_wmt14_cfg_capture.py \
  "$out/python.json" "$out/rust.json" "$out/comparison.json" \
  --reference-sha256 78f24bba0fecc8ce7c06abf22508516142a5331cc6df5572efd533f9211e9256 \
  --candidate-sha256 ff04571bfc72be02aee45ef7f64b6b4226431dca2264a7fd02d30be44294ef0f
for runtime in python rust; do
  if [ "$runtime" = python ]; then sha=78f24bba0fecc8ce7c06abf22508516142a5331cc6df5572efd533f9211e9256; else sha=ff04571bfc72be02aee45ef7f64b6b4226431dca2264a7fd02d30be44294ef0f; fi
  setpriv --inh-caps=-all env TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 .reference/tokenizers-venv/bin/python \
    tools/reference/wmt14_decode_capture.py "$out/$runtime.json" "$out/$runtime-text.json" --capture-sha256 "$sha"
done
for v in wmt14-scorer-venv wmt14-scorer-replay-venv; do
  setpriv --inh-caps=-all env OMP_NUM_THREADS=1 .reference/$v/bin/python tools/reference/wmt14_score_heldout.py \
    "$out/python-text.json" "$out/rust-text.json" "$out/comparison.json" "$out/$v.json" \
    --reference-text-sha256 1dc421ee6bdd115e050c7d85069b7f7076c177881d35752181b0ee45ca7bf1df \
    --candidate-text-sha256 43392bd5c8011956f514a5f6bf3f8090d72a09dd3cd08b3027fc02d5c3e3b0d2 \
    --comparison-sha256 36aab40106bf6054264c833beaa636b4d5dc6c605822186e720915915abffe65 \
    --protocol-sha256 26f5f13aeb440da2f95f56091a482e221fa0c311934494a334566d9bc5ff8de1 --mode cfg1
done
cmp "$out/wmt14-scorer-venv.json" "$out/wmt14-scorer-replay-venv.json"
```

The captures, logs and receipts remain locally under ignored `reports/phase5-wmt14-cfg1-cuda-retest-v1/`; they are not durably published. This shared-GPU run is not a performance measurement. Full current-CLI CFG=1 corpus equivalence, release performance, Metal and the overall Phase 5 gates remain pending.
