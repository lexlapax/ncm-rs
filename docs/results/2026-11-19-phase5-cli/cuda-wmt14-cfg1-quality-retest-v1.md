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

The captures, logs and receipts remain locally under ignored `reports/phase5-wmt14-cfg1-cuda-retest-v1/`; they are not durably published. This shared-GPU run is not a performance measurement. Release performance, Metal and the overall Phase 5 gates remain pending.

## Unified CUDA CLI corpus equivalence

The release-mode current unified CLI, built from `d5b2931` (the new runner and its tests were untracked during the build; tracked sources were clean), matched the frozen Python CFG=1 capture above on all 3,003 rows in the same 751 batches (750 of four and a three-row tail). This covers 939,939 codes and 192,192 IDs; it binds the scored corpus to the unified ID-input route. Candidate binary SHA-256 `83f9a9a243ac83a7b8f6cf888ddebcdafa9fd49f7dbf45b44df7d1a3429bd8ce`; runner SHA-256 `4a1f620b3dd1e9f386de842f6f5edb6bce45c524bda6961e4ae930517d7e5826`; ignored summary SHA-256 `ac756e22aa0a49b11917452ab768b191487c68acd4db23b97bd135ac7f761fa5`. The runner checked binary, input, reference and manifest hashes before and after and checked every output; an independent replay checked all stored inputs, output hashes, codes, IDs and empty stderr against the reference. Runner unit/negative tests and `setpriv --inh-caps=-all tools/check.sh` passed.

```sh
export LIBRARY_PATH="$PWD/.reference/phase4-cudnn-probe-lib"
export LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib"
export NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 OMP_NUM_THREADS=1
setpriv --inh-caps=-all cargo build -p ncm-cli --features cuda-cudnn --release --locked
setpriv --inh-caps=-all python3 tools/bench/phase5_wmt14_cfg1_cli.py \
  --binary target/release/ncm --binary-sha256 83f9a9a243ac83a7b8f6cf888ddebcdafa9fd49f7dbf45b44df7d1a3429bd8ce \
  --prefixes reports/phase4-wmt14-tokenizer-reference-v1.json \
  --reference reports/phase5-wmt14-cfg1-cuda-retest-v1/python.json \
  --artifact artifacts/canonical/wmt14-inference-v1-final --output reports/NEW-wmt14-cfg1-cli
```

Raw inputs, outputs, errors and summary remain under ignored `reports/phase5-wmt14-cfg1-cli-corpus-v1/`. This is a nonblind shared-GPU output retest, not a performance or Metal gate. The runner was subsequently extended for alias comparison below; the SHA above identifies the original runner at `06b6721`.

## Legacy and pre-change aliases

The extended runner invoked both the retained pre-change binary and current release binary through `sample-wmt14` at CFG=1, using the same frozen batch grouping. Both matched Python on all 3,003 rows. An independent replay then compared the saved input and stdout bytes across pre-change alias, current alias and the previously captured unified CLI: all 751 batches were byte-identical, and all stderr files were empty. This is broader than the fixed-request gate, but remains nonblind shared-GPU correctness evidence, not performance or Metal qualification.

The extended runner SHA-256 is `d021f36cef17965d848fb2ed26ae42767cec4c9f47918fb1696a2f3c383af980`; pre-change binary SHA-256 `f10abaaec672dd7c0c62034972fb4b15fc204f2b1f5693e4a09f5e9640767098`; current binary `83f9a9a243ac83a7b8f6cf888ddebcdafa9fd49f7dbf45b44df7d1a3429bd8ce`. The ignored old/current summary hashes are `f8d9ec03da879de9accbbad9f4fced6e24fa284d59ccf5e245914989cb89df1a` / `cacbc53af6d3700a496da218f6f35c2f5de1338caa78b53f507cbd5ee8140e1d`. Raw files remain under `reports/phase5-wmt14-cfg1-{old,current}-alias-v1/`. Four focused runner tests and `setpriv --inh-caps=-all tools/check.sh` passed.

Reproduce using the same pinned input, artifact, reference and library environment as above; supply a fresh output path for each invocation:

```sh
setpriv --inh-caps=-all python3 tools/bench/phase5_wmt14_cfg1_cli.py --route alias \
  --binary reports/phase5-cli-equivalence/baseline-ncm \
  --binary-sha256 f10abaaec672dd7c0c62034972fb4b15fc204f2b1f5693e4a09f5e9640767098 \
  --prefixes reports/phase4-wmt14-tokenizer-reference-v1.json \
  --reference reports/phase5-wmt14-cfg1-cuda-retest-v1/python.json \
  --artifact artifacts/canonical/wmt14-inference-v1-final --output reports/NEW-wmt14-cfg1-old-alias
# Repeat with target/release/ncm, binary hash 83f9a9a243ac83a7b8f6cf888ddebcdafa9fd49f7dbf45b44df7d1a3429bd8ce and a different output directory.
```
