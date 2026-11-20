# WMT14 CUDA default-pool counter diagnostic

2026-11-20. The WMT14 fast-argmax **model timing example** now reports CUDA default memory-pool current and high-water used/reserved bytes after its timed trials. The collector validates and retains these separately from PyTorch allocator peaks; the Phase 5 summarizer preserves them when present and remains compatible with prior reports. No production inference or CLI behavior changes. The counter is the device's default CUDA pool high-water mark since process/context start, including allocations before the timed loop. It excludes non-pool allocations, other pools, CUDA context and potentially native-library workspace: **not** total process or physical GPU peak, not a directly interchangeable PyTorch peak, and not per-stage measurement. The collector queries it after timing, not in each timed iteration.

On the shared RTX 3060 Ti, one CFG=0, batch-1 diagnostic ran the previous and modified examples with identical pinned artifacts/prefixes, five warmups and 30 trials each. All codes/IDs matched exactly. The modified example reported default-pool bytes: current used `1062175648`, high used `1101203588`, current reserved `1140850688`, high reserved `1140850688`. Positivity and high-water ordering were checked. These are one-process diagnostic values, not paired timing evidence, whole-device peaks, or a release gate.

Baseline binary SHA-256 `8f630ee1d40ce6a632dafc3f6449d4f810b5a7cfa438edc6571e95ba24ed9774`; modified binary `0c0eb8606d56e06c8400d90dfee0de5e40ea8a4f3d127d9e94a63148ac695f6d`; raw baseline/candidate JSON SHA-256 `ad31b90fab3f00369b526b99cde60433a655cc5449fdb5c92c18e07cad24a00e` / `0a84f7d595ee536fd10eebdf8831664a7c3dc9c698d20488e429e0a3246eab41`. Raw output and baseline binary remain locally under ignored `reports/phase5-wmt14-pool-diagnostic-v1/`. Artifact/prefix checks are built into the example.

Targeted CUDA-feature Clippy, pool-counter/aggregate negative tests, byte-identical replay of the earlier WMT14 aggregate and `setpriv --inh-caps=-all tools/check.sh` passed. The real-device single-example check above exercised the new native query. The modified full collector has not yet been run end-to-end with it; exclusive GPU, current CLI/previous Rust/Python comparisons, native coverage outside the default pool and Metal remain pending.

Reproduce the candidate diagnostic with a fresh output file, after setting the CUDA/cuDNN environment in the [WMT14 performance report](cuda-wmt14-performance-retest-v1.md):

```sh
setpriv --inh-caps=-all cargo build -p ncm-backend-cuda --features wmt14-experimental,cuda-cudnn --example wmt14_fast_argmax_perf --release --locked
setpriv --inh-caps=-all target/release/examples/wmt14_fast_argmax_perf \
  artifacts/canonical/wmt14-inference-v1-final reports/phase4-wmt14-tokenizer-reference-v1.json 1 0 > reports/NEW-wmt14-pool-diagnostic.json
```
