# WMT14 CFG CUDA diagnostic reference

The released checkpoint config specifies CFG zero. To cover the Phase 4 CFG
requirement, this separate diagnostic uses explicit CFG scale 1.0 with the
upstream null prefix (EOS 151645 followed by PAD 151643) and otherwise retains
the pinned temperature 1, top-k 1 and top-p 0 generation settings. The original
upstream guidance and sampler execute unchanged; hooks capture guided head
logits. Two verified source prefixes, eleven generated code levels, eleven
reconstruction levels, final logits and greedy IDs make 26 tensors. The model,
checkpoint and prefix files are checked before and after; TF32 is disabled and
math SDPA is selected.

```sh
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/reference/wmt14_cfg_trace.py --upstream .reference/upstream-ncm --source artifacts/upstream/wmt14-phase4 --prefixes reports/phase4-wmt14-tokenizer-reference-v1.json --output reports/phase4-wmt14-cfg-cuda-reference-v1
```

A second run with output suffix `v2` gave the same trace SHA-256
`14c7b32872de7fa4d6b7617f69b40a63b1831c02a133257a89546e8870ce241e` and
report SHA-256 `6312cd97dca116319a70f8e9c7899c01aa0c4f6feba5dfa363ad53b7741adbda`.
The script SHA-256 is `41739fa50d950ee5ac5092bad9a66b36cf0b372a86f6d1e7eea368851d97222b`.
All 626 sampled codes match the guided-logit argmax; no exact ties occurred and
the minimum top-two gap was 0.005832672119140625. The focused prefix
validation test passes. Both captures ran while the CUDA
test suite was compiling on CPU; desktop GPU processes were also present.
No timing claim is made. This is an upstream two-prefix diagnostic, not Rust
CFG parity, task quality, performance or Metal evidence.

## Experimental Rust comparison

The opt-in `wmt14-experimental` path constructs the EOS/PAD null branch,
applies the upstream level-dependent guidance and rejects tied or nonfinite
head maxima. It was compared using the exact checked artifact and reference:

```sh
setpriv --inh-caps=-all env RUSTC_WRAPPER= CUDA_COMPUTE_CAP=86 cargo build --release -p ncm-backend-cuda --features wmt14-experimental --example wmt14_cfg_trace
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 target/release/examples/wmt14_cfg_trace artifacts/canonical/wmt14-inference-v1-final reports/phase4-wmt14-cfg-cuda-reference-v1 reports/phase4-wmt14-cfg-rust-cuda-v2
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 .reference/venv/bin/python tools/reference/compare_wmt14_cfg_trace.py reports/phase4-wmt14-cfg-cuda-reference-v1/trace.safetensors reports/phase4-wmt14-cfg-rust-cuda-v2/traces.safetensors reports/phase4-wmt14-cfg-cuda-comparison-v2.json
```

All 26 tensors pass the unchanged bands and all 626 guided argmax choices,
generated codes and greedy IDs match. The largest absolute difference is
0.00011444091796875 in `generator.level_10.logits`. The native trace SHA-256 is
`8f781c0763a746df004470c58ecf2ed7331d08ec314ed80b472caa7dd54d1d6e`,
comparison SHA-256 `794a59ad603f72cbe4968d8f978e7b172840265e342fb7bdf6f22bcc05c958f0`,
and executable SHA-256 `2b21317b59e2c7dd7dd7c975d800451bb5cc0b97ef6dac1c0d66414ad0756f52`.
The CFG-zero top-1 and fixed-history teacher traces are byte-identical to their
earlier captures after this change. Focused model tests include null-prefix
and invalid-scale cases; comparator tests and strict feature Clippy pass. Full
`tools/check.sh --cuda` passed 18 real-device cases (log SHA-256
`85170a99a89868d019d8a4cafed6cd7891bfbd9c73827b34e9dc1f26da91c79a`). The
capture binds artifact and reference hashes but does not perform a source
postflight. This is a two-prefix diagnostic, not task quality, performance or
Metal evidence, and the normal WMT14 inference path remains unsupported.
