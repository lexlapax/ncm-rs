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
