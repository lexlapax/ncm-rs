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
