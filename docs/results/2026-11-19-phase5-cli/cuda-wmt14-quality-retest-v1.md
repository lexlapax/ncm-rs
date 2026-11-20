# WMT14 CUDA base quality retest and CLI corpus comparison

2026-11-20. Repeated the frozen [WMT14 protocol](../2026-11-16-phase4-wmt14/heldout-protocol-v1.md)
for the released CFG=0 mode. These rows were previously inspected, so this is a **nonblind retest**.
It does not replace or erase the previously recorded CFG=1 diagnostic failure.

The pinned tokenizer input capture was regenerated as
`reports/phase4-wmt14-tokenizer-reference-v1.json`, SHA-256
`22c76d597ae51dffadf61661d6b7c4d4290fd14b8809bfa6f1959b5c668df93f`. The existing
`tools/bench/capture_wmt14_cuda_full.py` supervisor captured pinned upstream Python and the
regular Rust CUDA example in the frozen batches of four, first for the 35-row smoke and then for
all 3,003 rows. It bound the clean source commit `a2e83e38afa79ca69460d10fd93a6625e52cb455`,
protocol hash `26f5f13aeb440da2f95f56091a482e221fa0c311934494a334566d9bc5ff8de1`, input,
artifact and binary identities. The 35-row Python capture matched the old pinned hash and had no
code or raw-ID differences. The full Python capture SHA-256 was
`e287a9c784d927cc319572fed6679bbf4cd9ffc1489c0f15028f71e13c47bce0`, exactly the original
pinned reference; regular Rust had zero differences against it. An initial build with only the
`cuda` feature failed because the example requires `wmt14-experimental`; the corrected build
used that feature and succeeded.

The current fast-argmax backend example was then built with `wmt14-experimental,cuda-cudnn`
(SHA-256 `6fe9e849a1aa1c6c5e7b43eba3f2980bf863077c3d5f9bfdd99abed41aee9ebc`) and captured the
same full corpus and batch grouping. Its output SHA-256 was
`d26b8a3b16a4ad07565f52d3e61f22b30e2512ea10348e674a24e9ed52fecc79`. The existing comparator
found **zero differences in all 939,939 codes and 192,192 raw IDs** against Python. The comparator
receipt SHA-256 `0d73b6cd0464f0e3dee464bfa27fdb05d46990298bde3413462716f54f0671c8` matches the
original base receipt. Binary, manifest, prefix and reference hashes were checked before/after.

Both captures were decoded using the pinned Qwen tokenizer. The two separately installed frozen
sacreBLEU scorer environments produced byte-identical held-out and full-set receipts (SHA-256
`84a87dcf08a7ca8e9498aab4cc198fa3e53bca341637d8124d83ea5e1e8e7eda`, matching the original
receipt). Both gates passed: full BLEU 21.7202, lowercase BLEU 22.6121, chrF 49.5649, length
ratio 0.9374; held-out BLEU 21.9187, lowercase BLEU 22.8200, chrF 49.7477, length ratio 0.9386.
The scorer explicitly used its mathematically exact identical-hypotheses bootstrap shortcut.

Separately, the current unified CLI (`target/release/ncm` SHA-256
`6194111fe3860584492676c6d14f50124a77f8dcf18f316a959b8abd16a20278`) was invoked on each of the
751 batches (750 of four and a three-row tail) using the same pinned prefixes. Its codes and raw
IDs exactly matched the Python reference for every row. The CLI comparison script verified the
binary, prefix, reference and qualified artifact manifest identities before and after; its ignored
summary `reports/phase5-wmt14-cli-corpus-cuda-v1/summary.json` SHA-256 is
`faf743c4df5a1bb97744939c8a7a7d618f7701ebcc3d61a69144eed02daf32f9`. The example and CLI used
the same optimized backend but are distinct executables; the full CLI row comparison directly
binds the scored output to the unified route.

Captures, per-batch CLI inputs/outputs and scorer receipts remain under ignored
`reports/phase5-wmt14-{base,fast}-cuda-full-v1/` and `reports/phase5-wmt14-cli-corpus-cuda-v1/`.
Generation used `setpriv --inh-caps=-all`, `OMP_NUM_THREADS=1` and
`TOKENIZERS_PARALLELISM=false`; the fast backend and CLI used the reference cuDNN library path.
The desktop GPU was shared and not continuously monitored; these are not performance measurements.
This establishes the scoped CUDA CFG=0 quality retest, not CFG=1, performance, Metal or the full
Phase 5 gate. The direct CLI corpus check used ID input; full source-text encoding and selected
text-input generation are separately documented in [the text screen](cuda-text-input-v1.md).
