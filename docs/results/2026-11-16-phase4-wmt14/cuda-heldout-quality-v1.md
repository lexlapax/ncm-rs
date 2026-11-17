# WMT14 CUDA held-out quality

The [protocol](heldout-protocol-v1.md), committed before held-out capture,
requires exact raw IDs and bounded paired quality differences separately for
released CFG=0 and explicit diagnostic CFG=1. On source commit
`68b697225c62237687921b965cbf1da181d440b5`, the source-bound supervisor's
35-row smoke matched both earlier captures exactly, including the partial tail.
The supervisor then ran all 3,003 rows sequentially on pinned upstream Python
and the matching Rust binary, with batches of four and a three-row tail. The
input, source, binary and protocol postflight checks passed. Full captures are
under ignored `reports/phase4-wmt14-{base,cfg}-cuda-full-v1/`.

The base supervisor command was:

```sh
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/bench/capture_wmt14_cuda_full.py --mode base --head 68b697225c62237687921b965cbf1da181d440b5 --binary-sha256 5ad08c020bb152b1bf8c3f06f82e46389267c2a39393da49570d2ee9e32a64e3 --protocol-sha256 26f5f13aeb440da2f95f56091a482e221fa0c311934494a334566d9bc5ff8de1 --rows 3003 --output reports/phase4-wmt14-base-cuda-full-v1
```

The CFG run used `--mode cfg1`, binary SHA-256
`0437378dcf1d5937ea55e2065fcf187f8f30856b0593b8fa31e4a38f5d126736`, and
output `reports/phase4-wmt14-cfg-cuda-full-v1`. Decode and score commands use
the same checked tools and environments as the development record, with
`wmt14_score_heldout.py` and the full capture's comparison receipt. The
source-bound manifests and complete receipts include the remaining exact hashes.

| Mode | Code differences | Raw-ID differences | Held-out gate | Full-set gate |
| --- | ---: | ---: | --- | --- |
| Base CFG=0 | 0 / 939,939 | 0 / 192,192 | pass | pass |
| Diagnostic CFG=1 | 7 / 939,939 | 7 / 192,192 | **fail** | **fail** |

For base, full-set Python/Rust scores are identical: BLEU 21.7202, lowercase
BLEU 22.6121, chrF 49.5649, length ratio 0.9374. Held-out scores are BLEU
21.9187, lowercase BLEU 22.8200, chrF 49.7477 and length ratio 0.9386. The
full and held-out paired differences are all zero. The capture comparison
SHA-256 is `0d73b6cd0464f0e3dee464bfa27fdb05d46990298bde3413462716f54f0671c8`;
the score and independent offline replay both have SHA-256
`84a87dcf08a7ca8e9498aab4cc198fa3e53bca341637d8124d83ea5e1e8e7eda`.
Direct calls to the upstream sacreBLEU APIs produced the same full-set scores.

For CFG, all differences occur in row 2181. The first EOS is at position 23 in
both implementations, and every differing raw ID occurs after it, so all
decoded hypotheses and all scores match. Full-set scores are BLEU 22.9464,
lowercase BLEU 23.8358, chrF 51.5160 and length ratio 0.9587. Held-out scores
are BLEU 23.1470, lowercase BLEU 24.0310, chrF 51.6912 and length ratio 0.9597.
Despite identical decoded text and metric values, the frozen raw-ID gate fails.
The capture comparison SHA-256 is
`f249b33509b8e2bdbb77a64d2f71928157c616d954ef8c0c886250e965b4ea98`;
the score and independent offline replay both have SHA-256
`62904e917ae833718881b79b81eb4c4eb75eeef45fc954fcfa04332f2db8b486`.
An initial score command supplied an incorrect comparison hash and failed
before writing a result; the corrected run and the retained failure log remain
separate. The identical-hypotheses bootstrap shortcut is explicitly recorded
in both score reports and yields mathematically exact zero paired intervals.

The [exact-batch diagnostic and attempted arithmetic remediation](cfg-row2181-investigation-v1.md)
are documented separately. These quality results are not performance, memory, Metal or Phase 4
acceptance. CUDA desktop GPU processes were present; no timings are claimed.
