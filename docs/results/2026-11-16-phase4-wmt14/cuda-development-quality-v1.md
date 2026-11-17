# WMT14 CUDA development scoring

Rows 0–127 are the designated development set; rows 0–31 had already been
inspected in the smaller screen. The pinned upstream and experimental Rust
CFG-zero captures used the [bounded tools](cuda-range-capture-v1.md), start 0,
count 128 and batches of four. They matched all 40,064 codes and 8,192 raw IDs.
The upstream capture SHA-256 is
`f0f277777f5b32105979bbe11cbb97df0dfd17c0e73e77f289094a4e828b95bf`,
Rust `8e88c9694b422d8ebe0281fae6e4375bedea48e1120103411a3ea71625055e5c`,
and exact comparator
`ed5d4a8677b15ac663672842891af9f1fe6e4e4bf466c592fed18282203bf6e9`.
The captures are development data and not a held-out or performance result.

The upstream requirements name sacreBLEU without pinning a version. For this
new evaluation environment, explicitly select sacreBLEU 2.5.1 and pin its
resolved dependencies in [the requirements](../../../tools/reference/requirements-wmt14-scorer-v1.txt)
and [wheel SHA-256 inventory](../../../tools/reference/wmt14-scorer-wheel-sha256-v1.txt).
The wheels live in ignored `artifacts/wmt14-scorer-wheelhouse-v1/`; they are not
runtime dependencies or an upstream pin change. An independent Python 3.14
venv installed those wheels offline and reproduced the scorer result byte for
byte. To reproduce, verify the wheel inventory first, create a fresh venv, and
install with `pip install --no-index --find-links artifacts/wmt14-scorer-wheelhouse-v1
-r tools/reference/requirements-wmt14-scorer-v1.txt`. The installation and
offline replay logs remain under `reports/`.

Using the already qualified tokenizer environment and pinned local Qwen
files, `tools/reference/wmt14_decode_capture.py` applies the upstream EOS cut,
PAD filtering and special-token-skipping decode. Python and Rust hypotheses
and references are identical. The first test attempt used the CUDA environment
which lacked Transformers; the first tokenizer-environment attempt imported a
helper that required unavailable Torch. Both failures were corrected by using
the qualified tokenizer environment and a Torch-free JSON writer; focused
decode validation passes.

`tools/reference/wmt14_score_dev.py` applies the upstream default cased BLEU,
lowercase BLEU, chrF and whitespace length ratio. It computes 1,000 paired-row
bootstrap resamples using seed 42, solely to characterize the development
baseline. Both implementations have the same score and bootstrap output:

```sh
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/tokenizers-venv/bin/python tools/reference/wmt14_decode_capture.py reports/phase4-wmt14-python-cuda-dev128-v1.json reports/phase4-wmt14-python-cuda-dev128-text-v1.json --capture-sha256 f0f277777f5b32105979bbe11cbb97df0dfd17c0e73e77f289094a4e828b95bf
setpriv --inh-caps=-all .reference/wmt14-scorer-venv/bin/python tools/reference/wmt14_score_dev.py reports/phase4-wmt14-python-cuda-dev128-text-v1.json reports/phase4-wmt14-python-cuda-dev128-score-v1.json --input-sha256 f4220cf0d5555b2380c70817963a7b6e0981dea3f0a97d89d3b52f9cedd71c77
```

The Rust decode and scoring used the corresponding Rust capture and checked
hashes `8e88c9694b422d8ebe0281fae6e4375bedea48e1120103411a3ea71625055e5c`
and `70dd1f35d5f5407ea76642680381205a99ca5b6ed830675f7fc6c2e873a6f98b`.

| Metric | Development value | Baseline 95% bootstrap interval |
| --- | ---: | ---: |
| BLEU | 17.0030 | 14.8030–19.2715 |
| BLEU lowercase | 17.6661 | 15.3391–19.8653 |
| chrF | 45.3214 | 42.9108–47.4052 |
| Length ratio | 0.9087 | 0.8765–0.9384 |

The full signatures are `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.5.1`,
`nrefs:1|case:lc|eff:no|tok:13a|smooth:exp|version:2.5.1` and
`nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.5.1` for the three
metrics respectively. The upstream score report and independent offline replay
both have SHA-256 `46cf7c718a3c4200a7452e525ee66d4894a9f057d2a6996825af8eb9baa222f5`.
The Rust score report differs only in its input provenance and has SHA-256
`904bb5c3a418b4bc518f92234381a50eaa95a8d0531b4480b2bc2bd6674ce401`.
The scorer unit test checks metric defaults and input rejection. A separately
captured CFG development baseline and explicit frozen held-out criteria are
still required before held-out scoring; no full-corpus, performance or Metal
gate is claimed.
