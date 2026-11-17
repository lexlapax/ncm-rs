# WMT14 CUDA reference candidate retest

The scoped reference-order initial reconstruction was developed after inspecting
the [original CFG failure](cuda-heldout-quality-v1.md) and its
[row-2181 diagnostics](cfg-row2181-investigation-v1.md). This full-set replay
is therefore a retest on previously inspected data, not a new blind held-out
experiment. The original failed capture and unchanged frozen protocol remain
preserved. Performance, memory and Metal validation are still pending.

On committed source `908e629b881836f04c3ccd4815a1bfd4cf7e2a2c`, both 35-row
source-bound supervisor smokes reproduced their retained Python and Rust output
hashes exactly. The full 3,003-row candidate captures then ran sequentially
against the original retained Python captures. The supervisor checked source,
protocol, artifact, prefix, binary and cuDNN library hashes before and after.
Each mode matched all 939,939 codes and 192,192 raw IDs; there were no code or
ID differences. The base candidate output was byte-identical to the previously
scored base Rust capture, so its previously independently replayed scores apply.

The CFG candidate output SHA-256 is
`ff04571bfc72be02aee45ef7f64b6b4226431dca2264a7fd02d30be44294ef0f`,
its comparison SHA-256 is
`36aab40106bf6054264c833beaa636b4d5dc6c605822186e720915915abffe65`,
and its source-bound manifest SHA-256 is
`6d28b458bd91da7013836dc4b37048e234d959204add087d8bd4856942a6157f`.
Scoring the CFG candidate under the unchanged protocol passes both held-out
and full-set rules; the score and independent offline replay are byte-identical
with SHA-256 `915cae3bf2351db1f90bd290ebbe79353a4b9bdb91a67cb494d30791403f89bc`.
The full-set values remain BLEU 22.9464, lowercase BLEU 23.8358, chrF 51.5160
and length ratio 0.9587. All paired metric differences and bootstrap intervals
are zero because all decoded hypotheses match exactly. The base full-set values
remain BLEU 21.7202, lowercase BLEU 22.6121, chrF 49.5649 and length ratio
0.9374; its manifest SHA-256 is
`0692c0c03c7506e85c06c2439f50c366f6af4ddcce02826d6affb6a08ad1b5a4`.

The CFG capture command was:

```sh
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/bench/capture_wmt14_reference_candidate.py --mode cfg1 --head 908e629b881836f04c3ccd4815a1bfd4cf7e2a2c --binary-sha256 b44831af313733364f2b1a6c48deff489480edafe209f6575523451208b232ee --protocol-sha256 26f5f13aeb440da2f95f56091a482e221fa0c311934494a334566d9bc5ff8de1 --rows 3003 --output reports/phase4-wmt14-reference-cfg-full-v1
```

The base used `--mode base`, binary SHA-256
`ef0a62401a64dc36afb2bd289e569a1016e67037b144426d5c7a90254dad63ca` and
output `reports/phase4-wmt14-reference-base-full-v1`. Score commands use the
frozen `wmt14_score_heldout.py` and the checked comparison receipts. Full
capture/score files and logs are ignored under `reports/`. Desktop GPU
processes were present. These results do not establish a performance or Metal
gate, and the separately unresolved XSum Metal raw-ID gate remains visible.
