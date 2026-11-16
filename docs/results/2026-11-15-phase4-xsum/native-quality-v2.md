# XSum CUDA reference-order quality comparison

The source- and linked-library-bound `xsum-inference-v2` CUDA capture covers
all 11,334 pinned XSum rows. The comparison uses the previously frozen rule:
exact raw output IDs and no decrease in the recorded quality metrics. All
544,032 raw IDs match Python, as do decoded texts, every per-row ROUGE score,
and aggregate metrics. One of 2,266,800 intermediate codes differs; the
[real-CUDA investigation](native-full-progress-v1.md) identifies an exact
Python maximum tie at that location and confirms identical final IDs. This
explains the observed choice but does not establish general tie handling.

| Range | Rows | ROUGE-1 | ROUGE-2 | ROUGE-L | GPT-2 PPL | Scored tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 11,334 | 34.9236 | 12.7078 | 27.9258 | 82.6234 | 277,595 |
| Held out, indices 256–11,333 | 11,078 | 34.9312 | 12.7069 | 27.9256 | 82.6764 | 271,249 |

The table values are identical for Rust and Python. Mean output lengths are
20.6573 words for the full set and 20.6512 for the held-out set. The full and
held-out comparisons both passed the frozen rule; this is not the full XSum
runtime gate. Repeated performance, CFG and general tie behavior, other
required feature checks and MLX remain pending.

The full comparator output is
`reports/phase4-xsum-native-quality-comparison-full-v1.json` (SHA-256
`0dd221351ff0650867705d2aada076c15f4cab6e66482fa555cad093cab50a4d`);
the held-out result is
`reports/phase4-xsum-native-quality-comparison-heldout-v1.json` (SHA-256
`9e2f7fd21c881dfc89aa1321d121fc0d18b04394ff4682075b4a13b371a73a21`).
The four scored inputs and raw shard receipts remain under `reports/`, with
source, binary, artifact, CUDA library and scorer fingerprints in their
manifests. The pinned GPT-2 evaluator was run separately for each runtime and
range with `--with-ppl`; the model is evaluation-only.

For each range, score the Python shards
`reports/phase4-xsum-quality-python-smoke-s0` and
`reports/phase4-xsum-quality-python-dev-s1`, then the Rust shards
`reports/phase4-xsum-quality-native-verified-s0` and
`reports/phase4-xsum-quality-native-verified-s1`, using the same options:

```sh
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/reference/xsum_quality_score.py --shard0 SHARD0 --shard1 SHARD1 --data artifacts/canonical/text-phase4-v1/xsum.json --upstream .reference/upstream-ncm --output NEW_SCORE.json --start START --stop 11334 --with-ppl
.reference/venv/bin/python tools/reference/xsum_quality_compare.py --python PYTHON_SCORE.json --rust RUST_SCORE.json --output NEW_COMPARISON.json
```

Use `START=0` for the full set and `START=256` for held-out, and choose new
output paths under `reports/`; the scorer and comparator refuse replacement.
