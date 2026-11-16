# XSum full Python CUDA reference capture

The pinned-Python CUDA generator completed both stride-two shards of the
11,334-row XSum dataset with seed 42, batch size 32 and the frozen evaluation
configuration. Each shard has 178 saved batches, including a final three-row
batch. The source-bound scorer's batch validation read all 356 receipts and
confirmed every row index 0–11,333, code and ID shape and structurally valid
saved RNG state;
none of the generated IDs were outside the GPT-2 vocabulary and padding range.
This is reference generation evidence, not a scored quality result or a Rust
comparison. The [development protocol and frozen rule](quality-development-protocol-v1.md)
remain unchanged.

The ignored shard directories are `reports/phase4-xsum-quality-python-smoke-s0/`
and `reports/phase4-xsum-quality-python-dev-s1/`. Their manifest SHA-256 values
are `16314aae2367479046e3bc80e78a5dd739cb65ddfe575c38fee40a12541441a4` and
`ee4b9cae4d11c7f6865c070db3d13169611aa054a25ac66ca896a23c624a2d91`.
An ordered batch inventory digest was computed by concatenating each sorted
batch filename, a zero byte and the binary SHA-256 digest of that file, then
SHA-256 hashing the result. The two inventory digests are
`86b9336f3fa44ac0b689af364b9a31f65fea1dadfde918522475bf577a2882a6` and
`0090aecc24986d7f476516bb9036262f7b692de57462699cc047b0779af94f57`.

The full runs resumed from four previously validated development batches in
each directory:

```sh
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/reference/xsum_quality_generate.py --upstream .reference/upstream-ncm --source artifacts/upstream/xsum-phase4 --data artifacts/canonical/text-phase4-v1/xsum.json --output reports/phase4-xsum-quality-python-smoke-s0 --shard 0 --seed 42 --stop 11334 > reports/phase4-xsum-quality-python-full-s0.log 2>&1
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/reference/xsum_quality_generate.py --upstream .reference/upstream-ncm --source artifacts/upstream/xsum-phase4 --data artifacts/canonical/text-phase4-v1/xsum.json --output reports/phase4-xsum-quality-python-dev-s1 --shard 1 --seed 42 --stop 11334 > reports/phase4-xsum-quality-python-full-s1.log 2>&1
PYTHONPATH=tools/reference .reference/venv/bin/python -c 'from pathlib import Path; import xsum_quality_score as q; rows,m=q.checked_rows([Path("reports/phase4-xsum-quality-python-smoke-s0"),Path("reports/phase4-xsum-quality-python-dev-s1")],11334); print("validated rows",len(rows),"range",rows[0]["index"],rows[-1]["index"],"invalid tokenizer IDs",sum(t>=50258 for r in rows for t in r["raw_ids"]))'
```

The CPU-only reference ROUGE pass on all 11,334 rows returned ROUGE-1/2/L of
34.9236/12.7078/27.9258 and a mean of 20.6573 words. Restricting the same
per-row scores to held-out indices 256–11,333 yields 34.9312/12.7069/27.9256
and 20.6512 words on 11,078 rows. The pinned GPT-2 perplexity for this full
reference has not yet been run; these are reference metrics, not a Rust
comparison or a complete quality gate. The output SHA-256 is
`9e8c211611890833aab2cd4582ffbce13decdac26de92b53e5eab9f86fc3f76d`.

```sh
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/reference/xsum_quality_score.py --shard0 reports/phase4-xsum-quality-python-smoke-s0 --shard1 reports/phase4-xsum-quality-python-dev-s1 --data artifacts/canonical/text-phase4-v1/xsum.json --upstream .reference/upstream-ncm --output reports/phase4-xsum-quality-python-full-rouge-v1.json --stop 11334 > reports/phase4-xsum-quality-python-full-rouge-v1.log 2>&1
python3 -c 'import json; x=json.load(open("reports/phase4-xsum-quality-python-full-rouge-v1.json")); z=x["rows"][256:]; print("full",x["metrics"]); print("heldout",{**{k:100*sum(r["rouge"][k] for r in z)/len(z) for k in ("rouge1","rouge2","rougeL")},"mean_words":sum(len(r["hypothesis"].split()) for r in z)/len(z)},"n",len(z))'
```

The Rust run, full and held-out perplexity scores, and comparison are pending.
Dataset, checkpoint and raw generated outputs remain outside Git.
