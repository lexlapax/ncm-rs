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

The Rust run, full and held-out scores, and comparison are pending. Dataset,
checkpoint and raw generated outputs remain outside Git.
