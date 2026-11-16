# XSum larger-batch capacity preparation

The XSum model validates batches up to 128, but the exploratory Rust and
Python timing examples previously refused anything above 32. Their explicit
input bounds now permit 1–128; this is a test-harness change, not evidence
that any larger batch fits in GPU memory or meets a performance gate. The
official XSum quality protocol remains at batch 32.

A local, ignored 128-row fixture uses the first 128 rows from the hash-checked
non-executable dataset and the same prefix truncation/padding as upstream. Its
first 32 rows exactly match the previously measured performance fixture. The
file `reports/phase4-xsum-capacity-inputs-v1/prefixes.json` has SHA-256
`96a215963a9acd6e7adb9af868a8d03ffd3110446bc3a5ac6434163a000345a2`.
The following preparation command created it:

```sh
.reference/venv/bin/python -c 'import hashlib,json,pathlib; p=pathlib.Path("artifacts/canonical/text-phase4-v1/xsum.json"); assert hashlib.sha256(p.read_bytes()).hexdigest()=="cba412c100c1264a7a107b2df3bacd20bff5846325ce48fd5aa353b4726daa6a"; d=json.loads(p.read_bytes()); rows=[(r[:512]+[50257]*512)[:512] for r in d["inputs"][:128]]; old=json.load(open("reports/phase4-xsum-real-inputs-v1/prefixes.json")); assert rows[:32]==old and len(rows)==128; out=pathlib.Path("reports/phase4-xsum-capacity-inputs-v1"); out.mkdir(); raw=(json.dumps(rows,separators=(",",":"))+"\n").encode(); (out/"prefixes.json").write_bytes(raw); print("rows",len(rows),"sha256",hashlib.sha256(raw).hexdigest())'
```

`python3 -m py_compile tools/reference/xsum_perf_screen.py`,
`cargo fmt --all -- --check`, and the strict-cuDNN example's targeted Clippy
check passed; the latter's output is
`reports/phase4-xsum-capacity-clippy-v1.log`. A real-device capacity screen is
pending after the current source-bound quality run; record each runtime's OOMs,
actual batch sizes, outputs, memory and measurement scope without equating
allocator counters. The prior 1/4/32 comparison remains tied to its original
source commit and is not replaced by this preparation.
