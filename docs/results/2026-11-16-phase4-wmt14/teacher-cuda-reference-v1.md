# WMT14 fixed-history CUDA reference

Two clean pinned-upstream captures used the first two independently verified
96-token WMT14 source prefixes. The upstream generator executed with eleven
deterministic forced code levels, no KV cache, CFG zero, TF32 disabled and math
SDPA. The trace contains all eleven generator head logits, all eleven codec
reconstruction levels, prefixes, forced codes, final decoder logits and greedy
IDs. Sampling and task quality are not evaluated. Source, checkpoint and prefix
hashes were checked before and after each run.

```sh
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python tools/reference/wmt14_teacher_trace.py --upstream .reference/upstream-ncm --source artifacts/upstream/wmt14-phase4 --prefixes reports/phase4-wmt14-tokenizer-reference-v1.json --output reports/phase4-wmt14-teacher-cuda-reference-v1
```

An independent run with output suffix `v2` produced the identical 26-tensor
payload SHA-256 `368d60d5960686faec87a00b73c9521c4dce15430e7748c33d0b611b3f6797e2`
and report SHA-256 `95cfc76db86032dfeedc73cbcff41b121644c1e7f767e84e937eb2afc95ffba8`.
The script SHA-256 is `5c5aefa4998844df46858b00fb3a29b1f2cf1ed4e050a45106c55033879672aa`.
The prefix bounds test passes in the pinned reference environment. GPU desktop
processes were present; this capture makes no timing claim. Reports and large
trace files remain ignored under `reports/`. No Rust generator comparison or
WMT14 quality, performance or Metal gate has yet passed.
