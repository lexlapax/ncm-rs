# LM1B CUDA CLI cohort routing retest

2026-11-20. For each LM1B size, all 256 rows of retained Rust cohort seed 2000 were rerun through the pre-change `sample-lm1b` alias and current unified `sample` command on CUDA. Both used the retained artifact, explicit pinned rotary table and 32 batches of eight, with the original request seeds `(2000 << 32) | batch`. For every batch, old and new CLI stdout matched byte-for-byte, and codes and IDs matched the pinned cohort capture: **768 rows total**. This links one previously observed seed per size to both CLI paths; it is not a rerun of all 80 seeds or a new blind quality test. The [small](cuda-lm1b-small-quality-retest-v1.md), [medium](cuda-lm1b-medium-quality-retest-v1.md) and [large](cuda-lm1b-large-quality-retest-v1.md) 80-seed CUDA quality retests remain separate.

The baseline and candidate binary SHA-256 values were `f10abaaec672dd7c0c62034972fb4b15fc204f2b1f5693e4a09f5e9640767098` and `6194111fe3860584492676c6d14f50124a77f8dcf18f316a959b8abd16a20278`. The runner verified their hashes, each original manifest and campaign receipt, each selected capture and the rotary table before and after; output shapes and ID bounds were checked against the captures. An independent local replay re-read all 192 CLI outputs, checked their bytes, request seeds, capture rows and summary hashes. The runner SHA-256 is `86562091eb039f31e7f9685decf758de87b724e262c797440a6439810527322e`.

| Size | Ignored summary SHA-256 |
| --- | --- |
| Small | `d092fc7a4c079e56270027a6940e6878976b799e34ac64eb8c3afa3bcfa4a9ef` |
| Medium | `b173a59cff2c4e10e1c91f168d4c600f03e55bfa61c263cca21f5f120509fcbe` |
| Large | `69bf02f63fa42f88a081d8b7b94b308535b1aaacdf3cb2e36fecad6e92059a74` |

Raw outputs and summaries remain local under `reports/phase5-lm1b-{small,medium,large}-cli-cohort-cuda-v1/`, not durably published. The desktop GPU was shared; this is not performance or Metal evidence. The runner's offline negative controls, small smoke and `setpriv --inh-caps=-all tools/check.sh` passed. No production code changed.

```sh
for size in small medium large; do
  setpriv --inh-caps=-all env OMP_NUM_THREADS=1 \
    LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" \
    python3 tools/bench/phase5_lm1b_cli_cohort.py --size "$size" \
    --baseline reports/phase5-cli-equivalence/baseline-ncm --candidate target/release/ncm \
    --artifacts artifacts/canonical --rope reports/phase4-lm1b-level7-replay-v1/reference-rope-tables.safetensors \
    --output "reports/NEW-lm1b-$size-cli-cohort"
done
```

The retained campaign captures and pinned binaries/artifacts must already be prepared as documented in the linked retests. The ordinary CLI default rotary sidecar, text input, other cohort seeds and Metal require their separately scoped evidence.
