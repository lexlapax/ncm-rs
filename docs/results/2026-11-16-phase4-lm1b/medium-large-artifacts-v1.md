# LM1B-medium and LM1B-large inference artifacts

`tools/reference/lm1b_small_runtime_artifact.py` now takes `--size small|medium|large` (default
`small`). It exports the pinned checkpoint for that size with the shared tokenizer checkpoint and the
pinned BERT tokenizer assets into an `lm1b-SIZE-inference-v1` profile. Every tensor is kept, with
unchanged layouts and dtypes. Medium and large differ from small only in generator depth, heads and
width (24/16/1024 and 36/20/1280 against 12/12/768). Head size is 64 for all three, so the pinned
rotary tables apply unchanged.

| Size | Generator / codec / training / inference tensors | Largest file | Total | Manifest SHA-256 |
| --- | --- | ---: | ---: | --- |
| small | 213 / 100 / 36 / 2 | 425 MiB (codec) | ~977 MiB | `58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76` |
| medium | 381 / 100 / 36 / 2 | 1,209 MiB (generator) | ~1,781 MiB | `2fea4a252de0cecbe6ab12b2c4a0077cff66498e6344894694ba02876a38c30c` |
| large | 549 / 100 / 36 / 2 | 2,766 MiB (generator) | ~3,338 MiB | `86676fed0f9813b3ec293aca87133eca61fc609d10d620978375d12c0b03698a` |

Two independent exports of each of medium and large produced byte-identical manifests. Re-exporting
small with the generalized exporter reproduced every tensor, config and tokenizer file of the existing
small artifact byte for byte. Only `provenance.json` changes, because it records the exporter's own hash, and
with it the manifest. The existing small artifact remains the pinned one.

`ncm-io` accepts the two new profiles with separate budgets, set 5–8% above the measured exports:
medium 1,280 MiB per file and 1,920 MiB in total, large 2,944 MiB and 3,584 MiB. No size inherits another's
limits. The new `lm1b_sizes_have_separate_budgets` test loads each profile and checks its per-file and total
limits, and checks that an unknown size is rejected. The generalized CPU probe
(`crates/ncm-io/examples/lm1b_small_artifact_probe.rs`) loaded each real artifact in full and inference-only
modes:

| Artifact | Full load: time / peak RSS | Inference-only load: time / peak RSS |
| --- | --- | --- |
| small | 1.30 s / 949 MiB | 1.02 s / 804 MiB |
| medium | 2.32 s / 1,787 MiB | 1.95 s / 1,642 MiB |
| large | 4.01 s / 3,344 MiB | 3.72 s / 3,199 MiB |

This is checked loading only. Model support, parity, quality and performance for medium and large are pending.
Artifacts are in ignored `artifacts/canonical/lm1b-{medium,large}-inference-v1/`.

Reproduce from the repository root:

```sh
PYTHONPATH=tools/reference .reference/venv/bin/python tools/reference/lm1b_small_runtime_artifact.py \
  --size medium --source artifacts/upstream/lm1b-phase4 \
  --tokenizer-asset artifacts/canonical/tokenizers-qualified/bert \
  --output artifacts/canonical/lm1b-medium-inference-v1
cargo run --locked --release -p ncm-io --example lm1b_small_artifact_probe -- \
  artifacts/canonical/lm1b-medium-inference-v1 inference
```
