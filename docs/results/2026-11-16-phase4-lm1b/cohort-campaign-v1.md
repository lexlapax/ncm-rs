# LM1B-small 80-seed quality campaign

This is the approved quality gate from [the user decisions](user-decisions-v1.md#1-lm1b-quality-gate-the-fresh-80-seed-protocol-is-approved).
The earlier five-seed failure stays on record in [the held-out report](small-heldout-cuda-v1.md).

## Tracked tooling

The campaign tool previously existed only in ignored `reports/phase4-lm1b-cohort-campaign-v1/`.
It is now tracked:

- `tools/reference/lm1b_campaign/`: `campaign.py` (init, capture, score and summary stages),
  `compare.py`, `python_capture.py` and `lm1b_shared_forward_score.py`;
- the Rust runner `crates/ncm-backend-cuda/examples/lm1b_small_cohort_campaign.rs`. It uses the
  same CUDA backend, pinned rotary tables and fully device sampler as `ncm sample-lm1b-small ... cuda-cudnn`.

The scorer, margins, bootstrap and metric code are unchanged from the rehearsed draft:
`compare.py`, `python_capture.py`, `lm1b_shared_forward_score.py` and the runner source are
byte-identical to it. Only paths changed in
`campaign.py`: the runner is now the built example and its source is the tracked file, and both
hashes are pinned in each manifest. `init` also takes a seed registry, records its SHA-256, selects
the Python and Rust CUDA lists, and in prospective mode requires every registered list to be disjoint.

## Rehearsal with the tracked tool

The observed seeds 42 and 0 were regenerated and rescored on CUDA with the tracked tool and a fresh
build of the runner. Every per-seed record (perplexity, summed NLL, token count and all 256 entropy
values) and the comparison result are identical to the archived rehearsal. The runner binary differs
byte-wise from the archived one; its outputs do not.

A first attempt failed closed: the capture validator still looked up the old runner path, so the
Rust capture was rejected after generation. The lookup was fixed and the whole rehearsal rerun in a
new directory, because the manifest pins `campaign.py` itself.

| Item | SHA-256 |
| --- | --- |
| Rehearsal manifest | `00dc0e85008789e1101b8a3a554736f6946fe3b62b9e72f520fbbb76762567de` |
| Rehearsal summary | `57546248c8cd71ab233fe6128abba11aab792e09c59f2539b82bf72bbb07175b` |
| Runner binary | `8868ff94534cebe72e72ae874248a75d8bf1e44558d8add86982290244dfdedc` |
| `campaign.py` | `81accbf8edfb240b3d152437de460859d98248af5890423f9fcd4d29dda551bd` |

Raw outputs are in ignored `reports/phase4-lm1b-campaign-tracked-rehearsal-v2/`; the failed first
attempt is retained in `-v1/`.

## Checks

- `cargo clippy --locked --release -p ncm-backend-cuda --features lm1b-experimental,cuda-cudnn --example lm1b_small_cohort_campaign -- -D warnings` passed.
- `setpriv --inh-caps=-all env RUSTC_WRAPPER= CARGO_BUILD_JOBS=4 ./tools/check.sh` passed, including
  all 533 report tests (`reports/phase4-lm1b-campaign-tooling-check-v2.log`). An earlier run without
  `setpriv` failed three clock-identity tests. The tests correctly rejected the shell's inheritable
  `CAP_WAKE_ALARM`; that log is retained as `-v1`.

## Reproduce

From the repository root, after building the runner:

```sh
tool="$PWD/tools/reference/lm1b_campaign/campaign.py"
run="$PWD/reports/NEW-RUN-DIRECTORY"
tokens="$PWD/.reference/tokenizers-venv/lib/python3.14/site-packages"
lib="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib"
export PYTHONPATH="$PWD/tools/reference:$PWD/tools/reference/lm1b_campaign:$tokens" LD_LIBRARY_PATH="$lib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
py=.reference/venv/bin/python
sha=$(setpriv --inh-caps=-all $py "$tool" init "$run" --mode prospective \
  --seeds docs/results/2026-11-16-phase4-lm1b/cohort-seeds-v1.json)
for runtime in python rust; do
  setpriv --inh-caps=-all $py "$tool" capture "$run" --manifest-sha256 "$sha" --runtime $runtime
done
setpriv --inh-caps=-all $py "$tool" score "$run" --manifest-sha256 "$sha"
setpriv --inh-caps=-all $py "$tool" summary "$run" --manifest-sha256 "$sha"
```

For a rehearsal, use `--mode rehearsal` with an observed-seed file such as
`reports/phase4-lm1b-cohort-campaign-v1/observed-seeds.json`. The summary stage exits 1 when a
prospective comparison fails.

For later CUDA repeats of the already observed approved 80-seed cohort, use `--mode retest` with
`docs/results/2026-11-16-phase4-lm1b/cohort-seeds-v1.json`. This mode is explicitly **nonblind**,
requires the original registry SHA-256 and exact ordered CUDA seed lists, preserves the scorer,
margins and bootstrap, and exits 1 if the comparison fails. It does not relabel the repeat as
prospective or authorize new cohorts. Retest mode currently rejects Metal rather than implying
that its import path has been requalified. The new mode extends the manifest's `mode` value; it
does not change earlier manifests or their recorded identities.

## Prospective CUDA result

The approved gate **passes** on CUDA. The campaign ran on 2026-11-19 from 05:31 to 06:15 UTC at
commit `90b494c`, using the pre-registered seed lists: Python 1000–1079 and Rust 2000–2079. Each
runtime produced 80 × 256 samples in batch-eight requests, and every seed was scored with the
pinned upstream FLM scorer (`upstream_exact` true for all 160 scores).

| Runtime | Seeds | Mean perplexity (seed SD) | Mean entropy, nats (seed SD) |
| --- | ---: | ---: | ---: |
| Python (pinned upstream, CUDA) | 80 | 93.9160 (2.3354) | 4.312720 (0.003564) |
| Rust (`sample-lm1b-small` CUDA path) | 80 | 94.3599 (2.2683) | 4.313447 (0.003390) |

| Check (Rust minus Python) | Point | 95% interval | Limit | Result |
| --- | ---: | --- | --- | --- |
| Perplexity | +0.443887 | [-0.271116, 1.145112] | at most 5.46 | pass |
| Entropy | +0.000727 | [-0.000325, 0.001795] | within ±0.0046 | pass |

The intervals use independent seed resampling (10,000 draws, RNG seed 42), as frozen. A second
`summary` invocation replayed the stored summary byte for byte. This qualifies LM1B-small CUDA
quality under the approved protocol. It does not erase the earlier five-seed failure, and it does
not qualify Metal, medium or large.

| Item | SHA-256 |
| --- | --- |
| Manifest ([copy](raw/cohort-cuda-v1-manifest.json)) | `c6f3e157ce86a0bf1b3dc6ee25f491ad2c3a87774d8f48be87a8565d1c54d85b` |
| Summary ([copy](raw/cohort-cuda-v1-summary.json)) | `b1dbedd9205ee9eb1dd645005bb9f3592fbbbbc86cb322dd458e4d9ceef5c511` |
| Python capture receipt | `2991e28de20959bbbe93d763a23fead4a6b813f72e64c75995466655ec08f09d` |
| Rust capture receipt | `a8274b7f5186a44df3191305c173b33b6df84d0be5694e30bc5f5abeb4976d2f` |
| Score receipt | `6992bfecc04844394601e001904818fb02501208349b564f25161b22e0d4ca62` |
| Runner binary | `8868ff94534cebe72e72ae874248a75d8bf1e44558d8add86982290244dfdedc` |

The raw captures and per-seed scores are in ignored `reports/phase4-lm1b-campaign-prospective-cuda-v1/`.

## Prospective Metal result

The approved gate **passes** on Metal. Rust Metal generated the pre-registered `rust_metal` list
(seeds 3000–3079) on the Mac Studio (M1 Ultra) with the default device FP32 sampler of
`ncm sample-lm1b-small ... mlx` at commit `0a3ce34`. The capture ran from 07:01:52 to 07:15:44 UTC on
2026-11-19. The comparison reuses the 80 published Python CUDA captures, imported by their receipt
hash. Scoring on Linux rescored those Python seeds identically to the CUDA campaign (80 of 80
records equal in perplexity, NLL, token count and every entropy value).

| Runtime | Seeds | Mean perplexity (seed SD) | Mean entropy, nats (seed SD) |
| --- | ---: | ---: | ---: |
| Python (pinned upstream, CUDA) | 80 | 93.9160 (2.3354) | 4.312720 (0.003564) |
| Rust Metal (device FP32 sampler) | 80 | 94.2083 (2.1679) | 4.313749 (0.003644) |

| Check (Rust Metal minus Python) | Point | 95% interval | Limit | Result |
| --- | ---: | --- | --- | --- |
| Perplexity | +0.292270 | [-0.412189, 0.970908] | at most 5.46 | pass |
| Entropy | +0.001029 | [-0.000070, 0.002125] | within ±0.0046 | pass |

| Item | SHA-256 |
| --- | --- |
| Manifest ([copy](raw/cohort-metal-v1-manifest.json)) | `c7ded8ccedd4d46346d6df361b779fc3eb01936229db2a285bebfb272420f60f` |
| Summary ([copy](raw/cohort-metal-v1-summary.json)) | `8beb69b4719e39f219306a3da768aabb76dc39213edbd4efbc99ea45b006c00f` |
| Metal runner binary | `966c80133b3b6c12f9df3d7016f9c87b26ef4e76eddbf277151fd3350baacda3` |
| Metal seed list bytes | `100c49c080217612cfce6d36fb7d6fbb47b936590c0a95ee7d54941216ad4f13` |
| Python capture receipt (imported) | `84afd445d3a3f584f72187f131938d138d8c006d4ea3371c8e6d6fb67aca4606` |
| Rust Metal capture receipt | `4635deff53a43b6c07c7bbaa2d073e0e98213f53154c617782830c48f72d06f5` |
| Score receipt | `a407003225b638bf5c384b401e1d72ef48669a176c4e193b4c022d23faa26d43` |

Raw captures are in ignored `reports/phase4-lm1b-campaign-metal-capture-v1/` (Mac and Linux) and
`reports/phase4-lm1b-campaign-prospective-metal-v1/` (Linux).

## Medium and large

The campaign takes `--size small|medium|large` (default small). The size selects the artifact,
its manifest pin, and the pinned upstream generator and config. Both Rust runners derive the size from the
artifact manifest. Medium and large reuse the pre-registered lists in
[cohort-seeds-v1.json](cohort-seeds-v1.json). No medium or large sample had been generated with any
of those seeds before the prospective runs. The only earlier medium samples are a two-seed rehearsal on
observed seeds 42 and 0. The approved scorer, margins and bootstrap are unchanged. After the
generalization, a small rehearsal on seeds 42 and 0 still reproduces the archived records exactly. A medium
rehearsal on the same seeds completed every stage (raw outputs in
`reports/phase4-lm1b-campaign-sizes-rehearsal-{small,medium}-v1/`).

## Prospective CUDA results: medium and large

The approved gate **passes** on CUDA for both sizes. Each campaign ran the same pre-registered seeds
(Python 1000–1079, Rust 2000–2079) with `--size medium` or `--size large`. Each runtime produced
80 × 256 samples in batch-eight requests, and the pinned FLM scorer reported `upstream_exact` for all
160 scores per size. Medium ran from 08:46 to 10:00 UTC on 2026-11-19 (manifest recorded at `bfe631f`).
Large ran from 10:00 to 12:08 UTC (manifest at `d5f2de3`, which changed only Metal tooling). Both
used the same CUDA runner binary.

| Size | Runtime | Mean perplexity (seed SD) | Mean entropy, nats (seed SD) |
| --- | --- | ---: | ---: |
| medium | Python (pinned upstream, CUDA) | 88.7748 (1.8059) | 4.339916 (0.003322) |
| medium | Rust (`sample-lm1b` CUDA path) | 88.8432 (1.7339) | 4.340177 (0.003814) |
| large | Python (pinned upstream, CUDA) | 81.1847 (1.6057) | 4.338293 (0.003821) |
| large | Rust (`sample-lm1b` CUDA path) | 81.5003 (1.6171) | 4.339096 (0.003777) |

| Size | Check (Rust minus Python) | Point | 95% interval | Limit | Result |
| --- | --- | ---: | --- | --- | --- |
| medium | Perplexity | +0.068388 | [-0.481525, 0.614443] | at most 5.46 | pass |
| medium | Entropy | +0.000261 | [-0.000839, 0.001368] | within ±0.0046 | pass |
| large | Perplexity | +0.315610 | [-0.182557, 0.805486] | at most 5.46 | pass |
| large | Entropy | +0.000803 | [-0.000403, 0.001989] | within ±0.0046 | pass |

The intervals use the frozen bootstrap (10,000 independent seed resamples, RNG seed 42). A second
`summary` invocation replayed each stored summary byte for byte. This qualifies LM1B-medium and
LM1B-large CUDA quality under the approved protocol; Metal for these sizes is separate.

| Item | SHA-256 |
| --- | --- |
| Medium manifest ([copy](raw/cohort-medium-cuda-v1-manifest.json)) | `d21f0a5f2f8c1eba5a5814a6c86c4dca969bba7c2ac20ddf4f59a4b7ab30a785` |
| Medium summary ([copy](raw/cohort-medium-cuda-v1-summary.json)) | `414212d9b5b98b3a95f8919174202fa185accd1c209bf7032aa23c0b60cce0ef` |
| Medium Python / Rust / score receipts | `ac9f61e4…`, `bc9dac7a…`, `b80313f8…` |
| Large manifest ([copy](raw/cohort-large-cuda-v1-manifest.json)) | `8641fa85a17cd4da941495e60b87b246e3d6b3c1ec5ceeeedc4441bc39d7076f` |
| Large summary ([copy](raw/cohort-large-cuda-v1-summary.json)) | `f6bd134c68c597b68cec5c640b8c4afc491dc65d601bfb3803ec31a5e3ec0a8e` |
| Large Python / Rust / score receipts | `4af4e29e…`, `b79115d6…`, `ab3f2a16…` |
| CUDA runner binary | `d2bbfc56612e8cb319bf283a8a7e10934ca334a70015a256ae94a99616c6206e` |

Raw captures and per-seed scores are in ignored `reports/phase4-lm1b-campaign-prospective-{medium,large}-cuda-v1/`.
The commands are those of the small run with `--size medium` or `--size large` on `init`.

## Prospective Metal results: medium and large

The approved gate **passes** on Metal for both sizes. Rust Metal generated the pre-registered
`rust_metal` list (seeds 3000–3079, the same seed-list bytes as the small Metal run) on the Mac Studio
with the default device FP32 sampler at commit `30349ae`. The captures ran from 14:22 to 14:54 UTC
(medium) and 14:54 to 15:49 UTC (large) on 2026-11-19. Each comparison imports the size's published
Python CUDA captures by receipt hash. Scoring on Linux rescored those Python seeds identically to the
CUDA campaigns (all 80 records equal at each size), all 160 scores per size are `upstream_exact`, and a
second `summary` invocation replayed each stored summary byte for byte.

| Size | Runtime | Mean perplexity (seed SD) | Mean entropy, nats (seed SD) |
| --- | --- | ---: | ---: |
| medium | Python (pinned upstream, CUDA) | 88.7748 (1.8059) | 4.339916 (0.003322) |
| medium | Rust Metal (device FP32 sampler) | 88.9431 (1.7587) | 4.340598 (0.003443) |
| large | Python (pinned upstream, CUDA) | 81.1847 (1.6057) | 4.338293 (0.003821) |
| large | Rust Metal (device FP32 sampler) | 81.2897 (1.6322) | 4.339164 (0.003801) |

| Size | Check (Rust Metal minus Python) | Point | 95% interval | Limit | Result |
| --- | --- | ---: | --- | --- | --- |
| medium | Perplexity | +0.168205 | [-0.385331, 0.706374] | at most 5.46 | pass |
| medium | Entropy | +0.000683 | [-0.000358, 0.001706] | within ±0.0046 | pass |
| large | Perplexity | +0.104994 | [-0.388800, 0.602007] | at most 5.46 | pass |
| large | Entropy | +0.000871 | [-0.000300, 0.002049] | within ±0.0046 | pass |

| Item | SHA-256 |
| --- | --- |
| Medium manifest ([copy](raw/cohort-medium-metal-v1-manifest.json)) | `e53dbba6b1648fe24f7bd6b0446dc1e1f65bd046be772550b7af9037eb86fa1f` |
| Medium summary ([copy](raw/cohort-medium-metal-v1-summary.json)) | `4780f52921fb0241891d098a2f8b4ff85b021b0bb3adb1ed3d301181c5613e16` |
| Medium Python (imported) / Rust Metal / score receipts | `753c5b7c…`, `ec25a2f3…`, `00e02d6a…` |
| Large manifest ([copy](raw/cohort-large-metal-v1-manifest.json)) | `6dd141b1b6b2569bf62353a3a366b36278b05638b552145ead929fd8e22ea08e` |
| Large summary ([copy](raw/cohort-large-metal-v1-summary.json)) | `2adb0931af45b3b3ad17133253b3b73b57e781b1d2d91ba15f2c2fd7fc748eb0` |
| Large Python (imported) / Rust Metal / score receipts | `1d59cdf8…`, `5572e391…`, `71cae1e8…` |
| Metal runner binary | `4c4ebd76f8c53c60d8d9d321fc118c066d7a3c7e7afbfedd5f0a277161bc5e40` |

Raw captures are in ignored `reports/phase4-lm1b-campaign-metal-capture-{medium,large}-v1/` (Mac and Linux) and
`reports/phase4-lm1b-campaign-prospective-{medium,large}-metal-v1/` (Linux). The Metal `init` adds
`--rust-device metal --size SIZE --metal-runner-sha256 HASH --python-from reports/phase4-lm1b-campaign-prospective-SIZE-cuda-v1/python --python-receipt-sha256 HASH`;
the Rust capture stage takes `--import` with the Mac capture directory.
