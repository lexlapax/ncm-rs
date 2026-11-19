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

## Prospective CUDA result

Pending.
