# Phase 2 reference-contract audit (no Rust codec result)

This establishes new fixtures and budgets **before Rust codec implementation**.
Phase 2 is in progress, Linux portion; neither runtime implementation gate passes
on the strength of this oracle-only report. See [the contract](../../PHASE2.md).

## Captures and identity

- Upstream `c70274273f489abc73832b2b98584a8483b6b203`, clean checkout; existing pinned
  Sudoku canonical files are authenticated against the committed Phase 1 inventory.
- Torch 2.15.0+cu132, Python 3.14.9, FP32, one CPU thread, seed 42, TF32 off,
  deterministic cuDNN, math SDPA, eval/inference mode. CUDA: RTX 3060 Ti 8 GiB,
  driver 615.78.08. No upstream source or math patches.
- Tiny untrained width-16 / length-16 codec: **236 traces across two cases**;
  multi-stage `[4,4]` learned cascade and GroupNorm enabled. The portable CPU
  fixture is `tests/fixtures/codec-v1.json` (170 interned typed arrays, ~608 KiB).
- Trained Sudoku-easy: **642 traces across three cases**, solved batch 2, blank
  batch 1, mixed-clue batch 2. Each includes free-running encoding/reconstruction
  and fixed-code decoding, intermediate module states and cumulative snapshots.
- Every snapshot is independently owned. Captured shadow decode sums match the
  unmodified decoder's latent input exactly, and encoding/decoding cumulative
  snapshots agree exactly. Neither broken `reconstruct` nor aliased `idx_to_zhats`
  is used. The encoder's direct `encode` codes also match the traced path.
- Two fresh generations of each canonical oracle produced identical trace-file
  hashes. The original Phase 1 input/pre-quant/code/logit tensors for the solved
  batch remain byte-exact; no established golden has been replaced.

Canonical trace SHA-256:

| Oracle | SHA-256 |
| --- | --- |
| Tiny CPU | `5c33cf7b4c89ac185518b30de476df87408927c3b1ae668f189ae23fff1577aa` |
| Sudoku CUDA | `89bee3830bf97485183a17d325f78183bcc6d5d573b2af312899b2ee9dfad7fa` |

Large traces (~92 MiB for Sudoku) stay ignored locally under
`reports/phase2-oracle/v1/`; durable archive publication is part of phase closure.
Small provenance/inventory reports: [tiny](raw/tiny-cpu.json),
[Sudoku](raw/sudoku-cuda.json). The latter has two near-margin code decisions in
`mixed_batch2` (levels 1 and 2); they are retained, not filtered from the suite.

## Cross-device oracle audit — failure retained

| PyTorch CPU vs CUDA | Traces | Failing tensors | Max absolute error: component / distance / accumulated |
| --- | ---: | ---: | --- |
| Tiny | 236 | 0 | 1.31e-6 / 1.53e-5 / 1.19e-6 |
| Sudoku | 642 | **1** | 0.001465 / 2.0 / 9.12e-5 |

The large-distance error passes its predeclared relative budget. The single failure
is one of 256 components in `blank_batch1/encode.level_00.quant_input`: channel 113,
CPU 4.46777105 vs CUDA 4.46671724, absolute error 0.00105381. This learned projection
has 22,784 product terms; their absolute sum for the CPU input is ~2267.99 while
the bias-adjusted F64 diagnostic result is ~4.46765312. Cancellation and preceding
FP32 differences matter. The F64 calculation is diagnostic only, **not a replacement
oracle or runtime precision change**. See [diagnostic](raw/long-dot-diagnostic.json).

All integer traces agree between CPU and CUDA, including these near-margin codes;
final logits/reconstructions satisfy their declared budgets. **No tolerance was
widened.** CPU is the canonical tiny oracle; CUDA is the canonical trained oracle
for both future GPU implementations. Full-checkpoint CPU compatibility is not
established, and this reference diagnostic cannot waive any candidate CUDA/MLX
failure against the canonical reference. Full comparisons, including relative
errors and zero-reference counts: [tiny](raw/tiny-cpu-vs-cuda.json),
[Sudoku](raw/sudoku-cpu-vs-cuda.json).

## Reproduction and tests

Use the existing Linux reference venv and verified bundle. Output directories/files
must be new; do not overwrite established goldens:

```sh
for profile in tiny sudoku; do
  for device in cpu cuda; do
    .reference/venv/bin/python tools/reference/codec_trace.py \
      --upstream .reference/upstream-ncm --bundle artifacts/canonical/sudoku-easy-qualified \
      --profile "$profile" --device "$device" --output "reports/codec-new/$profile-$device"
  done
done
cmp tests/fixtures/codec-v1.json reports/codec-new/tiny-cpu/fixture.json
.reference/venv/bin/python tools/reference/compare_codec_traces.py \
  reports/codec-new/tiny-cpu reports/codec-new/tiny-cuda reports/new-tiny-comparison.json
# Explicit diagnostic mode retains all_passed=false for the known CPU/CUDA difference.
.reference/venv/bin/python tools/reference/compare_codec_traces.py \
  reports/codec-new/sudoku-cpu reports/codec-new/sudoku-cuda reports/new-sudoku-comparison.json --diagnostic
.reference/venv/bin/python -m unittest discover -s tools/reference -p 'test_*.py' -v
```

Nine offline reference tests pass (four new policy/input/margin checks and five
existing exporter tests). All interned array keys, references and byte counts were
validated. Existing local checks still apply; no new Rust model implementation or
performance comparison is claimed. Instrumented oracle execution is deliberately
not used as a timing baseline; uninstrumented codec measurements remain pending.
