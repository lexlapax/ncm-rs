# Sudoku-easy artifact and oracle milestone — 2026-11-10

**Partial Phase 1 evidence, not a supported inference release.** This change adds
an offline exporter, bounded Rust inspection loader, Sudoku character tokenizer,
and pinned Python model traces. It does not implement a Rust codec/generator.

| Check | Result |
| --- | --- |
| Original `.pt` → canonical safetensors | All 238 tensors byte-exact; no dtype/layout changes |
| Rust checked load → Candle CPU upload/readback | All 238 tensors byte-exact |
| Rust checked load → Candle CUDA upload/readback | All 238 tensors byte-exact on RTX 3060 Ti |
| Original vs canonical reload, upstream Python CUDA | All 56 captured tensors byte-exact across two synthetic boards and 12 generator levels |
| Rust Sudoku text tokenizer vs pinned upstream | Five golden encode/decode cases pass, including BOS/blank/whitespace |
| Existing CPU/CUDA operators | All 18 fixtures still pass |
| Load latency / peak memory / tokenizer throughput | **Pending**; no performance qualification or speedup claim |
| Corpus quality / Rust model parity | **Pending**; conversion identity is not task quality |
| Apple MLX | **Pending**; Linux phase gate still open |

There is no previous implemented Rust loader/tokenizer baseline. Python supplies
conversion and tokenization identity comparisons; meaningful load/memory and
throughput comparisons have not yet been measured. Per the reporting policy this
candidate remains **unqualified for release**. The previous operator performance
report is historical; the operators themselves were not changed here.

## Artifact scope and provenance

- Upstream NCM: `c70274273f489abc73832b2b98584a8483b6b203`.
- HF `hazyresearch/ncm-checkpoints`: `d342ce8aaea58830eef3b8ad473664673e41f692`.
- Only `sudoku/easy/config.yaml`, `tokenizer.pt`, and `generator.pt` were downloaded.
  The exporter checks exact byte sizes and SHA-256 before weights-only loading.
- 58 codec tensors, 144 generator tensors, and 36 separately retained EMA/count/hit
  buffers. No source tensor is dropped. No aliases or extra checkpoint metadata
  exist in this pinned pair. Unexpected metadata/sharing is rejected.
- [Manifest/inventory](raw/artifact-manifest.json) lists every persistent tensor's
  name, shape, dtype, bytes and hash. [Oracle report](raw/oracle.json) inventories
  nonpersistent buffers and every captured trace. Defaults remain tied to the
  pinned upstream source; this inspection profile is not the general runtime schema.
- No upstream source patches. Captures clone accumulated reconstructions rather
  than relying on the known aliased `idx_to_zhats` snapshot list. The broken
  `reconstruct` helper and cached generator path are not used.

## Workload and environment

Linux, RTX 3060 Ti 8 GiB, Candle 0.11.1, PyTorch 2.15.0+cu132, Python 3.14.9,
Rust 1.99.1. The reference uses FP32, TF32 disabled, math SDPA, deterministic
cuDNN, seed 42, eval/inference mode, uncached dense generation, temperature 0,
top-k 0, top-p 1, CFG 0. Batch 2; 89 target tokens, 91 synthetic prefix tokens,
322 generated concept IDs across 12 levels. Prefixes are deliberately synthetic,
not a claim about the official evaluation dataset's packing.

The 56 traces cover pre-quant activations, codec codes/logits, first-transformer-
block inputs/outputs and logits at every generation level, cloned cumulative
reconstructions, final logits and output IDs. No tolerance relaxation is involved:
original and converted weights must produce identical raw bytes. No Rust model
computation is compared by the tensor-upload probe.

[Environment/source hashes](environment.json) identify the tested uncommitted
candidate atop `f7aad49`; the exact source digests avoid a circular report/commit
identifier. Runtime code is in the commit containing this report. No benchmark
timing or GPU memory numbers were collected for this milestone.

## Reproduce and inspect

Use the [offline reference commands](../../../tools/reference/README.md), which
require fresh output paths and explicit pinned downloads. Run `tools/check.sh
--cuda` with the documented host compiler settings, plus the optional reference
venv exporter tests. Completed checks: formatting, Clippy (CPU and CUDA), 15 Rust
tests, rustdoc, four Python aggregation tests, five Python exporter tests, syntax,
18 real-GPU operator cases, artifact probes on both CPU and CUDA, and the model
trace comparison.

Compact raw results: [export](raw/export.json), [CPU](raw/candle-cpu.json),
[CUDA](raw/candle-cuda.json), [oracle](raw/oracle.json). The 14 MiB trace file and
weights are not in Git; regenerate using the pinned commands. Trace content hashes
are public, but a durable large-trace download asset is still pending. Export
provenance includes the converter's base commit, so a later regeneration can change
manifest/provenance digests without changing any tensor bytes.

Next: finish Linux load/performance/memory and runtime-contract checks, then
announce and perform the corresponding MLX phase on the available Mac. Both
platform gates are required before Phase 2.
