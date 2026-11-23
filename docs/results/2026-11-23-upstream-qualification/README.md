# Upstream Qualification Report: `HazyResearch/ncm@094f25ca` & `hazyresearch/ncm-checkpoints@e34c3fab`

Date: 2026-11-23 UTC
Policy: [docs/UPSTREAM.md](../../UPSTREAM.md)
Prior triage: [2026-11-23 Upstream Triage](../2026-11-23-upstream-triage/README.md)
Receipts: [cuda-upstream-094f25ca-v1.json](cuda-upstream-094f25ca-v1.json) (RTX 3060 Ti), [mlx-upstream-094f25ca-v1.json](mlx-upstream-094f25ca-v1.json) (Apple M1 Ultra)

## 1. Summary & Disposition

Both the Linux/CUDA and Apple Silicon/Metal qualification gates **PASSED** for upstream code commit `094f25ca5fd8fe2bc7f0ff77bf3f860c3a34e7e8` and checkpoint revision `e34c3fab451ddbe7cf474cc97c44ab23321f2219`.

- **Pins advanced:** [upstream.toml](../../../upstream.toml) now tracks `094f25ca5fd8fe2bc7f0ff77bf3f860c3a34e7e8` / `e34c3fab451ddbe7cf474cc97c44ab23321f2219` (`status = "qualified-dual-platform"`).
- **Backward compatibility preserved:** `ncm-io` retains explicit dual pin-pair validation admitting both the legacy pair (`c7027427` / `d342ce8a`) and the new pair (`094f25ca` / `e34c3fab`). Legacy `tinystories-phase4-v1` artifacts continue to load and generate on both CUDA and Metal.
- **Unchanged profiles:** Sudoku (`easy`/`medium`/`hard`), XSum, WMT14, and LM1B (`small`/`medium`/`large`) checkpoints are byte-for-byte identical between `d342ce8a` and `e34c3fab`. OpenWebText remains unreleased (`owt/TODO.md`), so `OwtConfigOnly` refusal remains active.
- **Exporter & quickstart tooling:** `tools/reference/tinystories_artifact.py --release 15k-ema`, `tools/reference/fetch_checkpoints.py --profile tinystories-15k-ema`, and `tools/reference/quickstart_export.py --model tinystories` (defaulting to 15k EMA; `--model tinystories-legacy` retains `d342ce8a`) verify exact sizes and SHA-256 digests.

---

## 2. Schema & Runtime Changes

Only the TinyStories 15k EMA release changed in `e34c3fab` (`config.yaml`, `generator.pt`, `tokenizer.pt`):

1. **`ncm-core` (`crates/ncm-core/src/config.rs`):** made training-only fields trimmed from upstream's inference-only `tinystories/config.yaml` (`dropout`, `pre_quant_dropout`, `commitment_cost`, `decay`, `epsilon`, `aux`) default explicitly while keeping strict Sudoku schema checks intact.
2. **`ncm-io` (`crates/ncm-io/src/lib.rs`):** added `UPSTREAM_2026_11_23` and `CHECKPOINT_REVISION_2026_11_23` as an explicitly reviewed pin pair alongside the legacy pair.
3. **`ncm-model` (`crates/ncm-model/src/generator.rs`):** admitted the trimmed `tinystories` 15k EMA inference generator config (`dropout: 0.0`, `ctx_encoder: {enabled: false}`) alongside the legacy Phase 4 config.
4. **FP16 safety boundary:** the 15k EMA tokenizer still contains extreme codebook weights (`max_abs_codebook = 4.37e10`, exceeding IEEE 754 FP16 `65504.0`). Both CUDA and Metal reject `--precision fp16` explicitly at load time and direct users to `bf16`, `q8`, `q4`, or `fp32`.

---

## 3. Correctness & Parity Evidence

Canonical artifact tested: `tinystories-15k-ema-v1` (`manifest.json` SHA-256 `a044b20c15c85a352d2760068e477df17b2deb6b240e10dc008d1960263c9b0e`, `bit_exact_roundtrip = true`).

| Check | Linux/CUDA (RTX 3060 Ti) | Apple Silicon/Metal (M1 Ultra) |
| --- | --- | --- |
| Codec `reconstruct` codes vs upstream Python `094f25ca` | **0 / 2,564 diffs** | **0 / 2,564 diffs** (vs CUDA) |
| Codec `reconstruct` IDs vs upstream Python `094f25ca` | **0 / 512 diffs** | **0 / 512 diffs** (vs CUDA) |
| Sample-decode IDs vs upstream Python `094f25ca` | **0 / 512 diffs** | Fluent decoded stories (`fp32`/`bf16`/`q8`/`q4`) |
| `--precision fp16` load-time range guard | **Rejected explicitly** (`4.37e10 > 65504.0`) | **Rejected explicitly** (`> 65504.0`) |
| Legacy `tinystories-phase4-v1` regression | **Passed** | **Passed** |
| Local validation gate | `./tools/check.sh` passed | `./tools/check.sh --mlx` + `ncm-cli --features mlx` Clippy/tests passed |

---

## 4. Performance & Memory Comparison (`tinystories-15k-ema-v1`, batch 1)

### Linux / CUDA (NVIDIA RTX 3060 Ti, 8 GiB)

| Runtime / Mode | Load (ms) | Warm Median (ms) | Warm p95 (ms) | Peak VRAM (MiB) | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Upstream Python `094f25ca` eager `fp32` | — | 43.22 | 46.50 | 644.9 | Reference baseline |
| Upstream Python `094f25ca` fast `bf16`+KV+resize | — | 50.57 | 50.94 | 647.9 | Eager fast flags (no CUDA graphs at b=1) |
| **Rust CUDA `fp32` (default)** | **272.4** | **41.14** | **41.72** | **492.9** | **1.05× faster & -23.6% VRAM vs Py `fp32`** |
| **Rust CUDA `bf16`** | **270.3** | **42.81** | **43.61** | **253.6** | **1.18× faster & -60.9% VRAM vs Py fast `bf16`** |
| **Rust CUDA `q8`** | 593.4 | 51.94 | 52.17 | **183.2** | **-62.8% VRAM vs Rust `fp32`** |
| **Rust CUDA `q4`** | 496.6 | 51.54 | 53.80 | **138.7** | **-71.9% VRAM vs Rust `fp32`** |

### Apple Silicon / Metal (Apple M1 Ultra, `macstudio`)

| Runtime / Mode | Load (ms) | Warm Median (ms) | Warm p95 (ms) | MLX Peak (MiB) | Host RSS Peak (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Rust MLX `fp32` (default)** | **295.0** | **90.40** | **91.16** | 606.6 | 527.0 |
| **Rust MLX `bf16`** | 353.4 | 97.31 | 97.59 | **359.1 (-40.8%)** | **93.8 (-82.2%)** |
| **Rust MLX `q8`** | 407.4 | 98.33 | 98.86 | **271.2 (-55.3%)** | **118.1 (-77.6%)** |
| **Rust MLX `q4`** | 414.8 | 100.08 | 100.60 | **226.7 (-62.6%)** | **94.2 (-82.1%)** |

---

## 5. Disclosures & Scope

- Measurements are batch-1 warm in-process timings (`phase5_stage_profile`, 3 warmups / 10 trials) on RTX 3060 Ti (Linux) and M1 Ultra (macOS); unified Apple Silicon memory (`MLX Peak` + `Host RSS`) is reported separately from discrete CUDA VRAM.
- Upstream Python's `torch.compile` + CUDA graph path (`HazyResearch/ncm@094f25ca`) was not enabled in this single-batch eager comparison; `py_094f25ca_fast_bf16_kv_resize` measures upstream's eager `bf16` + prefix KV cache + matrix resize path.
- Neither Rust runtime reproduces PyTorch's RNG stream (`sample` parity is verified via exact codec `reconstruct`, deterministic sample-decode IDs on fixed codes, and qualitative coherence across precisions).
