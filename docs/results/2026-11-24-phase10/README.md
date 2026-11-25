# Phase 10 Dual-Platform Report (CUDA & Metal)

Date: 2026-11-24 UTC
Plan: [docs/PHASE10.md](../../PHASE10.md)
Hardware:
- **Linux / CUDA:** NVIDIA GeForce RTX 3060 Ti (8 GiB, SM86, CUDA driver 620.73.04)
- **macOS / Metal:** Apple M1 Ultra (64 GiB unified memory, macOS 26.7.4, native MLX)

All three Phase 10 milestones (**10A**, **10B**, **10C**) passed on **both** Linux/CUDA and Apple Silicon/Metal with **zero new external crates** and **100% byte-identical `FP32` defaults** preserved across all 9 qualified inference profiles.

## Raw Receipts

| Milestone | Linux / CUDA | Apple Silicon / Metal |
| --- | --- | --- |
| **10A** — Pure-Rust `ncm pull` (Zero Python) | [cuda-milestone10a-v1.json](cuda-milestone10a-v1.json) | [mlx-milestone10a-v1.json](mlx-milestone10a-v1.json) |
| **10B** — Fused Attention (CUDA FlashAttention-2 + Metal Fast SDPA) | [cuda-milestone10b-v1.json](cuda-milestone10b-v1.json) | [mlx-milestone10b-v1.json](mlx-milestone10b-v1.json) |
| **10C** — Multi-Profile LoRA + Unmerged Adapters + Hot-Swappable `ncm serve` | [cuda-milestone10c-v1.json](cuda-milestone10c-v1.json) | [mlx-milestone10c-v1.json](mlx-milestone10c-v1.json) |

---

## Milestone 10A: Pure-Rust `ncm pull` (Zero-Python Single Binary)

Replaced the Python/PyTorch export requirement in `ncm pull` with a bounded, streaming pure-Rust implementation (`crates/ncm-io/src/pytorch.rs`, `crates/ncm-io/src/yaml.rs`, `crates/ncm-tokenizers/src/convert.rs`, `crates/ncm-cli/src/pull.rs`). Legacy Python export remains opt-in via `--python-legacy`.

### Verified Zero-Python Exports

| Model | Profile | Platform | Env Constraint | Export Time | Bit-Exact Tensors | Inference Check |
| --- | --- | --- | --- | ---: | ---: | --- |
| `sudoku-easy` | `sudoku-easy-inspection-v1` | Linux/CUDA | `PATH=/nonexistent` | 0.54 s | **238 / 238** | Byte-identical CUDA JSON |
| `tinystories-legacy` | `tinystories-inference-v1` | Linux/CUDA | `PATH=/nonexistent` | 1.87 s | **304 / 304** | Byte-identical CUDA decode |
| `xsum` | `xsum-inference-v1` | Linux/CUDA | `PATH=/nonexistent` | 2.17 s | **275 / 275** | Fluent CUDA decode (v1 analytic RoPE) |
| `wmt14` | `wmt14-inference-v1` | Linux/CUDA | `PATH=/nonexistent` | 4.01 s | **282 / 282** | Byte-identical CUDA decode |
| `lm1b-small` | `lm1b-small-inference-v2` | Linux/CUDA | `PATH=/nonexistent` | 3.19 s | **351 / 351** | Byte-identical CUDA (`v2 == v1 + sidecar`) |
| `sudoku-easy` (live HTTPS) | `sudoku-easy-inspection-v1` | macOS/Metal | `PATH` = `curl` only | 5.17 s | **238 / 238** | Byte-identical Metal JSON |
| `tinystories` 15k EMA (live HTTPS) | `tinystories-inference-v1` | macOS/Metal | `PATH` = `curl` only | 14.95 s | **304 / 304** | Byte-identical Metal decode |

All 4 canonical tokenizers (`tinystories`, `gpt2`, `qwen3`, `bert`) are reproduced bit-for-bit in pure Rust against `tokenizers-v1.lock.json` on both platforms (**1,450 / 1,450** bit-exact tensors verified on Linux/CUDA and **542 / 542** live HTTPS tensors verified on Apple Silicon/Metal).

---

## Milestone 10B: Fused Attention (CUDA SM80+ Tensor-Core FlashAttention-2 & Metal Fast SDPA)

- **CUDA (`crates/ncm-backend-cuda/src/fused_attention.cu`):** Custom Ampere+ Tensor-Core FlashAttention-2 kernel (`mma.sync.aligned.m16n8k16.row.col.f32.{bf16,f16}`, `BR=64`, `BC=32`, `head_dim=64`, FP32 online `exp2f` softmax, zero $O(S^2)$ VRAM materialization, zero-branch `MaskMode` specialization).
- **Metal (`crates/ncm-backend-mlx/src/codec.rs`):** Native `mlx_rs::fast::scaled_dot_product_attention` wired for non-FP32 precisions (`bf16`/`fp16`/`q8`/`q4`).
- **Strict FP32 parity:** Both GPUs passed **9 / 9 FP32 CLI byte-equivalence** checks untouched, plus 6-shape numerical accuracy suites (`cuda_fused_attention_d64_matches_fp32_across_shapes_and_masks` and `mlx_fast_sdpa_matches_fp32_across_shapes_and_masks`).

### Paired Before/After Highlights

#### Linux / CUDA (RTX 3060 Ti)

| Profile | Batch | Precision | Before (ms) | After (ms) | Speedup | Peak VRAM Δ |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `xsum` | 8 | `bf16` | 130.25 | **121.75** | **1.070×** | **-36.0 MiB** |
| `xsum` | 8 | `fp16` | 130.82 | **123.21** | **1.062×** | **-36.0 MiB** |
| `xsum` | 1 | `bf16` | 30.39 | **30.33** | 1.002× | **-4.5 MiB** |
| `tinystories-15k-ema` | 8 | `bf16` | 105.13 | **100.89** | **1.042×** | -1.2 MiB |
| `tinystories-15k-ema` | 1 | `bf16` | 27.59 | **26.67** | **1.034×** | 0.0 MiB |
| `wmt14` | 8 | `fp16` | 79.03 | **76.17** | **1.038×** | -0.8 MiB |
| `lm1b-large` | 8 | `bf16` | 567.54 | **548.06** | **1.036×** | -2.1 MiB |
| `lm1b-large` | 16 | `q4` | 1,181.61 | **1,144.48** | **1.032×** | -4.2 MiB |

Quality checks (CUDA): `sudoku-easy` **128/128** (`bf16`/`fp16`), `xsum fp16` **28/32 exact (97.20% tok)**, `wmt14 fp16` **26/32 exact (94.92% tok)**.

#### Apple Silicon / Metal (M1 Ultra)

| Profile | Batch | Precision | Before (ms) | After (ms) | Speedup | MLX Peak Δ |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `xsum` | 1 | `bf16` | 80.54 | **76.29** | **1.056×** | **-63.1 MiB** |
| `xsum` | 1 | `fp16` | 64.27 | **62.79** | **1.024×** | **-63.1 MiB** |
| `wmt14` | 1 | `fp16` | 58.58 | **57.18** | **1.024×** | -3.7 MiB |
| `wmt14` | 8 | `fp16` | 132.38 | **129.31** | **1.024×** | -29.8 MiB |
| `lm1b-large` | 1 | `fp16` | 302.12 | **296.86** | **1.018×** | -0.4 MiB |

Quality checks (Metal): `sudoku-easy` **128/128** (`bf16`/`fp16`/`q8`), `wmt14 fp16` **28/32 exact (98.58% tok)**, `xsum fp16` **26/32 exact (92.97% tok)**.

---

## Milestone 10C: `ncm-train` v3 (Multi-Profile LoRA + Unmerged Adapters + Multi-Adapter `ncm serve`)

- **Verified unmerged adapter format (`ncm-lora-adapter-v1`):** `adapter.safetensors` + `adapter_manifest.json` pinned to `base_manifest_sha256` with per-tensor SHA-256 verification and fail-closed tamper/mismatch/symlink/non-finite rejection (`crates/ncm-io/src/lora_adapter.rs`).
- **Runtime adapter injection over any base precision (`fp32`/`bf16`/`fp16`/`q8`/`q4`):** `RuntimeLoraAdapter<T>` pre-scales `(alpha/rank) * B` once on host upload and applies the low-rank residual across all 5 text families (`LocalNcmModel`, `TinyStoriesTeacher`, `XsumTeacher`, `Wmt14Teacher`, `Lm1bSmallTeacher`) without altering strict no-adapter FP32 inference (`crates/ncm-model/src/lora.rs`).
- **Multi-profile `--init-from`:** extended `ncm train` beyond `local-ncm-v1` and `tinystories-inference-v1` to `wmt14-inference-v1`, `xsum-inference-v1`/`v2`, and `lm1b-{small,medium,large}-inference-{v1,v2}`, plus a native BF16 transformer training tape on CUDA so 560-token `xsum` trains comfortably inside 3.69 GiB free VRAM.
- **Hot-swappable multi-adapter `ncm serve`:** `--adapter ALIAS=BASE:DIR` mounts multiple adapters on a single shared resident GPU base (`--max-loaded 1`) with instant refcounted `Arc` hot-swaps and zero base reloads.

### Multi-Profile LoRA `--save-adapter` Convergence (Rank 8, `1,179,648` Trainable Params, `4.53 MiB` Adapter)

| Adapter / Profile | Compression vs Base | CUDA Loss (`bf16`, RTX 3060 Ti) | CUDA ms/step | Metal Loss (`fp32`, M1 Ultra) | Metal ms/step |
| --- | ---: | --- | ---: | --- | ---: |
| `tinystories_pirate` (15 steps, b=2) | **121.7×** | `6.3959 → 2.4755` | 554.6 | `6.3959 → 2.3954` | 348.2 |
| `tinystories_scifi` (15 steps, b=2) | **121.7×** | `6.6267 → 2.7907` | 539.4 | `6.6275 → 2.7655` | 348.5 |
| `wmt14` (10 steps, b=1) | **240.0×** | `2.0860 → 1.5838` | 429.8 | `2.0916 → 1.8035` | 343.0 |
| `xsum` (10 steps, b=1) | **138.4×** | `3.5707 → 2.2976` | 1,395.7 | `3.5705 → 2.2919` | 640.7 |
| `lm1b_small` (10 steps, b=1) | **208.0×** | `9.3540 → 3.2713` | 452.1 | `9.3541 → 3.3292` | 303.7 |

### Live `ncm serve` Multi-Adapter Hot-Swap over One Shared `Q8` Base (`--max-loaded 1`)

| Metric | Linux / CUDA (RTX 3060 Ti) | Apple Silicon / Metal (M1 Ultra) |
| --- | ---: | ---: |
| `/api/ps` resident models | **1** (`tinystories`, 5 reqs) | **1** (`tinystories`, 5 reqs) |
| Base reloads (`load_duration > 0`) | **0** | **0** |
| First adapter upload (`pirate` / `scifi`) | 60.84 ms / 59.17 ms | 136.83 ms / 137.33 ms |
| Base after hot-swap (bit-exact restoration) | **44.77 ms** (bit-exact) | **81.72 ms** (bit-exact) |
| Cached adapter swap (`pirate`) | **48.02 ms** (bit-exact) | **91.79 ms** (bit-exact) |
| Mismatched-base negative check | **Rejected fail-closed** | **Rejected fail-closed** |

Unmerged-vs-merged parity (`lora_unmerged_adapter_matches_merged_inference_and_rejects_tamper` in `crates/ncm-train/tests/train_suite.rs`) verified exact greedy code/ID match, `< 1e-4` logits diff, clean base restoration on detach, and fail-closed rejection on base mismatch and 1-byte tamper.
