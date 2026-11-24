# Phase 8 & Phase 9 Dual-Platform Report: Native Low-Precision Kernels, Verified Pre-Quantized Sidecars, and `ncm-train` v2 (LoRA + BF16 + JSONL)

Date: 2026-11-24
Plan: [docs/PHASE8.md](../../PHASE8.md)
Hardware:
- **Linux/CUDA:** NVIDIA GeForce RTX 3060 Ti (8 GiB VRAM, driver `620.73.04`, Arch Linux, shared desktop GPU; `~3.5–3.7 GiB` free VRAM during Phase 9 runs).
- **Apple Silicon/Metal:** Apple M1 Ultra (64 GiB unified memory, macOS `26.7.4`, `macstudio`, `rmlx serve` and `Ollama` idle/untouched).

All four milestones (**8A**, **8B**, **9A**, **9B**) passed full quality, parity, negative-control, and performance gates on **both** Linux/CUDA and Apple Silicon/Metal. Strict `FP32` defaults remain **100% byte-identical** across all 9 inference profiles on both GPUs.

---

## Receipts

| Milestone | Scope | Linux/CUDA receipt | Apple Silicon/Metal receipt |
| --- | --- | --- | --- |
| **8A** | Native low-precision kernels (`BF16`/`FP16`/`Q8`/`Q4`), fixed-matrix resampling GEMMs, CUDA-graph evaluation vs upstream Python `094f25ca` | [cuda-milestone8a-v1.json](../2026-11-23-phase8/cuda-milestone8a-v1.json) | [mlx-milestone8a-v1.json](../2026-11-23-phase8/mlx-milestone8a-v1.json) |
| **8B** | `ncm quantize` verified `Q8`/`Q4` pre-quantized sidecar cache (load speedup, byte-exact equivalence, fail-closed tamper controls) | [cuda-milestone8b-v1.json](../2026-11-23-phase8/cuda-milestone8b-v1.json) | [mlx-milestone8b-v1.json](../2026-11-23-phase8/mlx-milestone8b-v1.json) |
| **9A** | `ncm-train` v2: upstream `tinystories-15k-ema-v1` `--init-from` + native Rust LoRA (`--lora-rank`/`--lora-alpha`) + export merge & inference roundtrip | [cuda-milestone9a-v1.json](../2026-11-23-phase9/cuda-milestone9a-v1.json) | [mlx-milestone9a-v1.json](../2026-11-23-phase9/mlx-milestone9a-v1.json) |
| **9B** | Mixed-precision `BF16` training (`--train-precision bf16`), `--grad-accum N`, `.jsonl` datasets + sliding-window chunking, `--val-data` / `--eval-every` | [cuda-milestone9b-v1.json](../2026-11-23-phase9/cuda-milestone9b-v1.json) | [mlx-milestone9b-v1.json](../2026-11-23-phase9/mlx-milestone9b-v1.json) |

---

## Milestone 8A: Native Low-Precision Kernels & Fixed-Matrix Resampling

### Key changes
- **CUDA (`ncm-backend-cuda`):** eliminated per-layer FP32 casts in `bf16`/`fp16`/`q8`/`q4` via native `candle_nn` `layernorm_{bf16,f16}` (cached zero bias), `rmsnorm_{bf16,f16}`, `ugelu_erf_{bf16,f16}`, `rope_{bf16,f16}`, native `BF16` activations into `QMatMul::forward` (`fast_mmvq`/`fast_mmq`), custom fused FP32-accumulating `ncm_fused_softmax_{bf16,f16}`, generic 16-bit `ncm_pack_qkv_u16`/`ncm_merge_heads_u16`, and cached fixed-matrix resampling GEMMs (`resample.rs`).
- **Metal (`ncm-backend-mlx`):** eliminated per-layer FP32 casts in `bf16`/`fp16`/`q8`/`q4` across norms, RoPE, GELU, attention, and quantized matmuls plus cached fixed-matrix resampling GEMMs.
- **FP32 non-regression:** 9/9 CLI profile byte-equivalence vs `v0.4.1` passed on both CUDA and Metal.

### Paired before/after warm latency (ms)

| Profile / Batch / Precision | RTX 3060 Ti Before → After | CUDA Speedup | M1 Ultra Before → After | Metal Speedup | Quality note |
| --- | --- | --- | --- | --- | --- |
| `sudoku-easy` b=1 `bf16` | 30.99 → **17.55 ms** | **1.77×** | 61.12 → **38.89 ms** | **1.57×** | 128/128 exact (both GPUs) |
| `sudoku-easy` b=1 `fp16` | — | — | 57.05 → **33.24 ms** | **1.72×** | 128/128 exact |
| `tinystories-15k-ema` b=1 `bf16` | 42.83 → **27.45 ms** | **1.56×** | 99.93 → **78.58 ms** | **1.27×** | Fluent stories |
| `tinystories-15k-ema` b=1 `q8` | 51.82 → **36.35 ms** | **1.43×** | 101.12 → **76.54 ms** | **1.32×** | Fluent stories |
| `tinystories-15k-ema` b=1 `q4` | 50.90 → **35.70 ms** | **1.43×** | — | — | Fluent stories |
| `tinystories-15k-ema` b=8 `bf16` | 158.90 → **104.93 ms** | **1.51×** | 247.77 → **195.12 ms** | **1.27×** | `-213 MiB` MLX peak on Metal |
| `wmt14` b=1 `bf16` | 39.10 → **25.50 ms** | **1.53×** | — | — | Matches FP32 parity budget |
| `wmt14` b=1 `fp16` | 32.87 → **24.77 ms** | **1.33×** | 80.13 → **54.82 ms** | **1.46×** | 27/32 exact (96.14% tok) CUDA; 28/32 (98.58% tok) Metal |
| `wmt14` b=1 `q8` | — | — | 82.51 → **59.07 ms** | **1.40×** | 84.91% token match |
| `xsum` b=1 `bf16` | 47.44 → **30.34 ms** | **1.56×** | — | — | `-24 MiB` CUDA VRAM |
| `xsum` b=1 `fp16` | 40.17 → **29.58 ms** | **1.36×** | 84.58 → **62.76 ms** | **1.35×** | 28/32 exact (95.57% tok) CUDA; 28/32 (94.99% tok) Metal |
| `lm1b-large` b=1 `bf16` | 190.52 → **131.77 ms** | **1.45×** | — | — | — |
| `lm1b-large` b=1 `fp16` | — | — | 364.34 → **297.58 ms** | **1.22×** | — |
| `lm1b-large` b=1 `q8` | 199.75 → **141.99 ms** | **1.41×** | — | — | — |
| `lm1b-large` b=1 `q4` | 194.10 → **136.96 ms** | **1.42×** | 453.62 → **359.55 ms** | **1.26×** | — |

### Head-to-head vs upstream Python `094f25ca` & CUDA graph decision (`tinystories-15k-ema`, RTX 3060 Ti)
- **Batch 1:** eager Rust CUDA `bf16` (**27.45 ms, 253.6 MiB VRAM**) beats upstream Python `094f25ca` **with `torch.compile` + CUDA graphs** (**31.28 ms, 647.9 MiB VRAM**) by **1.14×** at **2.55× less VRAM** (and beats Python eager `bf16+kv+resize` at 49.85 ms by **1.82×**).
- **Batch 8:** eager Rust CUDA `bf16` (**104.93 ms, 330.1 MiB VRAM**) beats Python eager `bf16+kv+resize` (106.32 ms, 703.8 MiB VRAM). Python CUDA graphs alone save ~9 ms (97.17 ms; `torch.compile` + graphs reaches 73.66 ms at 704.9 MiB VRAM). Raw CUDA stream capture was intentionally skipped to preserve dynamic batching and keep runtime complexity low.

---

## Milestone 8B: `ncm quantize` Verified `Q8`/`Q4` Pre-Quantized Sidecar Cache

### Key changes
- Verified sidecar schema (`crates/ncm-io/src/prequant.rs`) stored as `prequant-{cuda-ggml|mlx-affine}-{q8|q4}.{safetensors,manifest.json}` inside the artifact directory, bound to parent `manifest_sha256` with parallel SHA-256 verification and atomic writes.
- Native offline quantizers + direct GPU uploaders in `ncm-backend-cuda` (GGML `Q8_0`/`Q4_0`) and `ncm-backend-mlx` (group-64 affine `8-bit`/`4-bit`).
- `ncm quantize ARTIFACT --precision q8|q4 --device cuda|mlx [--force]` CLI command and transparent verified sidecar discovery in `ncm sample`, `ncm run`, `ncm serve`, and `phase5_stage_profile`.
- **Fail-closed security & integrity (4/4 verified on both GPUs):** 1-byte weight flip, stale parent `manifest_sha256`, missing/incomplete sidecar manifest, and duplicate generation without `--force` all fail closed with explicit errors.

### Cold load time & disk footprint (100% byte-identical output vs on-the-fly quantization on all 8 cells)

| Artifact / Precision | Sidecar Size (vs FP32) | RTX 3060 Ti Load (On-the-fly → Sidecar) | CUDA Speedup | M1 Ultra Load (On-the-fly → Sidecar) | Metal Speedup |
| --- | --- | --- | --- | --- | --- |
| `tinystories-15k-ema-v1` `q8` | **164.2 MB** (3.06× smaller) | 625.8 → **130.5 ms** | **4.79×** | 391.8 → **156.9 ms** | **2.50×** |
| `tinystories-15k-ema-v1` `q4` | **117.5 MB** (4.27× smaller) | 524.9 → **100.1 ms** | **5.24×** | 382.8 → **123.1 ms** | **3.11×** |
| `lm1b-large-inference-v1` `q8` | **982.7 MB** (3.41× smaller) | 4,660.7 → **659.7 ms** | **7.07×** | 2,466.5 → **791.6 ms** | **3.12×** |
| `lm1b-large-inference-v1` `q4` | **613.9 MB** (5.45× smaller) | 3,861.2 → **418.5 ms** | **9.23×** | 2,436.0 → **512.5 ms** | **4.75×** |

---

## Milestone 9A: `ncm-train` v2 — Upstream TinyStories `--init-from` & Native Rust LoRA

### Key changes
- **Native Rust LoRA (`crates/ncm-train/src/lora.rs`):** targets all generator transformer block attention & MLP linear weights (`c_attn`, `c_proj`, `c_fc`) with Kaiming `A` / zero `B` init, exact FP64-accumulated `W + (alpha / r) * (B @ A)` merge on export (so exported artifacts are standard verified `local-ncm-v1` models with zero inference overhead), and deterministic LoRA checkpoint save/resume.
- **Upstream `tinystories-inference-v1` admission (`crates/ncm-train/src/artifact.rs`):** converts tied codebook heads + `level0_unigram` into mathematically equivalent `local-ncm-v1` `UntiedAffine` weights in parallel, and `LocalTokenizer` (`ncm-tokenizers`) now loads embedded TinyStories `ByteBpe` vocabularies offline.
- **Per-level backward & frozen-weight detach:** accumulates gradients one generator level at a time and detaches frozen base transposes (`w.t()?.detach()`) so 123M-parameter LoRA fine-tuning fits comfortably inside **3.7 GiB free VRAM** on an 8 GiB desktop GPU.
- **Verification:** central finite-difference gradient checks on LoRA `A` and `B` (`< 2e-4`), LoRA-vs-merged forward loss equivalence (`< 1e-5`), deterministic 5+5 vs 10-step LoRA checkpoint resume (`< 1e-5`), and end-to-end CUDA + Metal convergence tests all passed.

### Live 123.16M `tinystories-15k-ema-v1` LoRA rank-8 fine-tuning (`batch=2`, 30 steps, `lr=3e-4`)

| Metric | RTX 3060 Ti (CUDA) | Apple M1 Ultra (Metal) |
| --- | --- | --- |
| Trainable / total params | 1,179,648 / 123,160,704 (**0.96%**) | 1,179,648 / 123,160,704 (**0.96%**) |
| Generator loss (step 1 → 30) | **6.5286 → 3.4522** | **6.5286 → 3.4522** (4-decimal parity) |
| Mean step time / total wall time | **542.0 ms/step** (18.63s) | **352.1 ms/step** (12.47s) |
| Checkpoint resume (step 30 → 32) | **3.3977 → 3.3677** | **3.3977 → 3.3677** (4-decimal parity) |
| Merged inference roundtrip | Verified `local-ncm-v1`: `ncm inspect`, `ncm sample` (`fp32`/`bf16`), `ncm run`, `ncm quantize q8` + `ncm sample --precision q8` | Identical FP32 merged sample + fluent `bf16` & `q8` sidecar roundtrip |

---

## Milestone 9B: Mixed-Precision `BF16` Training, Gradient Accumulation, JSONL & Validation Eval

### Key changes
- **JSONL datasets (`crates/ncm-train/src/data.rs`):** `.jsonl` / `.ndjson` loader supporting `{"text": "..."}`, `{"prompt": "...", "completion": "..."}`, and `{"prefix": [...], "target": [...]}` with 50%-overlap sliding-window chunking for long texts plus sequential `eval_batches`.
- **Mixed-precision `BF16` training (`--train-precision fp32|bf16`):** executes linear GEMMs and attention matmuls in `BF16` and stores frozen 2D LoRA base weights in `BF16`, while keeping trainable master weights, AdamW moments, norms, softmax, and cross-entropy loss in `FP32`. Explicitly rejected on CPU.
- **Gradient accumulation (`--grad-accum N`) & validation eval (`--val-data PATH --eval-every N`):** verified exact equivalence (`batch=4, grad_accum=1` vs `batch=2, grad_accum=2` `< 1e-5` on both codec and generator) and detached no-grad `val_loss` reporting on both CUDA and Metal.
- **CUDA memory hygiene:** detached AdamW optimizer moments/updates, detached validation weights, and explicit CUDA stream synchronization between micro-batches/steps eliminated a `~24–56 MiB/step` CUDA async allocator growth observed under tight (`~3.5 GiB`) free VRAM.

### Live 123.16M `tinystories-15k-ema-v1` FP32 vs BF16 LoRA r=8 (`batch=2, grad_accum=2`, eff batch 4, 20 steps, JSONL train + held-out JSONL val, `--eval-every 10`)

| Platform & Mode | Train Loss (step 1 → 20) | Val Loss (step 10 → 20) | Mean Step Time | Total Wall Time | Δ vs FP32 (Train / Val) |
| --- | --- | --- | --- | --- | --- |
| **RTX 3060 Ti `fp32`** | **6.5840 → 3.9293** | **4.6507 → 4.0802** | 1,189.8 ms | 26.08s | baseline |
| **RTX 3060 Ti `bf16`** | **6.5831 → 3.9295** | **4.6519 → 4.0810** | **1,128.5 ms (1.05×)** | **24.82s** | `0.0002` / `0.0008` (~576 MiB lower peak VRAM) |
| **M1 Ultra `fp32`** | **6.5840 → 3.9293** | **4.6507 → 4.0802** | **724.3 ms** | **16.32s** | 4-decimal match vs CUDA FP32 |
| **M1 Ultra `bf16`** | **6.5834 → 3.9292** | **4.6510 → 4.0808** | 853.4 ms (0.85×) | 18.86s | `0.0001` / `0.0006` |

> **Hardware disclosure:** On Apple **M1** Ultra (which predates the hardware `bfloat16` ALU introduced in M2), MLX emulates `bfloat16` casts/matmuls at a small throughput cost (`0.85×` vs FP32) while still halving frozen-weight memory footprint. On NVIDIA Ampere (RTX 3060 Ti), `--train-precision bf16` delivers both a `1.05×` step speedup and `~576 MiB` lower peak VRAM at `< 1e-3` loss parity.

---

## Validation Commands Executed

- **Linux/CUDA:**
  - `./tools/check.sh`
  - `cargo clippy -p ncm-train --features cuda --all-targets -- -D warnings`
  - `cargo clippy -p ncm-cli --features cuda-cudnn --all-targets -- -D warnings`
  - `cargo test --release -p ncm-train --features cuda` (2 unit + 12 integration tests)
  - `cargo test --release -p ncm-backend-cuda --features cuda` (including `precision.rs` 5/5)
  - 9/9 FP32 CLI byte-equivalence, Milestone 8A/8B/9A/9B live RTX 3060 Ti validation harnesses
- **Apple Silicon/Metal (`macstudio`):**
  - `./tools/check.sh --mlx` (including `ncm-backend-mlx` `precision.rs` 4/4 and 18/18 Metal `mlx_probe`)
  - `cargo clippy -p ncm-train --features mlx --all-targets -- -D warnings`
  - `cargo test --release -p ncm-train --features mlx` (including `mlx_full_train`, `mlx_lora_finetune`, `mlx_bf16_grad_accum_and_val_eval`)
  - `cargo clippy -p ncm-cli --features mlx --all-targets -- -D warnings` & `cargo test -p ncm-cli --features mlx`
  - 9/9 FP32 CLI byte-equivalence, Milestone 8A/8B/9A/9B live M1 Ultra validation harnesses
