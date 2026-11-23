# Phase 7 Dual-Platform Results: Quantization (`Q8`/`Q4`), Serving v2, and Native Rust Training (`v0.4.0`)

Date: 2026-11-23 UTC
Plan: [docs/PHASE7.md](../../PHASE7.md)

All three Phase 7 milestones were implemented and verified on real target hardware — **Linux/CUDA first** (NVIDIA RTX 3060 Ti, 8 GiB VRAM) and **Apple Silicon/Metal second** (Mac Studio M1 Ultra, 128 GiB unified memory) — while preserving strict `FP32` defaults, pinned upstream behavior, and zero new external crate dependencies.

---

## 1. Milestone 7A — Native Weight Quantization (`--precision q8|q4`)

- **CUDA (`ncm-backend-cuda`):** native `QTensor::quantize_onto` (`GGML Q8_0` / `Q4_0`) + `QMatMul` stored in a backend-scoped side-table (zero FP32 GPU allocation, freed automatically on model drop).
- **Metal (`ncm-backend-mlx`):** native `mlx_rs::ops::quantize` (group size 64, 8-bit / 4-bit affine, immediately evaluated on upload) + `mlx_rs::ops::quantized_matmul`.
- **Shared routing (`ncm-model`):** all 2D linear projections (transformer QKV/out/MLP, codec linear layers, `decoder.output_layer.weight`, `lm_heads.*`) are quantized; embeddings (`encoder.wte`, `wpe`, `wle`, `vq.embeddings.*`), norms, biases, and 3D convs remain in `BF16` (so TinyStories runs cleanly in both `q8` and `q4`).

### 1.1 Linux/CUDA (RTX 3060 Ti, 8 GiB VRAM)

Baseline FP32 / FP16 values from [Phase 6 CUDA receipt](../2026-11-22-phase6-precision-serving/cuda-precision-v1.json); Q8 / Q4 values from [cuda-quantization-v1.json](cuda-quantization-v1.json):

| Profile | Batch | FP32 VRAM | FP16 VRAM | Q8 VRAM | Q4 VRAM | FP32 Warm | Q8 Warm | Q4 Warm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sudoku-easy` | 1 | 150.2 MiB | 109.1 MiB | **93.1 MiB** | **80.6 MiB** | 40.3 ms | 40.2 ms | 40.7 ms |
| `tinystories` | 1 | 527.1 MiB | *(rejected)* | **183.2 MiB** | **138.7 MiB** | 46.1 ms | 51.5 ms | 50.6 ms |
| `xsum` | 1 | 629.5 MiB | 351.2 MiB | **284.9 MiB** | **232.9 MiB** | 49.0 ms | 54.6 ms | 54.8 ms |
| `wmt14` | 1 | 1,038.3 MiB | 561.7 MiB | **415.3 MiB** | **365.5 MiB** | 38.4 ms | 46.3 ms | 45.2 ms |
| `lm1b-small` | 1 | 945.4 MiB | 531.6 MiB | **388.2 MiB** | **335.6 MiB** | 67.8 ms | 58.9 ms | 58.9 ms |
| `lm1b-medium` | 1 | 1,732.0 MiB | 924.8 MiB | **610.9 MiB** | **453.8 MiB** | 133.1 ms | 120.2 ms | 117.2 ms |
| `lm1b-large` | 1 | 3,215.6 MiB | 1,667.7 MiB | **1,025.1 MiB** (-68.1%) | **673.3 MiB** (-79.1%) | 246.2 ms | **199.0 ms** (1.24×) | **194.4 ms** (1.27×) |
| `lm1b-large` | 16 | 3,522.4 MiB | 1,955.1 MiB | **1,327.1 MiB** (-62.3%) | **975.4 MiB** (-72.3%) | 2,135.4 ms | 1,665.0 ms (1.28×) | 1,669.7 ms (1.28×) |

### 1.2 Apple Silicon/Metal (Mac Studio M1 Ultra)

Baseline FP32 / FP16 values from [Phase 6 MLX receipt](../2026-11-22-phase6-precision-serving/mlx-precision-v1.json); Q8 / Q4 values from [mlx-quantization-v1.json](mlx-quantization-v1.json):

| Profile | Batch | FP32 MLX Peak | FP16 MLX Peak | Q8 MLX Peak (RSS) | Q4 MLX Peak (RSS) | Q8 Warm | Q4 Warm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sudoku-easy` | 1 | 216.6 MiB | 175.4 MiB | **162.6 MiB** (84.0) | **149.3 MiB** (77.0) | 63.7 ms | 64.0 ms |
| `tinystories` | 1 | 620.2 MiB | *(rejected)* | **271.2 MiB** (117.2) | **226.7 MiB** (94.2) | 101.1 ms | 102.9 ms |
| `xsum` | 1 | 786.3 MiB | 506.9 MiB | **443.8 MiB** (574.4) | **391.9 MiB** (556.0) | 89.8 ms | 92.2 ms |
| `wmt14` | 1 | 1,100.8 MiB | 619.2 MiB | **437.3 MiB** (550.9) | **365.6 MiB** (550.8) | 82.6 ms | 84.2 ms |
| `lm1b-small` | 1 | 986.8 MiB | 563.9 MiB | **416.4 MiB** (270.8) | **363.8 MiB** (244.2) | 102.9 ms | 104.7 ms |
| `lm1b-medium` | 1 | 1,819.7 MiB | 994.3 MiB | **676.0 MiB** (363.9) | **518.9 MiB** (270.6) | 226.8 ms | 228.0 ms |
| `lm1b-large` | 1 | 3,414.4 MiB | 1,798.4 MiB | **1,131.4 MiB** (-66.9%, 544.2 RSS) | **780.0 MiB** (-77.2%, **324.5 RSS / -90.2%**) | 435.3 ms | 403.6 ms |
| `lm1b-large` | 16 | 4,767.4 MiB | 3,081.7 MiB | **2,519.2 MiB** (584.7 RSS) | **2,167.5 MiB** (363.7 RSS) | 2,449.8 ms | 2,480.4 ms |

### 1.3 Quality Screen & Honest Disclosures

- **Sudoku-easy (128 boards):**
  - CUDA: **128/128 (100%)** in `q8`; **126/128 (98.44%, 99.974% cells)** in `q4`.
  - Metal: **128/128 (100%)** in `q8`; **127/128 (99.22%, 99.991% cells)** in `q4`.
- **WMT14 De→En (32 rows):**
  - `q8`: **86.82%** token match on CUDA, **83.79%** on Metal (fluent translations at ~40% of FP32 VRAM).
  - `q4`: **57.71%** on CUDA, **57.08%** on Metal — noticeable translation quality degradation at 4-bit; `fp16` (98.9% / 97.7%) or `q8` is recommended for translation quality.
- **TinyStories:** fluent coherent stories in both `q8` and `q4` on both platforms (unlike `fp16`, which is cleanly rejected by the load-time range check due to `vq.embeddings.0.weight` magnitudes up to `2.48e10`).

---

## 2. Milestone 7B — Ollama-Class CLI & Serving v2

Implemented:
- `ncm run MODEL [PROMPT]`: Ollama-style one-shot or interactive hot-GPU REPL (`/bye`, `/exit`, `/seed`, `/cfg`, `/help`).
- `ncm pull (--list | MODEL)`: direct CLI wrapper over `tools/reference/quickstart_export.py`.
- `GET /api/ps`: Ollama-compatible active-model introspection (`size`, `size_vram`, `loaded_at`, `last_used_at`, `requests_served`, `details.quantization_level`).
- Dynamic micro-batching (`ncm serve --max-batch N`, default `8`): zero-overhead coalescing of concurrent compatible requests (preserving exact single-request determinism whenever an explicit `seed` is passed).

### 2.1 Concurrent Serving Throughput (`8` concurrent clients, `--max-batch 1` vs `--max-batch 8`)

Receipts: [cuda-serve-v2-v1.json](cuda-serve-v2-v1.json), [mlx-serve-v2-v1.json](mlx-serve-v2-v1.json).

| Platform | Workload | `--max-batch 1` | `--max-batch 8` | Throughput Speedup | Client p50 Latency |
| --- | --- | ---: | ---: | ---: | ---: |
| **Linux/CUDA** (RTX 3060 Ti) | `wmt14` (`fp16`, 24 reqs) | 25.22 req/s (1,614.4 tok/s) | **55.64 req/s (3,560.8 tok/s)** | **2.21×** | 306.4 ms → **141.1 ms** (-54%) |
| **Linux/CUDA** (RTX 3060 Ti) | `lm1b-large` (`q8`, 16 reqs) | 4.95 req/s (633.2 tok/s) | **8.10 req/s (1,037.4 tok/s)** | **1.64×** | 1,622.6 ms → **986.7 ms** (-39%) |
| **Apple Silicon/Metal** (M1 Ultra) | `wmt14` (`fp16`, 24 reqs) | 12.37 req/s (791.8 tok/s) | **22.91 req/s (1,466.0 tok/s)** | **1.85×** | 639.5 ms → **237.2 ms** (-63%) |
| **Apple Silicon/Metal** (M1 Ultra) | `lm1b-large` (`q8`, 16 reqs) | 2.31 req/s (295.3 tok/s) | **4.55 req/s (582.9 tok/s)** | **1.97×** | 3,471.6 ms → **1,779.7 ms** (-49%) |

### 2.2 CLI Smoke Verification

- `ncm pull --list` (7 starter models) & `--dry-run`: passed on both platforms.
- `ncm run wmt14 "Guten Morgen, wie geht es dir heute?" --precision q8`: passed on both platforms (**932 ms** cold-to-translation on M1 Ultra, **2,034 ms** on RTX 3060 Ti).
- `ncm run wmt14 --precision q8` (2-turn piped REPL): passed on both platforms, reusing hot GPU weights across turns (**+80.8 ms** for the 2nd turn on M1 Ultra).
- `GET /api/ps`: returned exact resident model metadata and accurate `requests_served` increment (`0 → 25` / `0 → 17`) on both platforms.

---

## 3. Milestone 7C — Native Rust Fine-Tuning & Training (`crates/ncm-train` + `ncm train`)

Implemented with **zero new external crate dependencies**:
- **Codec RVQ + Straight-Through + EMA Codebook Training (`--stage codec`):** full forward/backward pass with multiscale `area1d`/`linear1d`, straight-through estimator (`z_e + sg[z_q - z_e]`), commitment loss, EMA cluster/embedding updates with Laplace smoothing, and deterministic dead-code restarts.
- **Frozen-Codec Generator Training & Fine-Tuning (`--stage generator` / `--init-from` / `--stage all`):** teacher-forced multiscale generator with RoPE, QK-RMSNorm, block-diagonal masking, all 3 `local-ncm-v1` head layouts, decoupled AdamW, global L2 gradient clipping, linear warmup + cosine LR decay, atomic deterministic checkpoint save/resume, and direct verified `local-ncm-v1` export.
- **Dual-backend reverse-mode autodiff:** Candle CPU/CUDA (`candle-core` + `candle-nn`) and Apple Silicon Metal (`mlx_rs::transforms::value_and_grad_with_argnums`).

### 3.1 Correctness Verification (`crates/ncm-train/tests/train_suite.rs`)

- **Central finite-difference gradient check (`eps = 1e-3`):** verified analytic backward gradients across `transformer.wte.weight`, `l0_queries`, `attn.c_attn.weight`, `attn.q_norm.weight`, `mlp.c_fc.weight`, `lm_heads.0.weight`, and `lm_heads.2.bias` (all `< 2e-4` absolute diff). *Note:* this test caught and fixed a silent zero-gradient quirk in `candle_nn::ops::softmax_last_dim` by replacing it with a pure differentiable softmax.
- **EMA & dead-code restart test:** verified codebook convergence toward cluster mean and deterministic dead-code restarts.
- **Checkpoint save/resume determinism:** verified 5 steps → atomic checkpoint save → reload → 5 steps matches an uninterrupted 10-step run to `< 1e-5` on both Codec and Generator stages.

### 3.2 Dual-Platform End-to-End Training & Roundtrip Benchmark

Benchmark spec: 4-level `local-ncm-v1` (`in_vocab_size=64`, `context_length=16`, `embed_dim=64`, `levels=[2, 4, 8, 16]`, `vqvae_vocab_size=[32, 32, 32, 32]`, `n_embd=128`, `n_head=4`, `n_layer=2`, `batch=8`, 100 codec steps + 100 generator steps + 40 `--init-from` fine-tune steps), verified with `ncm inspect`, `ncm sample`, and `ncm serve`.

Receipts: [cuda-train-v1.json](cuda-train-v1.json), [mlx-train-v1.json](mlx-train-v1.json).

| Metric | Linux/CUDA (RTX 3060 Ti) | Apple Silicon/Metal (M1 Ultra) |
| --- | ---: | ---: |
| Scratch Codec Loss (`100` steps) | `4.5654 → 0.0426` (`8.76 ms/step`) | `4.5653 → 0.0425` (`16.75 ms/step`) |
| Scratch Generator Loss (`100` steps) | `3.4622 → 0.0011` (`17.97 ms/step`) | `3.4878 → 0.0010` (`18.55 ms/step`) |
| Scratch Total Wall Time (`200` steps + export) | **3.128 s** | **3.658 s** |
| Fine-Tune `--init-from` Loss (`40` steps) | `0.0011 → 0.0002` (`0.936 s`) | `0.0010 → 0.0002` (`0.804 s`) |
| `ncm inspect` Verified Profile | `local-ncm-v1` | `local-ncm-v1` |
| `ncm sample` Exact Sequence Recovery | **8/8 rows, 128/128 tokens (100.0%)** | **8/8 rows, 128/128 tokens (100.0%)** |
| `ncm serve` `/api/ps` & `/api/generate` Parity | **PASS (exact match)** | **PASS (exact match)** |

---

## 4. Published Receipts

- [cuda-quantization-v1.json](cuda-quantization-v1.json)
- [mlx-quantization-v1.json](mlx-quantization-v1.json)
- [cuda-serve-v2-v1.json](cuda-serve-v2-v1.json)
- [mlx-serve-v2-v1.json](mlx-serve-v2-v1.json)
- [cuda-train-v1.json](cuda-train-v1.json)
- [mlx-train-v1.json](mlx-train-v1.json)
