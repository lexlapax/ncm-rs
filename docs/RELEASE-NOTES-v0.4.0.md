# Release Notes — `v0.4.0`

Date: 2026-11-23 UTC
Theme: **Weight Quantization (`Q8`/`Q4`), Ollama-Class CLI & Serving v2, and Native Rust Training**

`v0.4.0` completes Phase 7 across **Linux/CUDA** (RTX 3060 Ti) and **Apple Silicon/Metal** (M1 Ultra) with **zero new external crate dependencies** and full preservation of strict `FP32` defaults and pinned upstream parity.

---

## Highlights

### 1. Native Weight Quantization (`--precision q8|q4`)
- **CUDA (`ncm-backend-cuda`):** native `GGML Q8_0`/`Q4_0` quantization on upload (`QTensor::quantize_onto` + `QMatMul`, zero FP32 GPU allocation).
- **Apple Silicon/Metal (`ncm-backend-mlx`):** native group-64 8-bit / 4-bit affine quantization (`mlx_rs::ops::quantize` + `quantized_matmul`).
- **Memory savings & speed:**
  - **`lm1b-large` (b=1):** VRAM drops **3,215.6 → 1,025.1 MiB (`q8`, -68.1%, 1.24× faster) → 673.3 MiB (`q4`, -79.1%, 1.27× faster)** on RTX 3060 Ti, and **3,414.4 → 1,131.4 MiB (`q8`, -66.9%) → 780.0 MiB (`q4`, -77.2%, 324.5 MiB RSS / -90.2%)** on M1 Ultra.
  - **`lm1b-large` (b=16):** fits in **975.4 MiB (`q4`) / 1,327.1 MiB (`q8`)** VRAM on RTX 3060 Ti (`1.28×` faster than FP32).
  - **Embeddings, norms, biases, and 3D convs stay in `BF16`**, allowing TinyStories to run cleanly in both `q8` and `q4` (unlike `fp16`).
  - **Quality disclosure:** `sudoku-easy` solves **128/128 (100%)** in `q8` and **126–127/128 (98.4–99.2%)** in `q4`; `wmt14` matches **83.8–86.8%** tokens in `q8` but degrades to **57%** in `q4` (`fp16` or `q8` recommended for translation).

### 2. Ollama-Class CLI & Serving v2 (`ncm run`, `ncm pull`, `/api/ps`, Dynamic Micro-Batching)
- **`ncm run MODEL [PROMPT]`:** one-shot generation or interactive hot-GPU REPL (`/bye`, `/exit`, `/seed`, `/cfg`, `/help`). Cold-to-translation in **932 ms** on M1 Ultra (`+80.8 ms` per subsequent REPL turn).
- **`ncm pull (--list | MODEL)`:** one-command CLI wrapper over `tools/reference/quickstart_export.py`.
- **`GET /api/ps`:** Ollama-compatible active-model introspection reporting resident models, `size`, `size_vram`, `loaded_at`, `last_used_at`, `requests_served`, and quantization level.
- **Dynamic micro-batching (`ncm serve --max-batch N`, default `8`):** coalesces compatible concurrent requests into a single GPU forward pass with zero single-request latency penalty (while preserving exact determinism for explicit `--seed` requests):
  - **RTX 3060 Ti (8 concurrent clients):** **2.21×** throughput on `wmt14` FP16 (`25.22 → 55.64 req/s`, `3,560.8 tok/s`, p50 `-54%`), **1.64×** on `lm1b-large` Q8.
  - **M1 Ultra (8 concurrent clients):** **1.85×** throughput on `wmt14` FP16 (`12.37 → 22.91 req/s`, p50 `-63%`), **1.97×** on `lm1b-large` Q8.

### 3. Native Rust Fine-Tuning & Training (`crates/ncm-train` + `ncm train`)
- Pure-Rust, Python-free training and fine-tuning on **CPU, NVIDIA CUDA (Candle), and Apple Silicon Metal (MLX `value_and_grad`)**:
  - **Codec training (`--stage codec`):** multiscale RVQ + straight-through estimator + commitment loss + EMA codebook updates with Laplace smoothing and deterministic dead-code restarts.
  - **Generator training & fine-tuning (`--stage generator` / `--init-from` / `--stage all`):** frozen-codec teacher forcing, RoPE, QK-RMSNorm, decoupled AdamW, global L2 gradient clipping, warmup + cosine LR decay, and atomic deterministic checkpoint save/resume (`--checkpoint-dir` / `--resume`).
  - **Direct verified `local-ncm-v1` export:** exported artifacts pass strict SHA-256 verification and load immediately in `ncm inspect`, `ncm sample`, `ncm run`, and `ncm serve`.
  - **Verified parity & convergence:** central finite-difference gradient checks (`< 2e-4` abs diff across all layers), exact checkpoint resume determinism (`< 1e-5`), and cross-backend 4-level NCM training parity (100-step codec loss `4.5654 → 0.0426` on CUDA vs `4.5653 → 0.0425` on Metal; **100.0%** 8/8 exact sequence recovery through `ncm sample` and `ncm serve`).

---

## Full Evidence & Reports

- [Phase 7 Dual-Platform Report](results/2026-11-23-phase7/README.md)
- [Quickstart Guide](QUICKSTART.md) · [Usage Guide](USAGE.md) · [Architecture](ARCHITECTURE.md) · [Developer Guide](DEVELOPING.md)
