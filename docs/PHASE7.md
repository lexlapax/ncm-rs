# Phase 7 Plan (`v0.4.0` → `v1.0.0`): Quantization, Ollama-Class Serving v2 & Native Rust Training

**Approved by user:** 2026-11-22 ("finish all three").
**Execution order:** Milestone 7A (Weight Quantization `Q8`/`Q4`) → Milestone 7B (Ollama-Class CLI & Serving v2) → Milestone 7C (Native Rust Fine-Tuning & Codec/Generator Training).
Within every milestone, **Linux/CUDA validation completes first, then Apple Silicon/Metal (`macstudio`)**.

## Milestone 7A — Weight-Only Quantization (`--precision q8|q4`)

- **Strict default preservation:** `ComputePrecision::Fp32` remains the default everywhere; `bf16` and `fp16` remain unchanged.
- **Zero model-code duplication:** add `ComputePrecision::{Q8, Q4}` and `WeightBackend::upload_linear_weight_f32` (defaulting to `upload_f32`) in `ncm-core`, routing 2D linear projection weights (transformer QKV/out/MLP, codec linear projections, and generator/decoder LM heads) into compact quantized GPU storage while keeping embeddings, 1D norms/biases, and 3D convolutions in mixed precision (`bf16` on CUDA/Metal where wide range is needed, or `fp16`).
- **Native GPU quantized matmul:**
  - **Linux/CUDA (`ncm-backend-cuda`):** `candle_core::quantized::{QTensor, QMatMul, GgmlDType::{Q8_0, Q4_0}}` quantized directly from CPU slices onto CUDA VRAM (`QTensor::quantize_onto`), executing native CUDA `QMatMul` kernels without ever allocating FP32 copies on GPU.
  - **Apple Silicon/Metal (`ncm-backend-mlx`):** `mlx_rs::ops::quantize` (8-bit / 4-bit, group size 64, affine, immediately materialized during upload so temporary FP32 buffers drop) + `mlx_rs::ops::quantized_matmul`.
- **Gates (CUDA first, then Metal):**
  - Quantization operator & roundtrip regression tests in `crates/ncm-backend-cuda/tests/precision.rs` and `crates/ncm-backend-mlx/tests/precision.rs`.
  - Dual-platform quality comparison (`fp32` vs `fp16` vs `q8` vs `q4` across Sudoku, TinyStories, XSum, WMT14, LM1B-large).
  - Dual-platform latency, VRAM / MLX-peak & RSS benchmarks (`lm1b-large` b=1 and b=16).

## Milestone 7B — Ollama-Class CLI & Serving v2 (`ncm run`, `ncm pull`, `/api/ps`, Dynamic Micro-Batching)

- **Dynamic request micro-batching in `ncm serve`:** when concurrent HTTP clients target the same model (`xsum`, `wmt14`, `tinystories`, `lm1b-*`, `local-ncm-v1`) with compatible settings, the GPU worker drains queued requests up to `--max-batch` (default `8`) and executes a single batched GPU pass, multiplying concurrent throughput while preserving single-request latency.
- **Active model introspection (`GET /api/ps`):** Ollama-compatible endpoint returning currently loaded resident models, precision, load timestamp, and cache slot usage.
- **Ollama-style interactive CLI:**
  - `ncm run <model> [--models-dir DIR] [--device DEV] [--precision ...]`: interactive stdin REPL (or one-shot prompt) keeping the model hot in GPU memory across turns (`exit` / `/bye` / Ctrl-D).
  - `ncm pull <model> [--models-dir DIR] [--python PY] [--dry-run]`: native CLI command invoking `tools/reference/quickstart_export.py` cleanly to fetch and verify a starter model.
- **Gates (CUDA first, then Metal):** unit/integration tests in `crates/ncm-serve` and `crates/ncm-cli`, plus live concurrent batched throughput benchmark (`4–8` concurrent clients vs sequential) and live `ncm run` / `/api/ps` verification on both RTX 3060 Ti and M1 Ultra.

## Milestone 7C — Native Rust Fine-Tuning & Training (`crates/ncm-train` + `ncm train`)

- **Frozen-codec generator training & fine-tuning first:**
  - Pure-Rust autodiff training engine (`crates/ncm-train`) using `candle-core` `VarMap` / `Var` / `AdamW` on CUDA & CPU and native MLX gradient computation (`mlx_rs::transforms::value_and_grad`) + `AdamW` on Apple Silicon Metal.
  - Teacher-forced multiscale NCM generator forward + level-weighted cross-entropy loss, gradient clipping, cosine/constant LR schedule, deterministic checkpoint save/resume, and direct verified export to `local-ncm-v1` (loadable immediately by `ncm sample` and `ncm serve`!).
- **Codec RVQ + straight-through + EMA codebook training:**
  - Reconstruction cross-entropy + VQ commitment loss + straight-through estimator + EMA codebook count/sum updates + dead-code restarts, exporting a full `(codec.safetensors, generator.safetensors, config.json, manifest.json)` `local-ncm-v1` bundle.
- **Gates (CUDA first, then Metal):**
  - Finite-difference gradient checks on synthetic tensors, EMA update unit tests, checkpoint save/resume determinism test, tiny-overfit loss convergence on real CUDA GPU and real Metal GPU, and end-to-end roundtrip (`ncm train` → `ncm inspect` → `ncm sample` → `ncm serve`).
