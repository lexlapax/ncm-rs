# Phase 6 Plan (`v0.3.0`): Half Precision, OpenAI/Ollama Serving & Quickstart

**Approved by user:** 2026-11-22.
**Execution order:** Milestone 6A (BF16/FP16) → Milestone 6B (`ncm serve` OpenAI + Ollama API) → Milestone 6C (Newbie Quickstart & CLI ergonomics) → `v0.3.0` release.
Within every milestone, **Linux/CUDA validation completes first, then Apple Silicon/Metal (`macstudio`)**.

## Milestone 6A — Opt-In Half Precision (`BF16` / `FP16`)

- **Strict FP32 default preservation:** `ComputePrecision::Fp32` remains the default everywhere and executes the exact unchanged FP32 path (zero numerical or output changes).
- **Shared trait & zero model duplication (`ncm-core`):** add `ComputePrecision::{Fp32, Bf16, Fp16}` and `WeightBackend::precision()`. Existing canonical FP32 safetensors artifacts load without re-exporting.
- **Mixed-precision execution (`ncm-backend-cuda` and `ncm-backend-mlx`):**
  - Store weights, KV caches, and activations in `BF16` or `FP16` (~2× smaller VRAM / unified memory).
  - Execute GEMMs (`project`, `project_bias`) and `attention` on native Tensor Cores (Ampere SM86+) / Metal FP16/BF16 pipelines (with FP32 softmax accumulation).
  - Accumulate sensitive reductions (`layer_norm`, `rms_norm`, `group_norm`, `rms_rope`, `area1d`, `linear1d`, `conv1d` on CUDA, `square`/`sum_last`, selection, categorical sampling, and FP64 decoding penalties) in FP32/FP64 to prevent overflow/underflow.
- **Gates (CUDA first, then Metal):**
  - FP32 regression check (exact byte-identity preserved).
  - Quality comparison (FP32 vs BF16 vs FP16 across Sudoku, TinyStories, XSum, WMT14, LM1B).
  - Dual-platform 3-alternating-process latency & memory benchmark (including `lm1b-large` batch 16 BF16 capacity on 8 GiB RTX 3060 Ti).

## Milestone 6B — Ollama & OpenAI-Compatible Serving (`ncm-serve` + `ncm serve`)

- **Zero new external runtime dependencies:** pure Rust HTTP/1.1 + SSE/NDJSON server in `crates/ncm-serve` using `std::net::TcpListener`, `serde`, and `serde_json`.
- **OpenAI drop-in endpoints:** `GET /v1/models`, `POST /v1/completions`, `POST /v1/chat/completions` (supporting both `stream: false` and `stream: true` SSE).
- **Ollama drop-in endpoints:** `GET /`, `GET /health`, `GET /api/tags`, `POST /api/show`, `POST /api/generate`, `POST /api/chat` (supporting both `stream: false` and `stream: true` NDJSON).
- **Model manager & concurrency:** auto-discover verified artifacts under `--models-dir` (plus `--model name=path`), lazy load with bounded LRU cache, single dedicated GPU worker thread serving concurrent HTTP clients without stream deadlocks, localhost-only default bind (`127.0.0.1:11434`), bounded request sizes, and CORS headers.
- **Gates (CUDA first, then Metal):** offline protocol unit tests in `./tools/check.sh`, plus live end-to-end HTTP + streaming + OpenAI-client compatibility tests on RTX 3060 Ti and M1 Ultra.

## Milestone 6C — Newbie Quickstart & CLI Ergonomics

- Direct `--prompt "..."` flag on `ncm sample`, `--precision fp32|bf16|fp16`, `ncm list`, and `ncm serve`.
- Streamlined `docs/QUICKSTART.md` (5-minute build → fetch/export → CLI & OpenAI/Ollama server path without compiling CPython from source; historical exact-hash recipes remain in `docs/USAGE.md`).
- Dual-platform results published in `docs/results/2026-11-22-phase6-precision-serving/README.md`, refreshed docs, and `v0.3.0` release notes.
