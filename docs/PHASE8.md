# Phase 8 & Phase 9 Plan (`v0.4.1` → `v0.5.0`): Inference Performance, Pre-Quantized Artifacts & `ncm-train` v2

**Approved by user:** 2026-11-23 ("do 1 then 2 and 3"; Item 1 `v0.4.1` released at `042111a`).
**Execution order:** Milestone 8A (Low-Precision Hot-Path Kernels & Fixed-Matrix Resampling) → Milestone 8B (Pre-Quantized `Q8`/`Q4` Artifact Cache `ncm quantize`) → Milestone 9A (`ncm-train` v2 Upstream Fine-Tuning + LoRA) → Milestone 9B (`BF16` Mixed-Precision Training, Gradient Accumulation, JSONL Datasets & Validation Eval).
Within every milestone, **Linux/CUDA work and validation complete first, then Apple Silicon/Metal (`macstudio`)**. Strict `FP32` defaults remain 100% byte-identical across all 9 profiles on both GPUs.

---

## Milestone 8A — Native Low-Precision Hot-Path Kernels & Fixed-Matrix Resampling

### Linux / CUDA (`crates/ncm-backend-cuda`)
1. **Eliminate per-layer FP32 round-trip casts in `bf16`/`fp16`/`q8`/`q4` (`src/codec.rs`):**
   - `candle_nn::ops::layer_norm`, `candle_nn::ops::rms_norm`, and `gelu_erf` in pinned `candle-core`/`candle-nn` `0.11.1` already ship native `layernorm_{bf16,f16}` and `rmsnorm_{bf16,f16}` CUDA kernels with internal FP32 accumulation. Call them directly on `bf16`/`f16` tensors (caching the zero bias vector per `(width, dtype)` instead of allocating on every bias-less `layer_norm`) instead of casting `x`, `weight`, `bias` to FP32 and back on every norm and MLP block.
   - Pass native `BF16` activations directly into `QMatMul::forward` when `fast_mmvq`/`fast_mmq` supports `BF16` (already supported in `candle-core-0.11.1/src/quantized/cuda.rs`), removing 2× full activation casts per quantized linear projection.
   - Streamline `bf16`/`fp16` attention to avoid redundant per-layer mask casts while verifying quality parity on Sudoku, WMT14, XSum, and TinyStories.
2. **Fixed-matrix resampling (`src/resample.rs`):**
   - For non-FP32 precisions (`bf16`/`fp16`/`q8`/`q4`), precompute and cache the exact `(out_len, in_len)` linear/area interpolation matrix in target dtype once per `(area, in_len, out_len, dtype)` and execute resampling as a single Tensor Core GEMM instead of FP32 casting + 4–6 gather/mul/sum/div kernels. Keep the strict FP32 path untouched.
3. **CUDA graph capture/replay evaluation:**
   - Benchmark updated Rust CUDA (`fp32`/`bf16`/`fp16`/`q8`/`q4`) head-to-head against upstream Python `094f25ca` (eager `fp32`, eager fast `bf16`+KV+resize, and `torch.compile` + CUDA graphs) on RTX 3060 Ti using retained `.reference/upstream-ncm-094f25ca` & `artifacts/upstream/tinystories-candidate-e34c3fab`. Prototype opt-in CUDA graph replay via `cudarc::driver::sys` stream capture only if measured end-to-end benefit justifies its complexity; revert any optimization without real measured gain.

### Apple Silicon / Metal (`crates/ncm-backend-mlx`)
1. **Fixed-matrix resampling (`src/ops.rs` / `src/codec.rs`):**
   - Notice `ops::area1d` currently builds `length` Python-style per-bin slices + `mean_axis` + `concatenate` (up to 128 slice+mean ops per level!). For non-FP32 (`bf16`/`fp16`/`q8`/`q4`), cache precomputed `(out_len, in_len)` linear/area matrices per `(area, in_len, out_len, dtype)` on `MlxBackend` and execute as a single `mlx_rs::ops::matmul`.
2. **Compiled MLX elementwise/norm/RoPE graphs:**
   - Reuse `mlx_rs::transforms::compile::compile` (already proven in `codec.rs` for `gelu_graph`) to fuse `rms_norm`, `layer_norm`, `gelu`, and `rotary` for `bf16`/`fp16`/`q8`/`q4` while preserving the strict FP32 path byte-identical.
3. **Gates (CUDA first, then Metal):**
   - Unit/operator parity & precision tests (`precision.rs`), full 9-profile FP32 byte-identity check, Sudoku/WMT14/XSum/TinyStories quality regression checks, and before/after warm latency + VRAM/MLX-peak benchmarks.

---

## Milestone 8B — Pre-Quantized `Q8`/`Q4` Artifact Cache (`ncm quantize`)

- **Problem:** Loading `--precision q8|q4` currently streams full FP32 weights (up to 3.26 GiB for `lm1b-large`) and quantizes every 2D matrix on the fly (+320 ms on TinyStories, +2–3 s on `lm1b-large`).
- **Design (`ncm-core`, `ncm-io`, `ncm-backend-cuda`, `ncm-backend-mlx`, `ncm-cli`):**
  - Native Rust command: `ncm quantize ARTIFACT --precision q8|q4 --device cuda|mlx` (zero Python, zero new crate dependencies).
  - Writes a verified pre-quantized sidecar bound to the parent artifact's `manifest.json` SHA-256:
    - CUDA (`ggml`): raw `Q8_0`/`Q4_0` byte tensors (`QTensor::data()`) in `prequant-ggml-{q8|q4}.safetensors` + `prequant-ggml-{q8|q4}.json` (recording `manifest_sha256`, precision, format, shapes, and per-tensor/file SHA-256s). Loaded directly onto GPU via `QStorage::from_data(Cow::Borrowed(bytes), device, ggml_dtype)` + `QTensor::new` + `QMatMul::from_qtensor` — zero CPU quantization!
    - Metal (`mlx`): exact `(weights: U32, scales: BF16, biases: BF16)` group-64 affine tensors in `prequant-mlx-{q8|q4}.safetensors` + `prequant-mlx-{q8|q4}.json`. Loaded directly into `MlxQuantized` — zero Metal quantization pass!
  - `ncm sample` / `ncm run` / `ncm serve` automatically use a verified matching pre-quantized sidecar when `--precision q8|q4` is requested (failing closed on any `manifest_sha256` or SHA-256 mismatch), while falling back cleanly to on-the-fly quantization when no sidecar is present.
- **Gates (CUDA first, then Metal):**
  - Tamper/mismatch negative tests, exact output identity (pre-quantized vs on-the-fly quantized `q8`/`q4`), and before/after model load latency + disk footprint benchmarks on `tinystories-15k-ema-v1` and `lm1b-large`.

---

## Milestone 9A — `ncm-train` v2: Upstream Checkpoint Fine-Tuning & Native LoRA (`--lora-rank`)

1. **Upstream checkpoint `--init-from` support:**
   - Extend `crates/ncm-train/src/artifact.rs` to admit verified upstream `tinystories-inference-v1` artifacts (15k EMA and legacy) in addition to `local-ncm-v1`, mapping codec + generator weights cleanly and exporting a runnable verified `local-ncm-v1` artifact.
2. **Native Rust LoRA fine-tuning (`--lora-rank R`, `--lora-alpha A`, default `0` = full fine-tuning):**
   - Freeze base projection weights `W` (`[out, in]`) on attention/MLP linears (`qkv`, `proj`, `fc1`, `fc2`) and train low-rank adapters `A: [r, in]` (Kaiming init) and `B: [out, r]` (zero init) with `y = x W^T + (alpha / r) * (x A^T) B^T`.
   - Cuts trainable parameters and AdamW optimizer state (`m`, `v`) by >95%, enabling real 89M-parameter TinyStories 15k EMA fine-tuning on an 8 GiB RTX 3060 Ti and Apple Silicon Metal.
   - Automatically merges `W_merged = W + (alpha / r) * (B @ A)` into `generator.safetensors` at export time for zero inference overhead across `ncm sample`, `ncm run`, and `ncm serve`.
3. **Gates (CUDA first, then Metal):**
   - Central finite-difference gradient check on LoRA `A`/`B` parameters (`< 2e-4`), exact LoRA-vs-merged forward equivalence (`< 1e-5`), checkpoint save/resume determinism with LoRA state, and live TinyStories 15k EMA LoRA fine-tuning convergence + inference roundtrip on RTX 3060 Ti and M1 Ultra.

---

## Milestone 9B — `BF16` Mixed-Precision Training, Gradient Accumulation, JSONL Datasets & Validation Eval

1. **Gradient accumulation (`--grad-accum N`, default `1`):** accumulate scaled gradients over `N` micro-batches per AdamW step; verified by exact equivalence test (`batch=4, accum=1` vs `batch=2, accum=2` `< 1e-5`).
2. **Mixed-precision `BF16` training (`--train-precision fp32|bf16`, default `fp32`):** FP32 master weights + AdamW moments with `BF16` forward/backward matmuls & activations on CUDA (Ampere+) and Metal (MLX).
3. **JSONL & streaming text datasets + validation eval (`--val-data FILE`, `--eval-every N`):**
   - Support `.jsonl` (`{"text": "..."}`, `{"prompt": "...", "completion": "..."}`, `{"prefix": [...], "target": [...]}`) with sliding-window chunking in `TokenDataset::load`.
   - Run no-grad validation loss evaluation every `N` steps and record `val_loss` in `StepMetrics` and `TrainReport`.
4. **Gates (CUDA first, then Metal):**
   - Unit tests for JSONL loading, gradient accumulation equivalence, `val_loss` evaluation, and `bf16` training convergence + VRAM/speed comparison on RTX 3060 Ti and M1 Ultra.
   - Final dual-platform report (`docs/results/2026-11-23-phase8-phase9/README.md`) and cleanup of temporary Linux `.reference/upstream-ncm-094f25ca` / `artifacts/upstream/tinystories-candidate-e34c3fab` if no longer needed.
