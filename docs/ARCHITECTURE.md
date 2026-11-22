# Architecture

Status: **Implemented across Phases 0–5 plus the pre-`v0.2.0` gap-audit bundle on Linux/CUDA and Apple Silicon/Metal** (`v0.1.0` published at `de2975a`; `v0.2.0` candidate verified on both GPUs at `e7b8cfa`). See [docs/DEVELOPING.md](DEVELOPING.md) for library usage and running custom local NCM models (`local-ncm-v1`), and [docs/results/2026-11-20-phase5-audit/phase5-final-audit-v1.md](results/2026-11-20-phase5-audit/phase5-final-audit-v1.md) for the requirement-to-evidence audit.

## Decision: one model, two runtime adapters

Model semantics live once in Rust (`ncm-model`), generic over narrow backend traits (`ncm-core`) with associated tensor and index types. The concrete backend (`CandleBackend` for CUDA/cuDNN or `MlxBackend` for Apple Silicon Metal) is chosen explicitly at the application boundary and monomorphized on hot paths—never dispatched per operator through trait objects and never silently falling back to CPU.

```text
ncm-cli (unified `ncm sample`, `inspect`, `generate`, `reconstruct`, aliases)
  ├── ncm-io                 manifest verification, parallel SHA-256, streaming weight I/O
  ├── ncm-tokenizers         pinned Sudoku, Byte BPE, GPT-2, Qwen3, and BERT tokenizers
  ├── ncm-model              Codec, Generator, TinyStories/XSum/WMT14/LM1B teachers, TraceSink
  │     └── ncm-core         validated configs, OwtConfigOnly, backend traits, typed errors
  ├── ncm-backend-cuda ─────> ncm-core + Candle -> CUDA / cuBLAS / cuDNN / custom CUDA kernels
  └── ncm-backend-mlx ──────> ncm-core + mlx-rs -> mlx-c -> native MLX / Metal
```

`ncm-core` has zero dependency on any concrete GPU runtime. A bounded Candle CPU backend exists solely for deterministic unit fixtures and offline development, **not** as a third production target.

### Why these runtime choices

- **Apple Silicon (`mlx-rs` → `mlx-c` → MLX):** satisfies the native Apple Silicon goal without Python at runtime. `MlxBackend` uses safe `mlx-rs` bindings over prebuilt native MLX libraries (`MLX_SYS_PREBUILT`) with explicit graph evaluation boundaries.
- **Linux (`candle` + CUDA/cuDNN + audited custom kernels):** provides Rust tensor execution without LibTorch or Python. Custom CUDA kernels cover exact resampling, stable tie-breaking argmax, masked softmax, QKV split, RMS+RoPE, categorical sampling, and FP64 sequential decoding penalties where composed ops were slower or numerically divergent.
- **No Python production bridge:** Python is used strictly offline under `tools/reference/` for checkpoint export, reference oracle traces, and official quality evaluation (ROUGE, sacreBLEU/chrF, GPT-2-large perplexity).

## Core types and invariants

- **Strict configuration validation (`ncm-core`):** `CodecConfig`, `GeneratorConfig`, `OwtConfigOnly`, and per-profile model/tokenizer envelopes reject unknown inference-critical fields, invalid level schedules, mismatched head/embed dimensions, invalid vocab/PAD/EOS IDs, and unsupported research branches before any GPU allocation.
- **Preflight-before-allocation (`ncm-model`):** every model constructor validates the complete manifest profile, tensor inventory, dtypes, shapes, finite FP32 weight values, `lvl_1L`, `attn_mask_for_masking`, head temperature scales, and `inference.safetensors` distributions/RoPE tables on the host **before** invoking `WeightBackend::upload_f32` for the first time.
- **Verified streaming artifact loading (`ncm-io`):**
  - `Artifact::load_inference_streaming` opens safetensors bundles with `O_NOFOLLOW`, validates `codec.safetensors`, `generator.safetensors`, and `training.safetensors` concurrently with parallel per-tensor SHA-256 and finite FP32 scanning using bounded 1 MiB scratch buffers (anonymous `mmap`/`munmap` on macOS aarch64), and retains open validated file descriptors for inference bundles (dropping `training.safetensors`).
  - `Artifact::visit_bundle_tensors` streams tensors sequentially in file-offset order into a single reusable `Scratch` buffer (immediately `munmap`ped on macOS drop) and uploads directly to CUDA or MLX. This eliminates host-RAM double-buffering (~48–49% RSS reduction on Metal, up to -2.8 GiB host HWM on CUDA LM1B-large) and speeds up verified model loading 1.7×–4.6×.
  - `Artifact::load` and `Artifact::load_inference` remain available for in-memory inspection and fixtures (also benefiting from parallel per-tensor SHA-256).
- **Immutable weights, request-local state:** loaded models own immutable device weights and precomputed static level geometry (attention masks, RoPE tables, codebook norms). Per-request activations, latent accumulators, and deterministic SplitMix64 RNG state are local to each inference call; shared models serialize requests via an internal `Mutex<()>`.

## Tensor and backend contract

Defined in [crates/ncm-core/src/lib.rs](../crates/ncm-core/src/lib.rs):

- Logical sequence layout is `BLC`; attention layout is `BHLD`; convolution weights retain PyTorch `OIK` (transposed: `IOK`) layout at the trait boundary.
- All production inference math is **FP32** (TF32 disabled on CUDA where configurable). Exact sequential final token penalties (XSum, LM1B) use FP64 accumulation on host or CUDA device as qualified.
- Selection (`select_last`) breaks exact ties by selecting the **first** index.
- Attention takes an explicit additive broadcast mask (`None` = bidirectional; never implicit causal).
- Resampling implements exact PyTorch adaptive-area downsampling (`area1d`) and half-pixel `align_corners=False` linear upsampling (`linear1d`).
- `evaluate(&[Tensor])` provides explicit materialization/synchronization for MLX lazy evaluation and reproducible CUDA/Metal benchmarks.

## Implemented profiles vs future architectural work

| Capability | Current Status (v0.1.0 / v0.2.0-rc) |
| --- | --- |
| Sudoku (easy, medium, hard) | Qualified FP32 greedy generation & codec reconstruction on CUDA and Metal |
| TinyStories | Qualified seeded generation + `--text`/`--decode` (Byte BPE) on CUDA and Metal (with recorded Metal exceptions) |
| XSum | Qualified top-1 CFG 0 + `--text`/`--decode` (GPT-2) + FP32 prefix K/V cache on CUDA and Metal (with recorded exceptions); experimental positive CFG and stochastic sampling supported on both CUDA and Metal |
| WMT14 | Qualified top-1 CFG 0/1 + `--text`/`--decode` (Qwen3) + FP32 prefix K/V cache (CFG 0) on CUDA and Metal; experimental CFG/padded-retry gated |
| LM1B (small, medium, large) | Qualified seeded generation (80-seed gate) + empty-only `--text`/`--decode` (BERT) on CUDA and Metal; `v1` uses sibling rotary sidecar, `v2` bundles hash-pinned RoPE tables in `inference.safetensors` |
| Custom local NCM (`local-ncm-v1`) | Generic config-driven `LocalNcmModel` + offline `LocalTokenizer` + FP32 prefix K/V cache + stdlib exporter (`tools/reference/export_local_ncm.py`) on CPU, CUDA, and Metal |
| OpenWebText (OWT) | Config validation (`OwtConfigOnly`), no-EOS-cut policy, and synthetic operator shapes implemented; runtime execution refused until upstream releases weights |
| BF16/FP16, quantization | Deferred past Phase 5; each requires its own equivalence/quality gate |
| Multi-request serving, streaming, training | Deferred (Phase 6+ / separate approval gate) |

## Platform, safety, and compatibility policy

- Build each platform separately (`--features cuda-cudnn` or `--features cuda` on Linux; `--features mlx` on Apple Silicon). Default CPU checks require neither CUDA nor MLX.
- `unsafe_code = deny` is enforced workspace-wide; the only `#[allow(unsafe_code)]` blocks are narrowly audited FFI/CUDA/Darwin-`mmap` wrappers with explicit safety invariants.
- Canonical artifact schemas and reference fixtures are versioned independently of Rust crates and upstream Git (`docs/UPSTREAM.md`). Upstream pins are never changed silently.
