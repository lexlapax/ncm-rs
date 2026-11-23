# Release Notes: `v0.3.0`

`v0.3.0` adds **opt-in `BF16` / `FP16` mixed precision** across CUDA and Apple Silicon Metal, a **zero-new-dependency drop-in OpenAI & Ollama HTTP/1.1 serving server (`ncm serve`)**, local model discovery (`ncm list`), inline `--prompt` sampling, and a streamlined **5-minute newbie quickstart** (`docs/QUICKSTART.md`).

Strict FP32 remains the default everywhere with zero behavioral or numerical changes.

## Highlights

### 1. Opt-In Half Precision (`--precision fp32|bf16|fp16`)
- Works directly on existing verified FP32 safetensors artifacts — no re-export required.
- Stores weights, KV caches, and activations in 16-bit while keeping sensitive reductions (LayerNorm, RMSNorm, GroupNorm, RoPE, resampling, CUDA Conv1d, codebook norms, categorical sampling, and FP64 decoding penalties) in FP32/FP64.
- **Linux/CUDA (RTX 3060 Ti, 8 GiB):**
  - Cuts VRAM peak **40.7%–48.5%** across all 7 profiles (`lm1b-large` b=1: `3,215.6 → 1,667.7 MiB`).
  - Speeds up warm inference up to **1.47×** on `lm1b-large` (`246.2 → 167.7 ms` `fp16`), **1.26×** on `lm1b-medium`, **1.19×** on `xsum`, and **1.10×** on `wmt14`.
  - Unlocks **`lm1b-large` batch 16 in 1,955.1 MiB VRAM** (`90.7 ms/seq` `fp16`, `97.6 ms/seq` `bf16`) on an 8 GiB consumer GPU.
- **Apple Silicon/Metal (M1 Ultra):**
  - Eager half-precision weight materialization cuts MLX GPU peak **36.4%–47.3%** (`lm1b-large` b=1: `3,414.4 → 1,798.4 MiB`; b=16: `-1.7 GiB`) and drops host RSS up to **90.2%** (`3,297.8 → 323.4 MiB`), while running `lm1b-large` b=16 **1.15× faster** (`2,113.3 → 1,838.3 ms`).
- **Quality & safety:**
  - `sudoku-easy`: **128/128 (100.00%)** exact boards in `bf16` and `fp16` on both CUDA and Metal.
  - `wmt14`: `fp16` matches **29/32 exact texts** (`98.93%` CUDA / `97.71%` Metal token match); `xsum`: `fp16` matches **29/32 CUDA / 27/32 Metal exact texts**.
  - `upload_f32` explicitly rejects finite weights exceeding IEEE 754 FP16 range (`65,504.0`) at load time (such as upstream TinyStories `vq.embeddings.0.weight` up to `2.48 × 10^10`), guiding users to `--precision bf16` or `--precision fp32`.

### 2. Drop-In OpenAI & Ollama Local Server (`crates/ncm-serve`, `ncm serve`, `ncm list`)
- Pure-Rust, zero-new-external-dependency HTTP/1.1 server (`std::net::TcpListener`, `serde`, `serde_json`), defaulting to localhost `127.0.0.1:11434`.
- **OpenAI drop-in endpoints:** `GET /v1/models`, `GET /v1/models/:id`, `POST /v1/completions`, `POST /v1/chat/completions` (non-streaming JSON and streaming `text/event-stream` SSE).
- **Ollama drop-in endpoints:** `GET /`, `GET /health`, `GET /api/version`, `GET /api/tags`, `POST /api/show`, `POST /api/generate`, `POST /api/chat` (streaming `application/x-ndjson` by default and `stream: false` JSON).
- **Safe GPU serving architecture:** auto-discovers verified local models (preferring self-contained `-v2` aliases and supporting `:latest` tags + `--model name=path` overrides), serves concurrent HTTP clients via a single serialized GPU worker thread with a bounded LRU model cache (`--max-loaded`), CORS headers, and 4 MiB request-body limits.
- Verified live end-to-end on both RTX 3060 Ti (`52.5 ms` warm XSum SSE, `42.0 ms` warm WMT14 Ollama chat, 4× concurrent clients) and M1 Ultra (`101.4 ms` warm XSum SSE, `87.9 ms` warm WMT14 Ollama chat, 4× concurrent clients).

### 3. Newbie 5-Minute Quickstart & CLI Ergonomics
- New inline `--prompt "Your text"` flag on `ncm sample` (no temporary JSON files required; automatically encodes and decodes text).
- New `ncm list [--models-dir DIR] [--json]` subcommand.
- New [docs/QUICKSTART.md](QUICKSTART.md) 5-minute guide and `tools/reference/quickstart_export.py` one-command wrapper (no CPython source build required; historical exact-hash reproducibility recipes preserved in [docs/USAGE.md](USAGE.md)).

## Full Evidence
- Dual-platform quality, memory, latency, and live serving report: [docs/results/2026-11-22-phase6-precision-serving/README.md](results/2026-11-22-phase6-precision-serving/README.md).
- Release scope: source-only release; no prebuilt binaries or weights are distributed.
