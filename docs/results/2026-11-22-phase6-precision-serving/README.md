# Phase 6 Results: Half Precision (`BF16`/`FP16`) & OpenAI/Ollama Serving

**Date:** 2026-11-22
**Platforms verified:**
- **Linux / NVIDIA CUDA:** RTX 3060 Ti (8 GiB GDDR6, SM86 Ampere), CUDA 13.2 + cuDNN 9, `llama-server.service` off.
- **macOS / Apple Silicon Metal:** Mac Studio (M1 Ultra, 64 GiB unified memory), native MLX C backend (`dotsetup.rmlx` idle/untouched).

**Machine-readable artifacts:**
- [cuda-precision-v1.json](cuda-precision-v1.json)
- [mlx-precision-v1.json](mlx-precision-v1.json)
- [cuda-serve-e2e-v1.json](cuda-serve-e2e-v1.json)
- [mlx-serve-e2e-v1.json](mlx-serve-e2e-v1.json)

---

## 1. Opt-In Half Precision (`--precision bf16|fp16`)

`ComputePrecision::Fp32` remains the default everywhere and executes the unchanged FP32 path. Opt-in `bf16` and `fp16` store weights, KV caches, and activations in 16-bit while accumulating norms (`layer_norm`, `rms_norm`, `group_norm`, `rms_rope`), resampling, convolutions on CUDA, codebook norm reductions, categorical sampling, and FP64 decoding penalties in FP32/FP64.

### Quality vs. Strict FP32

| Profile | Eval Set | CUDA `bf16` vs `fp32` | CUDA `fp16` vs `fp32` | Metal `bf16` vs `fp32` | Metal `fp16` vs `fp32` |
| --- | --- | --- | --- | --- | --- |
| `sudoku-easy` | 128 boards (11,392 cells) | **128/128 (100.00%)** | **128/128 (100.00%)** | **128/128 (100.00%)** | **128/128 (100.00%)** |
| `wmt14` | 32 prefixes (2,048 tokens) | 18/32 exact texts (87.45% tok) | **29/32 exact texts (98.93% tok)** | 17/32 exact texts (89.21% tok) | **29/32 exact texts (97.71% tok)** |
| `xsum` | 32 prefixes (1,536 tokens) | 16/32 exact texts (75.98% tok) | **29/32 exact texts (97.59% tok)** | 16/32 exact texts (78.91% tok) | **27/32 exact texts (92.51% tok)** |
| `tinystories` | Seeded generation | Finite fluent output | **Explicit load rejection** (`> 65504.0`)* | Finite fluent output | **Explicit load rejection** (`> 65504.0`)* |
| `lm1b-large` | Seed 42 generation | Finite fluent output | Finite fluent output | Finite fluent output | Finite fluent output |

\* *Upstream `tinystories-phase4-v1` `codec.safetensors` stores `vq.embeddings.0.weight` with magnitudes up to `2.48 × 10^10` (exceeding IEEE 754 FP16 max finite `65,504.0`). Both CUDA and MLX backends detect this during verified `upload_f32` and fail closed with an explicit message recommending `--precision bf16` or `--precision fp32` rather than silently producing `Inf`/`NaN`.*

### Linux / CUDA Latency & VRAM Peak (RTX 3060 Ti, 8 GiB)

| Profile | Batch | `fp32` warm (ms) | `bf16` warm (ms) | `fp16` warm (ms) | `fp16` speedup | `fp32` VRAM (MiB) | `bf16`/`fp16` VRAM (MiB) | VRAM reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sudoku-easy` | 1 | 25.81 | 30.90 | **25.24** | 1.02× | 168.8 | **100.1** | **-40.7%** |
| `tinystories` | 1 | 41.05 | 42.47 | *N/A (`>65504`)* | — | 492.9 | **253.6** (`bf16`) | **-48.5%** |
| `xsum` | 1 | 47.67 | 47.14 | **40.10** | **1.19×** | 686.1 | **364.6** | **-46.9%** |
| `wmt14` | 1 | 36.32 | 38.91 | **32.92** | **1.10×** | 1,050.2 | **559.6** | **-46.7%** |
| `lm1b-small` | 1 | 41.68 | 50.47 | 43.33 | 0.96× | 820.2 | **470.4** | **-42.6%** |
| `lm1b-medium` | 1 | 117.32 | 106.91 | **93.20** | **1.26×** | 1,657.3 | **889.0** | **-46.4%** |
| `lm1b-large` | 1 | 246.18 | 191.09 | **167.71** | **1.47×** | 3,215.6 | **1,667.7** | **-48.1%** |
| `lm1b-large` | 16 | *(>3.5 GiB FP32)* | 1,561.42 *(97.6 ms/seq)* | **1,450.76** *(90.7 ms/seq)* | — | — | **1,955.1** | Fits 8 GiB GPU |

### Apple Silicon / Metal Latency, MLX Peak & Host RSS (M1 Ultra)

Eagerly materializing half-precision casts in `MlxBackend::upload_f32` releases temporary FP32 upload arrays per tensor rather than retaining the full FP32 model graph, cutting both MLX GPU peak and host process RSS dramatically:

| Profile | Batch | `fp32` warm (ms) | `bf16` warm (ms) | `fp16` warm (ms) | `fp32` MLX peak (MiB) | `fp16`/`bf16` MLX peak (MiB) | `fp32` RSS (MiB) | `fp16`/`bf16` RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sudoku-easy` | 1 | 54.17 | 61.06 | 56.88 | 261.7 | **192.3** (-26.5%) | 197.0 | **76.2** (-61.3%) |
| `tinystories` | 1 | 93.42 | 97.50 | *N/A* | 606.6 | **359.1** (-40.8%) | 526.6 | **93.4** (-82.3%) |
| `xsum` | 1 | 86.59 | 96.20 | **84.17** | 930.6 | **566.9** (-39.1%) | 1,015.9 | **555.1** (-45.4%) |
| `wmt14` | 1 | 79.15 | 86.48 | 79.82 | 1,104.2 | **592.0** (-46.4%) | 1,121.4 | **550.3** (-50.9%) |
| `lm1b-small` | 1 | 96.33 | 101.71 | **94.14** | 926.7 | **518.2** (-44.1%) | 878.9 | **243.8** (-72.3%) |
| `lm1b-medium` | 1 | 203.22 | 219.79 | **194.44** | 1,806.4 | **982.6** (-45.6%) | 1,726.4 | **269.1** (-84.4%) |
| `lm1b-large` | 1 | 380.38 | 422.84 | **362.35** | 3,414.4 | **1,798.4** (-47.3%) | 3,297.8 | **323.4** (-90.2%) |
| `lm1b-large` | 16 | 2,113.30 | 2,137.34 | **1,838.34** (**1.15×**) | 4,683.1 | **2,978.1** (-36.4%) | 3,329.0 | **362.6** (-89.1%) |

---

## 2. Live OpenAI & Ollama Serving (`ncm serve`, `ncm list`, `--prompt`)

Both platforms passed offline unit/integration suites (`crates/ncm-serve/tests/http_api.rs`) and live end-to-end HTTP verification with `--precision fp16`:

| Check / Endpoint | Linux / CUDA (`cuda-cudnn`, `fp16`) | Apple Silicon / Metal (`mlx`, `fp16`) |
| --- | --- | --- |
| `ncm list` model auto-discovery | PASS (34 local artifacts discovered) | PASS (9 canonical artifacts discovered) |
| `ncm sample ... --prompt "..."` (inline WMT14) | PASS (`"The research shows a significant progress in local models."`) | PASS (`"The research shows a significant progress in local models."`) |
| `GET /`, `GET /health`, `GET /v1/models`, `GET /api/tags` | PASS | PASS |
| `POST /v1/chat/completions` (`xsum` warm, non-stream) | PASS (`197.1 ms` first warm) | PASS (`148.2 ms` first warm) |
| `POST /v1/chat/completions` (`xsum` warm, **SSE `stream: true`**) | PASS (**52.5 ms**, exact match) | PASS (**101.4 ms**, exact match) |
| `POST /api/generate` (`wmt14:latest`, **NDJSON streaming** + cold load/LRU) | PASS (`1,244.3 ms`, `56.1 ms` eval) | PASS (`883.7 ms`) |
| `POST /api/chat` (`wmt14` warm, `stream: false`) | PASS (**42.0 ms**) | PASS (**87.9 ms**) |
| `POST /v1/completions` (`lm1b-small` LRU switch) | PASS (`651.5 ms`) | PASS (`688.3 ms`) |
| 4 concurrent HTTP clients (`lm1b-small` warm) | PASS (`48.4–173.0 ms`, all `200 OK`) | PASS (`99.5–382.2 ms`, all `200 OK`) |
| Unknown model (`404`) & malformed JSON (`400`) & body >4 MiB (`413`) | PASS | PASS |
