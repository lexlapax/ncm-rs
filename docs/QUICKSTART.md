# 5-Minute Quickstart

Run Neural Compression Models (NCM) natively in Rust on NVIDIA CUDA or Apple Silicon Metal — from CLI prompts or as a drop-in local **OpenAI & Ollama API server** — with **zero Python at runtime**.

> Looking for bit-exact historical manifest reproduction (exact upstream commit pins, CPython 3.14.9 source build, and audit checks)? See [docs/USAGE.md](USAGE.md).

---

## 1. Build `ncm`

### Option A: Linux with NVIDIA CUDA + cuDNN

Prerequisites: Rust (`>= 1.99.1`), CUDA Toolkit (`nvcc` in `PATH`), and cuDNN 9 shared libraries.

```bash
git clone https://github.com/lexlapax/ncm-rs.git
cd ncm-rs
cargo build --release --locked -p ncm-cli --features cuda-cudnn --bin ncm
```

*(If CUDA or cuDNN libraries are in non-default locations such as `/opt/cuda/lib64` or a Python venv's `nvidia/cudnn/lib`, either export `LD_LIBRARY_PATH` before running `ncm` or bake them in at build time with `RUSTFLAGS="-C link-arg=-Wl,-rpath,/opt/cuda/lib64 -C link-arg=-Wl,-rpath,/path/to/cudnn/lib"`.)*

### Option B: macOS with Apple Silicon (Metal / MLX)

Prerequisites: Rust (`>= 1.99.1`), Xcode Command Line Tools, and CMake (`brew install cmake`).

Build the prebuilt native MLX C library once (see [docs/USAGE.md](USAGE.md#apple-silicon-macos--mlx) for pinned commit details), then build `ncm`:

```bash
git clone https://github.com/lexlapax/ncm-rs.git
cd ncm-rs
export MLX_SYS_PREBUILT="$PWD/.reference/mlx-native"
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$PWD/.reference/mlx-native/lib"
cargo build --release --locked -p ncm-cli --features mlx --bin ncm
```

---

## 2. Get a Model

### Option A: Pull an Upstream NCM Model (`ncm pull` — Zero Python!)

`ncm pull` downloads, SHA-256 verifies, and exports upstream NCM checkpoints and canonical tokenizers **natively in Rust** — no Python or PyTorch environment required:

```bash
# See available starter models:
./target/release/ncm pull --list

# Preview the exact URLs and SHA-256 pins without downloading:
./target/release/ncm pull wmt14 --dry-run

# Pull & export WMT14 German->English (~1.06 GiB) or XSum news summarizer (~627 MiB):
./target/release/ncm pull wmt14
```

*(For self-contained `lm1b-*-inference-v2` without a sibling rotary file, pass `--reference-rope artifacts/canonical/lm1b-reference-rope-v1.safetensors`. The legacy Python exporter remains available opt-in via `--python-legacy`.)*

### Option B: Run Your Own Local Model (`local-ncm-v1`)

Have your own `.safetensors` weights + HuggingFace `tokenizer.json`? Export them with **Python stdlib only** (no PyTorch required):

```bash
python3 tools/reference/export_local_ncm.py \
  --config my_config.json \
  --codec my_codec.safetensors \
  --generator my_generator.safetensors \
  --tokenizer my_tokenizer.json \
  --output artifacts/canonical/my-local-model
```

See [docs/DEVELOPING.md](DEVELOPING.md#4-running-your-own-local-ncm-model-local-ncm-v1) for the complete `config.json` schema.

---

## 3. List Your Local Models

```bash
./target/release/ncm list
```

Example output:

```text
NAME                     PROFILE                          SIZE  TEXT  PATH
lm1b-small               lm1b-small-inference-v1     942.8 MiB  yes   artifacts/canonical/lm1b-small-inference-v1
tinystories              tinystories-inference-v1    551.7 MiB  yes   artifacts/canonical/tinystories-phase4-v1
wmt14                    wmt14-inference-v1           1.06 GiB  yes   artifacts/canonical/wmt14-inference-v1-final
xsum                     xsum-inference-v2           627.4 MiB  yes   artifacts/canonical/xsum-inference-rope-v2-final
```

---

## 4. Run Prompts Interactively (`ncm run`) or One-Shot

Use Ollama-style `ncm run` for either a one-shot prompt or an interactive REPL that keeps the model hot in GPU memory across turns:

```bash
# One-shot prompt with 8-bit quantization (~68% less VRAM, works on CUDA & Apple Silicon):
./target/release/ncm run wmt14 "Die Forschung zeigt deutliche Fortschritte bei lokalen KI-Modellen." --precision q8

# Interactive REPL (type /bye or Ctrl-D to exit; /seed N or /cfg S to adjust):
./target/release/ncm run wmt14 --precision fp16
```

You can also use `ncm sample` directly with explicit artifact paths and `--prompt` / `--plain`:

```bash
# Linux CUDA (use `mlx` instead of `cuda` / `cuda-cudnn` on Apple Silicon):
./target/release/ncm sample artifacts/canonical/wmt14-inference-v1-final cuda \
  --prompt "Die Forschung zeigt deutliche Fortschritte bei lokalen KI-Modellen." \
  --precision fp16 --plain
```

Output:

```text
The research shows a significant progress in local models.
```

*(Omit `--plain` to receive the full structured JSON object `{"codes": ..., "ids": ..., "texts": [...]}`.)*

**Precision guide:**
- `--precision fp32` (default): strict reference FP32, byte-identical to upstream parity baselines.
- `--precision fp16`: recommended for `wmt14`, `xsum`, `lm1b-{small,medium,large}`, and `sudoku-*` — cuts VRAM/MLX memory nearly in half (~47–48% reduction) and runs up to **1.47× faster** on Ampere+ Tensor Cores.
- `--precision bf16`: wide dynamic range (`±3.39 × 10^38`); required for `tinystories` half precision (whose upstream `vq.embeddings.0.weight` exceeds FP16's `65,504` range, which `ncm` explicitly catches on load).
- `--precision q8` / `--precision q4`: native 8-bit (`Q8_0` / MLX 8-bit) and 4-bit (`Q4_0` / MLX 4-bit) weight quantization cutting VRAM up to **-79%** (`lm1b-large` fits in **673 MiB** VRAM, and `tinystories` works cleanly in both `q8` and `q4`). For translation (`wmt14`), prefer `fp16` or `q8`.

---

## 5. Serve as a Drop-In OpenAI & Ollama Endpoint

Start the built-in, zero-dependency Rust HTTP/1.1 server on the default Ollama port (`127.0.0.1:11434`). If Ollama is already running on `11434`, pass `--port 11435`:

```bash
./target/release/ncm serve --precision fp16 --preload xsum
# Or if port 11434 is already in use:
# ./target/release/ncm serve --precision fp16 --preload xsum --port 11435
```

Optional flags:
- `--models-dir DIR` — scan a custom directory (default: `artifacts/canonical`)
- `--model myname=/path/to/artifact` — register an explicit model alias (repeatable)
- `--adapter alias=base:/path/to/adapter` — mount an unmerged `ncm-lora-adapter-v1` over a shared resident GPU base with zero-reload hot-swapping (repeatable)
- `--device cuda-cudnn|cuda|mlx|cpu` — select backend explicitly (defaults to compiled best GPU)
- `--max-loaded N` — number of models kept hot in GPU memory via LRU eviction (default: `1`, ideal for 8 GiB GPUs; increase to `2–4` on 24 GiB+ GPUs or Mac Studio)
- `--max-batch N` — max concurrent requests dynamically coalesced into one GPU forward pass (default: `8`; up to **2.21×** CUDA / **1.97×** Metal concurrent throughput)
- `--host 127.0.0.1 --port 11434` — bind address (localhost-only by default)

### Drop-In OpenAI Python SDK

Point any OpenAI-compatible client, app, or agent at `http://127.0.0.1:11434/v1`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ncm")

# Non-streaming chat completion
resp = client.chat.completions.create(
    model="wmt14",
    messages=[{"role": "user", "content": "Guten Morgen, wie geht es dir heute?"}],
)
print(resp.choices[0].message.content)

# Streaming SSE chat completion
stream = client.chat.completions.create(
    model="xsum",
    messages=[{"role": "user", "content": "Scientists in Geneva unveiled a fast native Rust model..."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
```

### Drop-In `curl` (OpenAI & Ollama)

```bash
# OpenAI model list
curl http://127.0.0.1:11434/v1/models

# OpenAI chat completion (SSE streaming)
curl -N http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"wmt14","messages":[{"role":"user","content":"Guten Morgen!"}],"stream":true}'

# Ollama model list & currently loaded models
curl http://127.0.0.1:11434/api/tags
curl http://127.0.0.1:11434/api/ps

# Ollama generate (NDJSON streaming by default; pass "stream": false for single JSON)
curl http://127.0.0.1:11434/api/generate \
  -d '{"model":"wmt14","prompt":"Guten Morgen!","stream":false}'

# Ollama chat
curl http://127.0.0.1:11434/api/chat \
  -d '{"model":"wmt14","messages":[{"role":"user","content":"Guten Morgen!"}],"stream":false}'
```

---

## 6. Pre-Quantize for Instant Cold Loads (`ncm quantize`)

Build a hash-verified `Q8` or `Q4` sidecar cache once (`3.06×–5.45×` smaller on disk); `ncm sample`, `ncm run`, and `ncm serve` automatically detect, SHA-256 verify, and load it (`4.79×–9.23×` faster on CUDA, `2.50×–4.75×` faster on Metal) with 100% byte-identical outputs:

```bash
# Pre-quantize for CUDA (GGML Q8_0 / Q4_0) or Metal (group-64 affine):
./target/release/ncm quantize artifacts/canonical/tinystories-15k-ema-v1 --precision q8 --device cuda
# (Use --device mlx on Apple Silicon)

# Subsequent Q8 runs automatically use the verified sidecar:
./target/release/ncm run tinystories "Once upon a time" --precision q8
```

---

## 7. Train or Fine-Tune a Local NCM Model Natively in Rust (`ncm train`)

Train a `local-ncm-v1` codec (RVQ + straight-through + EMA codebooks) and/or generator directly on **CUDA**, **Apple Silicon Metal**, or **CPU** — zero Python required — or LoRA fine-tune **any upstream text checkpoint** (`tinystories`, `wmt14`, `xsum`, or `lm1b-{small,medium,large}`) and export either a compact **4.5 MiB verified unmerged adapter** (`--save-adapter`) or a standalone merged checkpoint (`--output`):

```bash
# LoRA fine-tune any upstream checkpoint (TinyStories, WMT14, XSum, or LM1B) into a ~4.5 MiB verified adapter:
./target/release/ncm train \
  --init-from artifacts/canonical/tinystories-15k-ema-v1 \
  --data train.jsonl --val-data val.jsonl --eval-every 10 \
  --save-adapter artifacts/adapters/tinystories-pirate \
  --device cuda --lora-rank 8 --lora-alpha 16 --train-precision bf16 \
  --steps 50 --batch 2 --grad-accum 2 --lr 3e-4 --overwrite

# Run the unmerged adapter directly over a Q8/BF16/FP16/FP32 base (zero-copy base weights!):
./target/release/ncm run tinystories "Once upon a time" \
  --precision q8 --adapter artifacts/adapters/tinystories-pirate

# Or hot-swap multiple LoRA adapters over ONE shared Q8 base in ncm serve (zero base reloads!):
./target/release/ncm serve --precision q8 --max-loaded 1 --preload tinystories \
  --adapter pirate=tinystories:artifacts/adapters/tinystories-pirate \
  --adapter scifi=tinystories:artifacts/adapters/tinystories-scifi \
  --port 11435

# Or train codec + generator from scratch and export a standalone verified local-ncm-v1 artifact:
./target/release/ncm train \
  --config my_config.json \
  --data train.json \
  --output artifacts/canonical/my-trained-ncm \
  --device cuda \
  --stage all --steps 200 --batch 8 --lr 0.005 --overwrite
```

*(Replace `--device cuda` with `--device mlx` on Apple Silicon, or `--device cpu` for FP32 CPU tests. JSONL files accept `{"text": "..."}`, `{"prompt": "...", "completion": "..."}`, or `{"prefix": [...], "target": [...]}` with automatic 50%-overlap sliding-window chunking.)*

---

## Next Steps

- [docs/USAGE.md](USAGE.md) — full CLI reference, pinned exact-hash export workflows, and platform env setup.
- [docs/DEVELOPING.md](DEVELOPING.md) — extending `ncm-rs`, adding custom `local-ncm-v1` models, and backend internals.
- [docs/results/2026-11-24-phase10/README.md](results/2026-11-24-phase10/README.md) — measured CUDA & Apple Silicon Phase 10 zero-Python `ncm pull`, fused attention, multi-profile LoRA, and hot-swappable `ncm serve` results.
- [docs/results/2026-11-23-phase8-phase9/README.md](results/2026-11-23-phase8-phase9/README.md) — measured CUDA & Apple Silicon Phase 8 & Phase 9 low-precision kernel, `ncm quantize` sidecar, LoRA, and `BF16` training results.
- [docs/results/2026-11-23-phase7/README.md](results/2026-11-23-phase7/README.md) — measured CUDA & Apple Silicon `Q8`/`Q4` quantization, Serving v2 micro-batching, and native Rust training results.
- [docs/results/2026-11-22-phase6-precision-serving/README.md](results/2026-11-22-phase6-precision-serving/README.md) — measured CUDA & Apple Silicon FP32/BF16/FP16 quality, latency, VRAM/RSS, and serving results.
