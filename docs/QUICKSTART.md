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

### Option A: Export an Upstream NCM Model (1 Command)

Any modern Python 3.10+ virtual environment with `torch`, `safetensors`, `pyyaml`, and `tokenizers` works:

```bash
python3 -m venv .venv
.venv/bin/pip install torch safetensors pyyaml tokenizers

# See available starter models:
python3 tools/reference/quickstart_export.py --list

# Preview the exact commands without downloading:
python3 tools/reference/quickstart_export.py --model wmt14 --python .venv/bin/python --dry-run

# Fetch & export WMT14 German->English (~1.06 GiB) or XSum news summarizer (~627 MiB):
python3 tools/reference/quickstart_export.py --model wmt14 --python .venv/bin/python
```

*(For self-contained `lm1b-*-inference-v2` without a sibling rotary file, pass `--reference-rope artifacts/canonical/lm1b-reference-rope-v1.safetensors`.)*

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

## 4. Run Your First Prompt (No Temp Files Needed)

Pass `--prompt "..."` directly to `ncm sample`. Add `--precision fp16` for ~2× lower GPU memory and faster inference (or keep the default `--precision fp32` for exact parity with upstream):

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
- `--device cuda-cudnn|cuda|mlx|cpu` — select backend explicitly (defaults to compiled best GPU)
- `--max-loaded N` — number of models kept hot in GPU memory via LRU eviction (default: `1`, ideal for 8 GiB GPUs; increase to `2–4` on 24 GiB+ GPUs or Mac Studio)
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

# Ollama model list
curl http://127.0.0.1:11434/api/tags

# Ollama generate (NDJSON streaming by default; pass "stream": false for single JSON)
curl http://127.0.0.1:11434/api/generate \
  -d '{"model":"wmt14","prompt":"Guten Morgen!","stream":false}'

# Ollama chat
curl http://127.0.0.1:11434/api/chat \
  -d '{"model":"wmt14","messages":[{"role":"user","content":"Guten Morgen!"}],"stream":false}'
```

---

## Next Steps

- [docs/USAGE.md](USAGE.md) — full CLI reference, pinned exact-hash export workflows, and platform env setup.
- [docs/DEVELOPING.md](DEVELOPING.md) — extending `ncm-rs`, adding custom `local-ncm-v1` models, and backend internals.
- [docs/results/2026-11-22-phase6-precision-serving/README.md](results/2026-11-22-phase6-precision-serving/README.md) — measured CUDA & Apple Silicon FP32/BF16/FP16 quality, latency, VRAM/RSS, and serving results.
