# Changelog

All notable releases of `ncm-rs` are documented here. Every release is validated on both **Linux/CUDA** (RTX 3060 Ti) and **Apple Silicon/Metal** (M1 Ultra) while preserving strict `FP32` upstream parity by default.

## [v0.7.0] — 2026-11-25

Theme: **Packaging, Distribution & `ncm doctor`** ([Release Notes](docs/RELEASE-NOTES-v0.7.0.md) · [Dual-Platform Report](docs/results/2026-11-24-phase11/README.md))

- **`ncm --version` & `ncm doctor [--self-test] [--json]`:** probes compiled backends, GPU/driver/library readiness, `curl`, local models & pre-quantized sidecars, and runs a zero-download FP32/BF16/Q8/fused-attention self-test (`128.15 ms` CUDA, `24.94 ms` Metal).
- **Global-install models & cache resolution:** unified path resolution (`--models-dir` > `NCM_MODELS_DIR` > `./artifacts/canonical` > `~/.local/share/ncm/models` on Linux / `~/Library/Application Support/ncm/models` on macOS) across `list`, `pull`, `run`, `serve`, and `doctor`.
- **Relocatable release packaging & installers:** added `tools/setup-mlx-native.sh`, `tools/package.sh` (self-contained `66.9 MB` Apple Silicon Metal bundle with zero `/Users/` paths + stripped-env verification; `16.5 MB` Linux CUDA bundle with `$ORIGIN/../lib` RUNPATH), `tools/install.sh`, and `tools/render_homebrew_formula.py` (published with prebuilt Linux/CUDA & macOS/Metal tarballs, `.sha256` checksums, and `ncm.rb`).

## [v0.6.0] — 2026-11-24

Theme: **Zero-Python Single-Binary `ncm pull`, Fused Attention & Hot-Swappable Multi-Profile LoRA** ([Release Notes](docs/RELEASE-NOTES-v0.6.0.md) · [Dual-Platform Report](docs/results/2026-11-24-phase10/README.md))

- **Pure-Rust `ncm pull` (10A):** bounded ZIP/pickle + YAML + streaming safetensors exporter and bit-exact canonical tokenizer builders (`tinystories`, `gpt2`, `qwen3`, `bert`) with zero Python in `PATH` (**1,450/1,450** bit-exact tensors on Linux/CUDA; live HTTPS `curl`-only pulls on Metal; `--python-legacy` retained opt-in).
- **Fused Attention (10B):** custom SM80+ CUDA Tensor-Core FlashAttention-2 (`mma.sync.aligned.m16n8k16`, FP32 online softmax, zero $O(S^2)$ VRAM) and native Metal Fast SDPA for non-FP32 precisions (**1.070× / -36.0 MiB** `xsum` b=8 `bf16` on CUDA; **1.056× / -63.1 MiB** `xsum` b=1 `bf16` on Metal) with **9/9 FP32 byte-equivalence** untouched.
- **Multi-Profile LoRA & Hot-Swappable `ncm serve` (10C):** `ncm train --init-from` expanded to `wmt14`, `xsum`, and `lm1b-{small,medium,large}` with `--save-adapter` exporting verified `4.53 MiB` `ncm-lora-adapter-v1` adapters (**121.7×–240.0×** smaller), runtime adapter application over `fp32`/`bf16`/`fp16`/`q8`/`q4` bases, and `ncm serve --adapter ALIAS=BASE:DIR` hot-swapping over **1 shared resident GPU base** (`load_ns = 0`, bit-exact base restoration).

## [v0.5.0] — 2026-11-24

Theme: **Native Low-Precision Kernels, Verified `Q8`/`Q4` Pre-Quantized Sidecars & `ncm-train` v2** ([Release Notes](docs/RELEASE-NOTES-v0.5.0.md) · [Dual-Platform Report](docs/results/2026-11-23-phase8-phase9/README.md))

- **Native Low-Precision Kernels & Fixed-Matrix Resampling (8A):** eliminated per-layer FP32 casts in `bf16`/`fp16`/`q8`/`q4` (**1.33×–1.77×** faster CUDA; **1.22×–1.72×** faster Metal). Eager Rust CUDA `bf16` b=1 on `tinystories-15k-ema` (`27.45 ms`, `253.6 MiB`) beats upstream Python `094f25ca` with `torch.compile` + CUDA graphs (`31.28 ms`, `647.9 MiB`) by **1.14×** at **2.55× less VRAM**.
- **Pre-Quantized `Q8`/`Q4` Sidecars (`ncm quantize`, 8B):** SHA-256-verified sidecars (`3.06×–5.45×` smaller on disk) load **4.79×–9.23× faster on CUDA** (`lm1b-large` `q4` `3,861 → 418.5 ms`) and **2.50×–4.75× faster on Metal** (`2,436 → 512.5 ms`) with 100% byte-identical outputs and fail-closed tamper detection.
- **`ncm-train` v2 (9A/9B):** upstream TinyStories `--init-from`, native Rust LoRA (`--lora-rank`, exact FP64 merge on export, 4-decimal parity across CUDA & Metal), `--train-precision bf16`, `--grad-accum`, `.jsonl` sliding-window datasets, and `--val-data`/`--eval-every`.

## [v0.4.1] — 2026-11-23

Theme: **Upstream `094f25ca` / `e34c3fab` Qualification (TinyStories 15k EMA)** ([Release Notes](docs/RELEASE-NOTES-v0.4.1.md) · [Qualification Report](docs/results/2026-11-23-upstream-qualification/README.md))

- Dual-platform qualified TinyStories 15k EMA release (`0/2,564` code diffs, `0/512` ID diffs on CUDA & Metal) and trimmed inference `config.yaml`, preserving backward-compatible verification for legacy `c7027427`/`d342ce8a` artifacts and advancing `upstream.toml`.

## [v0.4.0] — 2026-11-23

Theme: **Weight Quantization (`Q8`/`Q4`), Serving v2 & Native Rust Training (`ncm train`)** ([Release Notes](docs/RELEASE-NOTES-v0.4.0.md) · [Dual-Platform Report](docs/results/2026-11-23-phase7/README.md))

- **Native `Q8`/`Q4` Quantization (7A):** `GGML Q8_0`/`Q4_0` on CUDA and group-64 affine on Metal, cutting `lm1b-large` b=1 VRAM **-68.1% (`q8`) / -79.1% (`q4`, 673 MiB)** on CUDA (**b=16 in 975 MiB**) and **-66.9% / -77.2% MLX peak** (-90.2% RSS) on Metal.
- **Serving v2 & Ollama-Class CLI (7B):** added `ncm run` (one-shot + hot-GPU REPL), `ncm pull`, `GET /api/ps`, and dynamic micro-batching (`--max-batch 8`) delivering **2.21× CUDA / 1.85× Metal** throughput on `wmt14` and **1.64× / 1.97×** on `lm1b-large` Q8.
- **Native Rust Training (`crates/ncm-train`, 7C):** zero-new-dep CPU/CUDA/Metal reverse-mode autodiff for RVQ/STE/EMA codec + frozen-codec generator training, deterministic checkpoint save/resume, and verified `local-ncm-v1` export.

## [v0.3.0] — 2026-11-22

Theme: **Opt-In Half Precision (`BF16`/`FP16`) & OpenAI/Ollama Local Serving** ([Release Notes](docs/RELEASE-NOTES-v0.3.0.md) · [Dual-Platform Report](docs/results/2026-11-22-phase6-precision-serving/README.md))

- Added `--precision bf16|fp16` (with load-time FP16 range check), zero-new-dep OpenAI & Ollama HTTP/1.1 server (`crates/ncm-serve`, `ncm serve`, `ncm list`), inline `--prompt` + `--plain`, and the 5-Minute Quickstart (`docs/QUICKSTART.md`).

## [v0.2.0] — 2026-11-22

Theme: **Streaming Verified Loading, Prefix K/V Caching & Custom `local-ncm-v1`** ([Release Notes](docs/RELEASE-NOTES-v0.2.0.md) · [Optimization Report](docs/results/2026-11-20-phase5-optimization/README.md))

- Added parallel SHA-256 + streaming verified loading (`1.71×–4.65×` faster loads, up to `-2.8 GiB` CUDA HWM / `-49%` Metal RSS), FP32 prefix K/V caching (**3.24× CUDA / 2.42× Metal** faster XSum), self-contained `lm1b-*-inference-v2`, and config-driven `local-ncm-v1`.

## [v0.1.0] — 2026-11-21

Theme: **Initial Dual-Platform Verified Inference Release** ([Release Notes](docs/RELEASE-NOTES-v0.1.0.md) · [Phase 4 Audit](docs/results/2026-11-19-phase4-audit/phase4-final-audit-v1.md))

- Initial unified `ncm sample` CLI and verified FP32 inference across Sudoku (easy/medium/hard), TinyStories, XSum, WMT14, and LM1B (small/medium/large) on both Linux/CUDA and Apple Silicon/Metal.

[v0.7.0]: https://github.com/lexlapax/ncm-rs/releases/tag/v0.7.0
[v0.6.0]: https://github.com/lexlapax/ncm-rs/releases/tag/v0.6.0
[v0.5.0]: https://github.com/lexlapax/ncm-rs/releases/tag/v0.5.0
[v0.4.1]: https://github.com/lexlapax/ncm-rs/releases/tag/v0.4.1
[v0.4.0]: https://github.com/lexlapax/ncm-rs/releases/tag/v0.4.0
[v0.3.0]: https://github.com/lexlapax/ncm-rs/releases/tag/v0.3.0
[v0.2.0]: https://github.com/lexlapax/ncm-rs/releases/tag/v0.2.0
[v0.1.0]: https://github.com/lexlapax/ncm-rs/releases/tag/v0.1.0
