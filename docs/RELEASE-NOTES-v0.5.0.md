# Release Notes — `v0.5.0`

Date: 2026-11-24 UTC
Theme: **Native Low-Precision Kernels, Verified `Q8`/`Q4` Pre-Quantized Sidecars (`ncm quantize`), and `ncm-train` v2 (LoRA + `BF16` + JSONL)**

`v0.5.0` completes **Phase 8** and **Phase 9** across both **Linux/CUDA** (RTX 3060 Ti) and **Apple Silicon/Metal** (M1 Ultra) while preserving **100% byte-identical `FP32` defaults** across all 9 qualified inference profiles and adding zero external crate dependencies.

---

## Highlights

### 1. Native Low-Precision Kernels & Fixed-Matrix Resampling (Milestone 8A)
- Eliminated per-layer FP32 round-trip casts in `bf16`, `fp16`, `q8`, and `q4` across both `ncm-backend-cuda` and `ncm-backend-mlx`:
  - **CUDA:** native `layernorm_{bf16,f16}`, `rmsnorm_{bf16,f16}`, `ugelu_erf_{bf16,f16}`, `rope_{bf16,f16}`, native `BF16` activations into `QMatMul::forward` (`fast_mmvq`/`fast_mmq`), custom fused FP32-accumulating `ncm_fused_softmax_{bf16,f16}`, 16-bit `ncm_pack_qkv_u16`/`ncm_merge_heads_u16`, and cached fixed-matrix resampling GEMMs.
  - **Metal:** native `bf16`/`fp16`/`q8`/`q4` norms, RoPE, GELU, attention, and quantized matmuls plus cached fixed-matrix resampling GEMMs.
- **Measured warm-inference speedups vs `v0.4.1`:**
  - **RTX 3060 Ti (CUDA): `1.33×–1.77×`** (`sudoku-easy` `bf16` `30.99 → 17.55 ms`, `1.77×`; `tinystories-15k-ema` b=1 `bf16` `42.83 → 27.45 ms`, `1.56×`; `xsum` `bf16` `47.44 → 30.34 ms`, `1.56×`; `wmt14` `bf16` `39.10 → 25.50 ms`, `1.53×`; `lm1b-large` `bf16` `190.52 → 131.77 ms`, `1.45×`, `q4` `194.10 → 136.96 ms`, `1.42×`).
  - **M1 Ultra (Metal): `1.22×–1.72×`** (`sudoku-easy` `fp16` `57.05 → 33.24 ms`, `1.72×`; `wmt14` `fp16` `80.13 → 54.82 ms`, `1.46×`; `xsum` `fp16` `84.58 → 62.76 ms`, `1.35×`; `tinystories-15k-ema` b=1 `q8` `101.12 → 76.54 ms`, `1.32×`; `lm1b-large` `q4` `453.62 → 359.55 ms`, `1.26×`).
- **Head-to-head vs upstream Python `094f25ca` (`tinystories-15k-ema`, RTX 3060 Ti):** eager Rust CUDA `bf16` b=1 (**27.45 ms, 253.6 MiB VRAM**) beats upstream Python `094f25ca` **with `torch.compile` + CUDA graphs** (**31.28 ms, 647.9 MiB VRAM**) by **1.14×** at **2.55× less VRAM**.

### 2. Verified Pre-Quantized `Q8`/`Q4` Sidecar Cache (`ncm quantize`, Milestone 8B)
- Added `ncm quantize ARTIFACT --precision q8|q4 --device cuda|mlx [--force]`, writing a parent-manifest-bound, parallel SHA-256-verified sidecar (`prequant-{cuda-ggml|mlx-affine}-{q8|q4}.{safetensors,manifest.json}`) atomically inside the artifact directory.
- `ncm sample`, `ncm run`, and `ncm serve` automatically verify and load matching sidecars (`100%` byte-identical output vs on-the-fly quantization), fall back cleanly only when absent, and **fail closed** on 1-byte tamper, stale parent `manifest_sha256`, missing/incomplete sidecar, or symlinks.
- **Measured cold-load speedups & disk savings:**
  - `tinystories-15k-ema-v1`: `q8` **164.2 MB** (`3.06×` smaller) loads **4.79× faster on CUDA** (`625.8 → 130.5 ms`) / **2.50× on Metal** (`391.8 → 156.9 ms`); `q4` **117.5 MB** (`4.27×` smaller) loads **5.24× CUDA** (`524.9 → 100.1 ms`) / **3.11× Metal** (`382.8 → 123.1 ms`).
  - `lm1b-large-inference-v1`: `q8` **982.7 MB** (`3.41×` smaller) loads **7.07× faster on CUDA** (`4,660.7 → 659.7 ms`) / **3.12× on Metal** (`2,466.5 → 791.6 ms`); `q4` **613.9 MB** (`5.45×` smaller) loads **9.23× CUDA** (`3,861.2 → 418.5 ms`) / **4.75× Metal** (`2,436.0 → 512.5 ms`).

### 3. `ncm-train` v2: Native Rust LoRA, Upstream TinyStories `--init-from`, `BF16`, Gradient Accumulation & JSONL (Milestones 9A & 9B)
- **Upstream TinyStories `--init-from`:** `ncm train --init-from` now admits verified upstream `tinystories-inference-v1` artifacts (converting tied codebook heads + `level0_unigram` to mathematically equivalent `local-ncm-v1` `UntiedAffine` weights in parallel), and `LocalTokenizer` loads embedded TinyStories `ByteBpe` vocabularies offline.
- **Native Rust LoRA (`--lora-rank R --lora-alpha A`):** trains low-rank adapters across all generator transformer attention/MLP weights (`c_attn`, `c_proj`, `c_fc`) with per-level backward accumulation, frozen-weight detach, deterministic checkpoint resume, and exact FP64-accumulated `W + (alpha / r) * (B @ A)` merge on export (zero inference overhead).
  - Live **123.16M `tinystories-15k-ema-v1` LoRA r=8** (`0.96%` trainable, `batch=2`, 30 steps): loss **6.5286 → 3.4522** (exact 4-decimal parity between CUDA & Metal; `542.0 ms/step` on RTX 3060 Ti fitting inside 3.7 GiB free VRAM; `352.1 ms/step` on M1 Ultra), with verified merged `ncm inspect`, `ncm sample`, `ncm run`, and `ncm quantize q8` roundtrip.
- **Mixed-precision `BF16` training (`--train-precision bf16`), `--grad-accum N`, `.jsonl` datasets & `--val-data`/`--eval-every`:**
  - `.jsonl`/`.ndjson` loader supports `{"text"}`, `{"prompt","completion"}`, and `{"prefix","target"}` with 50%-overlap sliding-window chunking.
  - Exact gradient accumulation (`batch=4, accum=1` vs `batch=2, accum=2` `< 1e-5` on both codec and generator) and detached no-grad validation evaluation.
  - Live 123.16M `tinystories-15k-ema-v1` JSONL train+val (`batch=2, grad_accum=2`, 20 steps): `fp32` train **6.5840 → 3.9293** / val **4.6507 → 4.0802** (4-decimal parity across CUDA `1,189.8 ms/step` & Metal `724.3 ms/step`); `bf16` matches within `< 1e-3` (`1.05×` faster & `~576 MiB` lower peak VRAM on RTX 3060 Ti; `0.85×` on M1 Ultra which predates M2's hardware BF16 ALU).

---

## Full Evidence & Reports

- [Phase 8 & Phase 9 Dual-Platform Report](results/2026-11-23-phase8-phase9/README.md) (all 8 CUDA & Metal receipts)
- [v0.4.1 Release Notes](RELEASE-NOTES-v0.4.1.md) · [v0.4.0 Release Notes](RELEASE-NOTES-v0.4.0.md) · [Quickstart Guide](QUICKSTART.md) · [Usage Guide](USAGE.md)
