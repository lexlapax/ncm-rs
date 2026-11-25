# Release Notes — `v0.6.0`

Date: 2026-11-24 UTC
Theme: **Zero-Python Single-Binary `ncm pull`, Fused Attention (CUDA FlashAttention-2 + Metal Fast SDPA), and Multi-Profile Hot-Swappable LoRA Adapters**

`v0.6.0` completes **Phase 10** across both **Linux/CUDA** (RTX 3060 Ti) and **Apple Silicon/Metal** (M1 Ultra) while adding **zero new external crate dependencies** and preserving **100% byte-identical `FP32` defaults** across all 9 qualified inference profiles.

---

## Highlights

### 1. Pure-Rust `ncm pull` — Zero-Python Single Binary (Milestone 10A)
- Replaced the Python/PyTorch checkpoint conversion requirement with a bounded, streaming pure-Rust downloader and exporter (`crates/ncm-io/src/pytorch.rs`, `crates/ncm-io/src/yaml.rs`, `crates/ncm-tokenizers/src/convert.rs`, `crates/ncm-cli/src/pull.rs`). Legacy Python export remains available opt-in via `--python-legacy`.
- Reproduces all 4 canonical tokenizers (`tinystories`, `gpt2`, `qwen3`, `bert`) bit-for-bit against `tokenizers-v1.lock.json` using the already-vendored `tokenizers` crate.
- **Verified with zero Python in `PATH`:**
  - **Linux/CUDA (`PATH=/nonexistent`):** exported `sudoku-easy` (`0.54s`, 238/238 tensors), `tinystories-legacy` (`1.87s`, 304/304), `xsum` (`2.17s`, 275/275), `wmt14` (`4.01s`, 282/282), and self-contained `lm1b-small` v2 (`3.19s`, 351/351) — **1,450 / 1,450 bit-exact tensors** with byte-identical CUDA inference.
  - **Apple Silicon/Metal (`PATH` = `curl` only):** live HTTPS pull + export of `sudoku-easy` (`5.17s`, 238/238) and `tinystories` 15k EMA (`14.95s`, 304/304) with **100% byte-identical Metal inference**.

### 2. Fused Attention: CUDA Ampere Tensor-Core FlashAttention-2 & Metal Fast SDPA (Milestone 10B)
- **CUDA (`crates/ncm-backend-cuda/src/fused_attention.cu`):** custom SM80+ Tensor-Core FlashAttention-2 kernel (`mma.sync.aligned.m16n8k16.row.col.f32.{bf16,f16}`, `BR=64`, `BC=32`, `head_dim=64`, FP32 online `exp2f` softmax, zero $O(S^2)$ VRAM materialization, zero-branch mask specialization) wired for non-FP32 precisions.
- **Metal (`crates/ncm-backend-mlx/src/codec.rs`):** native `mlx_rs::fast::scaled_dot_product_attention` wired for non-FP32 precisions (`bf16`/`fp16`/`q8`/`q4`).
- **Paired before/after wins (with 9/9 FP32 CLI byte-equivalence untouched on both GPUs):**
  - **RTX 3060 Ti (CUDA):** `xsum` b=8 `bf16` **130.25 → 121.75 ms (1.070×, -36.0 MiB VRAM)** & `fp16` **130.82 → 123.21 ms (1.062×, -36.0 MiB VRAM)**; `tinystories` b=8 `bf16` **1.042×** / b=1 `bf16` **1.034×** (`26.67 ms`); `wmt14` b=8 `fp16` **1.038×**; `lm1b-large` b=8 `bf16` **1.036×** & b=16 `q4` **1.032×**.
  - **M1 Ultra (Metal):** `xsum` b=1 `bf16` **80.54 → 76.29 ms (1.056×, -63.1 MiB MLX peak)** & `fp16` **64.27 → 62.79 ms (1.024×, -63.1 MiB MLX peak)**; `wmt14` b=1 & b=8 `fp16` **1.024×** (`-29.8 MiB` at b=8); `lm1b-large` b=1 `fp16` **1.018×**.

### 3. `ncm-train` v3: Multi-Profile LoRA, Verified Unmerged Adapters & Hot-Swappable `ncm serve` (Milestone 10C)
- **Multi-profile `--init-from`:** `ncm train` now supports LoRA fine-tuning across all 4 upstream text families (`tinystories`, `wmt14`, `xsum`, and `lm1b-{small,medium,large}`), plus a native BF16 transformer training tape on CUDA so 560-token `xsum` trains cleanly inside 3.69 GiB free VRAM.
- **Verified unmerged adapter format (`ncm-lora-adapter-v1`, `--save-adapter DIR`):** exports a compact **4.53 MiB** adapter (`121.7×–240.0×` smaller than the base checkpoint) with `adapter_manifest.json` pinned to `base_manifest_sha256` and per-tensor SHA-256 verification (fail-closed on tamper, base mismatch, symlinks, or non-finite values).
- **QLoRA-style runtime inference (`--adapter DIR`):** `RuntimeLoraAdapter` applies pre-scaled `(alpha / rank) * B` over any base precision (`fp32`, `bf16`, `fp16`, `q8`, `q4`) in `ncm sample`, `ncm run`, and `ncm serve` across all 5 text families without modifying strict no-adapter FP32 inference.
- **Hot-swappable multi-adapter `ncm serve` (`--adapter ALIAS=BASE:DIR`):** serves multiple LoRA adapters alongside the base model on **one shared resident GPU base** (`--max-loaded 1`, `/api/ps` = `1` resident model) with instant refcounted `Arc` hot-swaps and **zero base reloads (`load_duration = 0`)**:
  - **RTX 3060 Ti (Q8 base):** bit-exact base restoration at **44.77 ms** and cached adapter swap at **48.02 ms** (`load_ns = 0`).
  - **M1 Ultra (Q8 base):** bit-exact base restoration at **81.72 ms** and cached adapter swap at **91.79 ms** (`load_ns = 0`).

---

## Full Evidence & Reports

- [Phase 10 Dual-Platform Report](results/2026-11-24-phase10/README.md) (all 6 CUDA & Metal receipts)
- [v0.5.0 Release Notes](RELEASE-NOTES-v0.5.0.md) · [v0.4.1 Release Notes](RELEASE-NOTES-v0.4.1.md) · [v0.4.0 Release Notes](RELEASE-NOTES-v0.4.0.md) · [Quickstart Guide](QUICKSTART.md) · [Usage Guide](USAGE.md)
