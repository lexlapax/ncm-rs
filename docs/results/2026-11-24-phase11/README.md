# Phase 11 Dual-Platform Report — Packaging, Distribution & `ncm doctor`

Date: 2026-11-25 UTC
Plan: [docs/PHASE11.md](../../PHASE11.md)
Hardware:
- **Linux / CUDA:** NVIDIA GeForce RTX 3060 Ti (8 GiB, SM86, CUDA driver 620.73.04)
- **macOS / Metal:** Apple M1 Ultra (64 GiB unified memory, macOS 26.7.4, native MLX)

Phase 11 passed on **both** Linux/CUDA and Apple Silicon/Metal with **zero new external crates** and **100% unchanged `FP32` inference defaults**.

## Raw Receipts

| Platform | Receipt |
| --- | --- |
| Linux / CUDA (`x86_64`, `cuda-cudnn`) | [cuda-phase11-v1.json](cuda-phase11-v1.json) |
| Apple Silicon / Metal (`arm64`, `mlx`) | [mlx-phase11-v1.json](mlx-phase11-v1.json) |

---

## Verification Summary

| Check | Linux / CUDA (RTX 3060 Ti) | Apple Silicon / Metal (M1 Ultra) |
| --- | --- | --- |
| Release archive (`tools/package.sh`) | `ncm-v0.6.0-linux-x86_64-cuda.tar.gz` (**16.5 MB**) | `ncm-v0.6.0-macos-arm64-mlx.tar.gz` (**66.9 MB**, self-contained w/ `libmlx.dylib`, `libjaccl.dylib`, `mlx.metallib`) |
| Relocatable RPATH / RUNPATH | `[$ORIGIN/../lib:/usr/local/cuda/lib64:/opt/cuda/lib64]` | `@executable_path/../lib` (zero `/Users/` build-host paths in `otool -L`/`otool -l`) |
| `tools/install.sh --archive` (isolated prefix) | Passed (SHA-256 verified) | Passed (SHA-256 verified, `bin/` + `lib/` installed) |
| `ncm --version` (from `/tmp`) | `ncm 0.6.0 (linux/x86_64; features: cpu, cuda, cuda-cudnn)` | `ncm 0.6.0 (macos/aarch64; features: mlx)` |
| `models_dir` resolution modes verified | `user-data-dir`, `env:NCM_MODELS_DIR` (35 models), `repo:artifacts/canonical` | `user-data-dir`, `env:NCM_MODELS_DIR` (11 models), `repo:artifacts/canonical` |
| `ncm doctor --self-test --json` | **PASS in 128.15 ms** | **PASS in 24.94 ms** (stripped env: no `MLX_SYS_PREBUILT`, no `DYLD_LIBRARY_PATH`) |
| Self-test `FP32` linear / attention max abs | `7.15e-7` / `3.35e-8` | `1.58e-6` / `3.73e-8` |
| Self-test `BF16` fused attention / Fast SDPA max abs | `1.05e-3` (SM80+ FlashAttention-2) | `8.25e-4` (Metal Fast SDPA) |
| Self-test `Q8` linear max abs | `4.10e-2` (GGML `Q8_0`) | `3.19e-2` (MLX group-64 affine) |
| Installed binary outside repo (`ncm run wmt14 "Guten Morgen!" --precision fp16`) | `"Good morning!"` | `"Good morning!"` (completely stripped env) |
| Homebrew formula (`tools/render_homebrew_formula.py`) | 3/3 Python unit tests | Rendered with real archive SHA-256; `/usr/bin/ruby -c` → **`Syntax OK`** |
| Shell lint (`bash -n` + `shellcheck`) | Passed on all 4 scripts in `tools/check.sh` | Passed via `./tools/check.sh --mlx` |

## Packaging & License Boundaries

- **Apple Silicon (`macos-mlx`):** bundles MIT-licensed native MLX runtime assets (`libmlx.dylib`, `libjaccl.dylib`, `mlx.metallib`) alongside `bin/ncm`, rewrites RPATHs to `@executable_path/../lib` / `@loader_path`, ad-hoc signs the Mach-O binaries, and fails closed if any `/Users/` path leaks into linked dependencies or RPATH entries. Runs from any directory with zero Xcode, Rust, CMake, or Python installed.
- **Linux (`linux-cuda`):** ships `bin/ncm` with relocatable `$ORIGIN/../lib:/usr/local/cuda/lib64:/opt/cuda/lib64` RUNPATH plus `LICENSE`, `NOTICE`, and `README-INSTALL.txt`. Proprietary NVIDIA CUDA/cuDNN `.so` binaries are intentionally not redistributed; `ncm doctor` checks driver, CUDA/cuDNN readiness, and prints actionable remediation hints if missing.
