# Release Notes — `v0.7.0`

Date: 2026-11-25 UTC
Theme: **Packaging, Zero-Friction Distribution, Global Models Directory, and `ncm doctor`**

`v0.7.0` completes **Phase 11** across both **Linux/CUDA** (RTX 3060 Ti) and **Apple Silicon/Metal** (M1 Ultra) while adding **zero new external crate dependencies** and preserving **100% byte-identical `FP32` defaults** across all 9 qualified inference profiles.

---

## Highlights

### 1. `ncm --version` & `ncm doctor [--self-test] [--json]`
- Added `ncm --version` (`-V`) reporting semantic version, OS/architecture, and compiled runtime features.
- Added `ncm doctor [--models-dir DIR] [--device DEV] [--self-test] [--json]` (`crates/ncm-cli/src/doctor.rs`) to inspect:
  - Binary metadata, compiled features, default runtime device, and live GPU/driver/library readiness (`nvidia-smi` + CUDA/cuDNN on Linux; chip, unified memory, macOS version, Metal/MLX, and `mlx.metallib` on Apple Silicon).
  - External `curl` availability for pure-Rust `ncm pull`.
  - Resolved local models directory, discovered model profiles, pre-quantized `Q8`/`Q4` sidecars, and total disk footprint.
  - **`--self-test`:** zero-download synthetic operator check exercising `FP32` linear + attention, `BF16` linear + SM80+ FlashAttention-2 / Metal Fast SDPA, and `Q8` quantized matmul directly on the active GPU (`128.15 ms` on RTX 3060 Ti, `24.94 ms` on M1 Ultra).

### 2. Global-Install Models & Upstream Cache Resolution (`NCM_MODELS_DIR`)
- Added unified path resolution (`crates/ncm-cli/src/paths.rs`) wired across `ncm list`, `ncm pull`, `ncm run`, `ncm serve`, and `ncm doctor`:
  1. Explicit `--models-dir` / `--upstream-dir` CLI flag
  2. `NCM_MODELS_DIR` / `NCM_UPSTREAM_DIR` environment variable
  3. `./artifacts/canonical` (and `./artifacts/upstream`) when running inside a repository checkout (100% backward-compatible)
  4. Standard user directories (`$XDG_DATA_HOME/ncm/models` or `~/.local/share/ncm/models` on Linux; `~/Library/Application Support/ncm/models` on macOS) so globally installed `ncm` binaries work cleanly from any directory.

### 3. Relocatable Release Packaging, Installers & Homebrew Formula
- **`tools/setup-mlx-native.sh`:** one-command reproducible `.reference/mlx-native` extractor on Apple Silicon from pinned `tools/reference/requirements-macos.lock`.
- **`tools/package.sh`:** builds SHA-256-verified relocatable release archives and self-verifies them in an isolated temp directory:
  - **macOS (`ncm-v<ver>-macos-arm64-mlx.tar.gz`, `66.9 MB`):** completely self-contained bundle with `bin/ncm` (`@executable_path/../lib`), `lib/libmlx.dylib` (`@loader_path`), `lib/libjaccl.dylib`, and `lib/mlx.metallib`, ad-hoc signed, audited to contain **zero `/Users/` build-host paths**, and verified under a stripped environment (`PATH=/usr/bin:/bin:/usr/sbin:/sbin`, no `MLX_SYS_PREBUILT`, no `DYLD_LIBRARY_PATH`).
  - **Linux (`ncm-v<ver>-linux-x86_64-cuda.tar.gz`, `16.5 MB`):** relocatable `bin/ncm` with `$ORIGIN/../lib:/usr/local/cuda/lib64:/opt/cuda/lib64` RUNPATH plus `README-INSTALL.txt` (does not redistribute proprietary NVIDIA CUDA/cuDNN `.so` binaries).
- **`tools/install.sh`:** SHA-256-verifying installer supporting `--archive PATH` or local builds into `~/.local` (or `--prefix DIR`).
- **`tools/render_homebrew_formula.py`:** deterministic Apple Silicon Homebrew formula renderer (unit-tested in `tools/check.sh` and verified with `/usr/bin/ruby -c` on macOS).

---

## Full Evidence & Reports

- [Phase 11 Dual-Platform Report](results/2026-11-24-phase11/README.md) (CUDA & Metal receipts)
- [v0.6.0 Release Notes](RELEASE-NOTES-v0.6.0.md) · [v0.5.0 Release Notes](RELEASE-NOTES-v0.5.0.md) · [Quickstart Guide](QUICKSTART.md) · [Usage Guide](USAGE.md)
