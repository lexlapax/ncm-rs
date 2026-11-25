# Phase 11 Plan — Packaging, Distribution & `ncm doctor` (`v0.6.0` → `v0.7.0`)

Date: 2026-11-24 UTC
Status: **Approved by user; executing Linux/CUDA first, then Apple Silicon/Metal per milestone.**

## Goals

Make `ncm` effortless to install, diagnose, and run from **any working directory** on both Linux/CUDA and Apple Silicon/Metal, without adding external crate dependencies, without restoring GitHub build CI, and without touching strict `FP32` inference semantics.

## Milestone 11A — CLI Ergonomics (`ncm --version`, `NCM_MODELS_DIR` / User Data Dir, & `ncm doctor`)

1. **`ncm --version` / `-V`:** print semantic version, OS/arch, and compiled runtime features cleanly.
2. **Global-install-friendly models directory resolution (`crates/ncm-cli/src/paths.rs`):**
   - Precedence:
     1. Explicit CLI flag (`--models-dir` / `--output-root`)
     2. `NCM_MODELS_DIR` environment variable (when non-empty)
     3. `./artifacts/canonical` if it exists in the current working directory (100% backward-compatible for repo checkouts)
     4. Standard user data dir: `$XDG_DATA_HOME/ncm/models` (`~/.local/share/ncm/models`) on Linux; `~/Library/Application Support/ncm/models` on macOS.
   - Wired consistently across `ncm list`, `ncm pull`, `ncm run`, `ncm serve`, and `ncm doctor`.
3. **`ncm doctor [--models-dir DIR] [--self-test] [--json]` (`crates/ncm-cli/src/doctor.rs`):**
   - Reports version, OS/arch, executable path, compiled features, default device, GPU/driver/library readiness (CUDA/cuDNN or Metal/MLX with actionable fix hints when libraries are missing), `curl` availability for `ncm pull`, and resolved local models inventory (valid/invalid artifacts, `Q8`/`Q4` prequant sidecars, disk usage).
   - `--self-test` executes a fast (<200 ms), zero-download in-memory backend operator & precision check (`FP32`, `BF16`, `Q8` linear + masked softmax + fused attention/SDPA) on the active GPU/CPU runtime and exits non-zero on failure.
4. **Validation Gate (Linux/CUDA first, then Apple Silicon/Metal):**
   - `./tools/check.sh` (unit tests for `--version`, `paths.rs` precedence, `doctor` CLI parsing & CPU `--self-test`).
   - Platform Clippy (`-D warnings`) & release build (`cuda-cudnn` on Linux, `mlx` on macOS).
   - Live `ncm --version`, `ncm doctor --self-test --json`, and isolated CWD `NCM_MODELS_DIR` E2E test.
   - Receipts: `docs/results/2026-11-24-phase11/{cuda,mlx}-milestone11a-v1.json`.

## Milestone 11B — Relocatable Release Packaging, One-Command Installers & Homebrew Formula

1. **`tools/setup-mlx-native.sh`:** one-command reproducible `.reference/mlx-native` extractor on Apple Silicon from pinned `tools/reference/requirements-macos.lock`.
2. **`tools/package.sh`:** local reproducible packager producing versioned tarballs + `.sha256` checksums under `dist/` (gitignored):
   - **Linux (`ncm-<ver>-linux-<arch>-<cuda|cpu>.tar.gz`):** relocatable `bin/ncm` with `$ORIGIN/../lib:/usr/local/cuda/lib64:/opt/cuda/lib64` RPATH, `LICENSE`, `NOTICE`, `README-INSTALL.txt`. (Does not redistribute proprietary NVIDIA CUDA/cuDNN `.so` binaries.)
   - **macOS (`ncm-<ver>-macos-arm64-mlx.tar.gz`):** fully self-contained Apple Silicon bundle (`bin/ncm` with `@executable_path/../lib` RPATH, `lib/libmlx.dylib` with `@loader_path` RPATH, `lib/libjaccl.dylib`, `lib/mlx.metallib`, `LICENSE`, `NOTICE`, `README-INSTALL.txt`), ad-hoc signed, audited to contain **zero `/Users/` absolute paths**.
3. **`tools/install.sh`:** installs from a release tarball (`--archive`) or local build into `~/.local` (or `--prefix DIR`) and verifies with `ncm doctor`.
4. **`packaging/homebrew/ncm.rb` + `tools/render_homebrew_formula.py` (+ unit test in `tools/check.sh`):** deterministic Homebrew formula renderer for Apple Silicon tarballs, verified with `ruby -c` on macOS.
5. **Validation Gate (Linux/CUDA first, then Apple Silicon/Metal):**
   - Build package, verify `.sha256`, extract into a clean isolated directory outside the repo checkout, and run `bin/ncm --version`, `bin/ncm doctor --self-test --json`, and live GPU inference from outside the repo.
   - On macOS, verify the extracted tarball with **stripped environment (`PATH=/usr/bin:/bin`, unset `MLX_SYS_PREBUILT` and `DYLD_LIBRARY_PATH`)** and verify `otool -L`/`otool -l` show zero build-host paths.
   - Publish `docs/results/2026-11-24-phase11/README.md`, update `README.md`, `docs/QUICKSTART.md`, `docs/USAGE.md`, and draft `docs/RELEASE-NOTES-v0.7.0.md`.
