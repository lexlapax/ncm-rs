# Release Notes — `v0.4.1`

Date: 2026-11-23 UTC
Theme: **Upstream `094f25ca` / `e34c3fab` Qualification & TinyStories 15k EMA Support**

`v0.4.1` qualifies `ncm-rs` against upstream `HazyResearch/ncm@094f25ca5fd8fe2bc7f0ff77bf3f860c3a34e7e8` and `hazyresearch/ncm-checkpoints@e34c3fab451ddbe7cf474cc97c44ab23321f2219` across both **Linux/CUDA** (RTX 3060 Ti) and **Apple Silicon/Metal** (M1 Ultra) while preserving full backward compatibility with legacy `c7027427` / `d342ce8a` artifacts.

---

## Highlights

### 1. TinyStories 15k EMA Release Support (`e34c3fab`)
- **Trimmed inference config compatibility:** `ncm-core` and `ncm-model` now admit upstream's inference-only `tinystories/config.yaml` (training-only fields default explicitly; strict Sudoku schema boundaries remain enforced).
- **Explicit dual pin-pair verification (`ncm-io`):** admits both the newly qualified (`094f25ca` / `e34c3fab`) and legacy (`c7027427` / `d342ce8a`) pin pairs without weakening manifest verification. Legacy `tinystories-phase4-v1` artifacts continue to load and generate unchanged.
- **FP16 safety guard:** the 15k EMA tokenizer retains extreme codebook weights (`max_abs = 4.37e10 > 65504.0`), so `--precision fp16` is explicitly rejected at load on both CUDA and Metal (`bf16`, `q8`, `q4`, and `fp32` all work cleanly).
- **Updated exporter & quickstart tooling:**
  - `tools/reference/tinystories_artifact.py --release 15k-ema`
  - `tools/reference/fetch_checkpoints.py --profile tinystories-15k-ema` (legacy `--profile tinystories` preserved for historical replay)
  - `ncm pull tinystories` / `tools/reference/quickstart_export.py --model tinystories` now defaults to the qualified 15k EMA release (`--model tinystories-legacy` retains `d342ce8a`).

### 2. Dual-Platform Parity & Performance (`tinystories-15k-ema-v1`, b=1)
- **Exact parity:** **0 / 2,564** codec reconstruct code diffs and **0 / 512** ID diffs across upstream Python `094f25ca`, Rust CUDA, and Rust Metal (**0 / 512** sample-decode ID diffs on CUDA).
- **Linux/CUDA (RTX 3060 Ti):** Rust `fp32` **41.14 ms / 492.9 MiB** and `bf16` **42.81 ms / 253.6 MiB** (`q8` 183.2 MiB, `q4` 138.7 MiB) vs upstream Python `094f25ca` eager `fp32` **43.22 ms / 644.9 MiB** and eager fast `bf16`+KV+resize **50.57 ms / 647.9 MiB**.
- **Apple Silicon/Metal (M1 Ultra):** Rust `fp32` **90.40 ms / 606.6 MiB MLX**, `bf16` **97.31 ms / 359.1 MiB MLX (93.8 MiB RSS)**, `q8` **98.33 ms / 271.2 MiB MLX**, `q4` **100.08 ms / 226.7 MiB MLX (94.2 MiB RSS)**.

### 3. Unchanged Profiles & Upstream Pins
- Sudoku (`easy`/`medium`/`hard`), XSum, WMT14, and LM1B (`small`/`medium`/`large`) checkpoints are byte-for-byte identical between `d342ce8a` and `e34c3fab`.
- OpenWebText weights remain unreleased (`owt/TODO.md`); `OwtConfigOnly` refusal remains active.
- [upstream.toml](../upstream.toml) updated to `094f25ca` / `e34c3fab` (`status = "qualified-dual-platform"`).

---

## Full Evidence & Reports

- [Upstream Qualification Report (`094f25ca` / `e34c3fab`)](results/2026-11-23-upstream-qualification/README.md)
- [Upstream Triage Report (2026-11-23)](results/2026-11-23-upstream-triage/README.md)
- [v0.4.0 Release Notes](RELEASE-NOTES-v0.4.0.md) · [Quickstart Guide](QUICKSTART.md) · [Usage Guide](USAGE.md)
