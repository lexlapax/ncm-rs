# Upstream Triage Report: `HazyResearch/ncm` & `hazyresearch/ncm-checkpoints` (2026-11-23)

Date: 2026-11-23 UTC
Policy: [docs/UPSTREAM.md](../../UPSTREAM.md) (metadata & diff inspection only; no upstream code executed and no pins modified in [upstream.toml](../../../upstream.toml)).

## 1. Revision Comparison

| Source | Pinned ([upstream.toml](../../../upstream.toml)) | Observed (2026-11-23 UTC) | Status |
| --- | --- | --- | --- |
| Code (`HazyResearch/ncm` `main`) | `c70274273f489abc73832b2b98584a8483b6b203` | `094f25ca5fd8fe2bc7f0ff77bf3f860c3a34e7e8` (`2026-11-23T03:32:31Z`) | **Changed (+14 commits)** |
| Checkpoints (`hazyresearch/ncm-checkpoints`) | `d342ce8aaea58830eef3b8ad473664673e41f692` | `e34c3fab451ddbe7cf474cc97c44ab23321f2219` (`2026-11-23T06:21:02Z`) | **Changed (TinyStories updated)** |
| OpenWebText (`owt/`) | Unreleased (`owt/TODO.md`) | Unreleased (`owt/TODO.md` = `TODO`) | **Still unreleased** |

---

## 2. Classification of Upstream Code Changes (`c7027427...094f25ca`)

1. **Inference optimizations (behavior-preserving; parallel to `ncm-rs` v0.2.0–v0.4.0):**
   - `8b596811`: request-local prefix K/V caching across levels (`models/common.py`, `models/next_concept.py`). Already implemented in `ncm-rs` (`v0.2.0`).
   - `0db2e8de`: fixed-matrix `area`/`linear` resizing (`utils/fast_generation.py`), CUDA graph capture/replay, `BF16` autocast default (`utils/inference.py`), skipping nucleus sort when `top_p == 1.0`, and unsorted top-k cutoff (`models/next_concept.py`).
   - `d4962d2b`: fused Triton FP64 frequency/adjacency penalized argmax (`kernels/penalized_argmax.py`, `utils/token_selection.py`), preserving FP64 arithmetic and first-index tie-breaking (already matched by `ncm-rs` CUDA/Metal FP64 selection).
2. **Bug fixes:**
   - `7ca83e18`: fixes `z_hat.add_` in-place aliasing in `models/quant.py` (`z_hat = z_hat + z_q_level`) and switches `tokenization/vqvae_tokenizer.py` to `torch.load(..., weights_only=True)`. Both were already handled cleanly in `ncm-rs` since Phase 1/2.
   - `3623320d` / `094f25ca` (PR #4): avoids Python `OmegaConf` `getattr` exception-traceback GC leak in `generate()` (Python-specific; N/A to Rust).
   - `6d20afa8` / `7270808b` / `906cf923` (PR #22): fixes quadratic level schedule formula (`(n-1)**2`). Affects only non-manual quadratic schedules (`use_manual_levels: false`); all released checkpoints use `use_manual_levels: true`.
3. **Benchmarking, demo & docs:**
   - Added `benchmark.py`, `config/benchmark/*.yaml` (including `owt.yaml`), `generate.py`, `configs/generation/*.yaml`, and H100 `BF16`+compile+CUDA-graphs benchmark tables in `README.md`.

**Impact on historical outputs (Sudoku, XSum, WMT14, LM1B):** None expected for FP32 reference semantics. However, upstream Python's default inference speed baseline has improved substantially (`BF16` + `torch.compile` + CUDA graphs + prefix KV cache + Triton penalized argmax), so future Rust-vs-Python performance comparisons should benchmark against the new fast Python flags as well.

---

## 3. Classification of Upstream Checkpoint Changes (`d342ce8a...e34c3fab`)

Only 3 files changed out of 35 in `hazyresearch/ncm-checkpoints`:

| File | Change |
| --- | --- |
| `tinystories/config.yaml` | Modified (`2,483 → 2,242` bytes) — packaged inference-only config |
| `tinystories/generator.pt` | Modified (`356,160,651 → 356,150,923` bytes) — 15k checkpoint EMA release |
| `tinystories/tokenizer.pt` | Modified (`221,861,835 → 221,863,115` bytes) — 15k checkpoint EMA release |

All Sudoku (`easy`/`medium`/`hard`), XSum, WMT14, and LM1B (`small`/`medium`/`large`) checkpoints are **byte-for-byte unchanged**. OpenWebText remains unreleased (`owt/TODO.md`).

### Schema Impact on `ncm-rs` (TinyStories `v2` Candidate)

The new `tinystories/config.yaml` trims training-only fields:
- Removed from `tokenizer.vqvae`: `pre_quant_dropout`, `vector_quantizer_config.{decay, epsilon, commitment_cost, aux, ema_scale_by_world_size}`.
- Removed from `model.ctx_encoder`: `n_layer`, `vocab_size` (leaving only `enabled: false`).
- Changed `model.dropout` (`0.1 → 0.0`) and `tokenizer.vqvae.dropout` (`0.3 → 0.0`).

Because `ncm-core` enforces strict config deserialization for `tinystories-inference-v1`, supporting the updated 15k EMA TinyStories checkpoint cleanly requires:
1. Making those training-only fields optional (with explicit inference defaults) in `ncm-core` while preserving full backward compatibility with existing `tinystories-inference-v1` artifacts.
2. Qualifying the new 15k EMA TinyStories weights via a versioned candidate profile or lock update (and checking whether the old `2.48e10` FP16 codebook outlier is still present in the 15k EMA tokenizer).

---

## 4. Recommended Next Actions

1. **Candidate Upstream Qualification (optional next PR):**
   - Preserve existing `d342ce8a` artifacts intact.
   - Add support for the trimmed inference-only `tinystories/config.yaml` fields in `ncm-core`, qualify the new TinyStories 15k EMA checkpoint on Linux/CUDA and Apple Silicon/Metal, and run a 4-way benchmark (old/new Python with `BF16` + `compile` + `kv-cache` + CUDA graphs vs. Rust `FP32`/`BF16`/`FP16`/`Q8`).
2. **Keep `OwtConfigOnly` refusal active:** OpenWebText weights remain unreleased (`owt/TODO.md`).
