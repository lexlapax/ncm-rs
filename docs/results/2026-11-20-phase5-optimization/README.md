# Phase 5 Step 3: Model-Load SHA-256 & Streaming Weight-Loading Optimization (CUDA & Metal)

**Date:** 2026-11-21
**Selected target (user choice, 2026-11-20):** Both GPUs — parallelize `ncm-io` SHA-256 verification and stream verified weight bundles one tensor at a time into GPU/MLX storage (byte-identical, zero numerical changes).
**Commit:** `ee8769f` (baseline before optimization: `51046ab` / `v0.1.0` `de2975a`).

## Hypothesis

In the [Step 2 stage-breakdown profile](../2026-11-20-phase5-profile/README.md), single-threaded `Artifact::load` / `Artifact::load_inference` accounted for **80–96% of total model load time** across every profile on both GPUs (`4,668 ms` of `5,345 ms` on CUDA LM1B-large; `3,682 ms` of `3,987 ms` on Metal LM1B-large), because it read multi-GB safetensors bundles into process heap, then hashed the whole file and every tensor sequentially on one thread. On Apple Silicon unified memory, retaining the full safetensors buffers in host RAM during GPU weight upload also **doubled peak process RSS** (~2.0× weight size: `6,455 MiB` for `3.2 GiB` LM1B-large, `1,674 MiB` for `0.8 GiB` LM1B-small), and on Linux inflated host peak HWM to `3,637 MiB`.

**Changes (`ee8769f`):**
1. **`Artifact::load_inference_streaming`** (`crates/ncm-io/src/lib.rs`): opens `codec.safetensors`, `generator.safetensors`, and `training.safetensors` with `O_NOFOLLOW` and validates them concurrently in `std::thread::scope`. For bundles ≥ 4 MiB, one thread computes the whole-file SHA-256 while up to 6 worker threads verify per-tensor SHA-256 digests and finite FP32 values (skipping `attn_mask_for_masking`, whose `0.0`/`-inf` bits are checked explicitly in model preflight) in file-offset order using bounded 1 MiB scratch buffers (anonymous `mmap`/`munmap` on macOS). Small bundles (< 4 MiB) run in one sequential pass with zero thread-spawn overhead.
2. **One-tensor-at-a-time verified streaming upload** (`Artifact::visit_bundle_tensors`, `Artifact::read_bundle_tensor`, `Artifact::bundle_tensors`): `ncm-model` performs complete host preflight (manifest inventory, exact tensor names/dtypes/shapes, finite-value confirmation, `lvl_1L`, `attn_mask_for_masking`, `lm_heads.*.log_inv_T`, `inference.safetensors`) **before the first GPU allocation**, then streams each weight tensor sequentially into a single reusable `Scratch` buffer (immediately `munmap`ped on macOS on drop; no per-tensor zero-fill on Linux) and uploads directly to CUDA or MLX.
3. Wired `Artifact::load_inference_streaming` into `ncm-cli` (`inspect`, `generate`, `reconstruct`, `sample`, `sample-*`), `phase5_stage_profile` (CUDA & MLX), and `lm1b_small_mlx_timing_probe`. Existing `Artifact::load` and `Artifact::load_inference` remain intact (and also gain parallel per-tensor hashing).

## Byte-Identical Equivalence Gate (All 9 Profiles, Both GPUs)

`tools/bench/phase5_cli_equivalence.py` compared pre-Phase-5 baseline alias, Step 3 candidate alias, and Step 3 unified `ncm sample` in separate processes across all nine qualified artifacts. Every profile produced **byte-identical stdout** on both GPUs:

- **CUDA (`ee8769f`):** baseline `f10abaae...`, candidate `fdfd8d9407ae38437d863c20cb0f68b75f65ce07a245c33bce5353039de8fd4e` — 9/9 exact matches (`reports/phase5-step3-bench/cuda-equiv-v2/`).
- **Metal (`ee8769f`):** baseline `d8224daa...`, candidate `1e2a81e455979954ce70df85e481eea5d2667e6462f3fd2bc63e163210f52f7e` — 9/9 exact matches (`reports/phase5-step3-bench/mlx-equiv-v1/`).

## Before vs After: Linux / CUDA (RTX 3060 Ti)

Protocol: `tools/bench/run_phase5_step3_bench.py --platform cuda --processes 3` — three alternating fresh processes per side (`before`, `after`, repeated; 30 pooled warm trials per candidate; batch 1; `llama-server.service` stopped during measurements and restored afterwards).
Before binary: `2cf5ea59...` (`51046ab`). After binary: `6cb1fa97...` (`ee8769f`).
Summary JSON: [cuda-step3-summary-v1.json](cuda-step3-summary-v1.json) (SHA-256 `bfd5302fb45aa8d5b23d684104897e3f7a47fa376f8da1c04e9b2abb327783c9`).

| Profile | Load before -> after (ms) | Load speedup | Host HWM before -> after (MiB) | Host RAM saved | Warm median before -> after (ms) | CUDA pool peak (MiB, before / after) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sudoku-easy` | 207.8 -> 99.4 | **2.09×** | 607 -> 608 (1.00×) | -1 MiB (CUDA ctx floor) | 25.78 -> 25.79 (1.000×) | 168.8 / 168.8 |
| `tinystories` | 715.5 -> 307.9 | **2.32×** | 936 -> 808 (0.86×) | **-128 MiB** | 41.55 -> 41.42 (0.997×) | 492.9 / 492.9 |
| `xsum` | 817.9 -> 371.8 | **2.20×** | 1,076 -> 888 (0.82×) | **-188 MiB** | 154.92 -> 155.27 (1.002×) | 632.7 / 632.7 |
| `wmt14` | 1,391.5 -> 619.8 | **2.25×** | 1,518 -> 808 (0.53×) | **-710 MiB** | 58.56 -> 58.61 (1.001×) | 1,050.2 / 1,050.2 |
| `lm1b-small` | 1,152.5 -> 487.1 | **2.37×** | 1,242 -> 796 (0.64×) | **-445 MiB** | 42.79 -> 42.81 (1.000×) | 838.7 / 838.7 |
| `lm1b-medium` | 2,184.9 -> 1,124.0 | **1.94×** | 2,079 -> 797 (0.38×) | **-1,282 MiB** | 118.97 -> 119.13 (1.001×) | 1,675.9 / 1,675.9 |
| `lm1b-large` | 4,133.4 -> 2,415.1 | **1.71×** | 3,637 -> 800 (0.22×) | **-2,837 MiB** | 248.47 -> 248.57 (1.000×) | 3,233.4 / 3,233.4 |

## Before vs After: Apple Silicon / Metal (M1 Ultra, 64 GiB)

Protocol: `tools/bench/run_phase5_step3_bench.py --platform mlx --processes 3` — three alternating fresh processes per side (`before`, `after`, repeated; 30 pooled warm trials per candidate; batch 1; `dotsetup.rmlx` idle and untouched).
Before binary: `1dddcb8b...` (`51046ab`). After binary: `ee8769f`.
Summary JSON: [mlx-step3-summary-v1.json](mlx-step3-summary-v1.json) (SHA-256 `2131cb162600aaa8070ce804e72d20241b2e9cc31e65cb2c165273b3dd3f0494`).

| Profile | Load before -> after (ms) | Load speedup | Peak RSS/HWM before -> after (MiB) | Memory saved | Warm median before -> after (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sudoku-easy` | 188.3 -> 84.1 | **2.24×** | 358 -> 197 (0.55×) | **-162 MiB** | 54.14 -> 54.25 (1.002×) |
| `tinystories` | 672.0 -> 302.8 | **2.22×** | 1,016 -> 526 (0.52×) | **-490 MiB** | 93.59 -> 93.78 (1.002×) |
| `xsum` | 1,429.5 -> 348.9 | **4.10×** | 1,584 -> 1,016 (0.64×) | **-568 MiB** | 212.30 -> 211.29 (0.995×) |
| `wmt14` | 1,975.4 -> 583.6 | **3.38×** | 1,561 -> 1,122 (0.72×) | **-440 MiB** | 96.87 -> 96.69 (0.998×) |
| `lm1b-small` | 1,084.1 -> 402.2 | **2.70×** | 1,674 -> 878 (0.52×) | **-796 MiB** | 101.49 -> 101.51 (1.000×) |
| `lm1b-medium` | 4,571.2 -> 982.6 | **4.65×** | 3,328 -> 1,726 (0.52×) | **-1,602 MiB** | 208.76 -> 208.69 (1.000×) |
| `lm1b-large` | 3,863.8 -> 2,125.9 | **1.82×** | 6,455 -> 3,296 (0.51×) | **-3,158 MiB** | 384.34 -> 384.31 (1.000×) |

## Direct Head-to-Head vs Python MLX on Metal (LM1B Small / Medium / Large)

Protocol: `tools/bench/run_lm1b_small_mlx_timing.py --size {small,medium,large} --batch 1` on M1 Ultra — three alternating fresh processes per runtime, 5 warmups, 30 measured trials each (90 trials per runtime), `/usr/bin/time -l` maximum resident set size. Python MLX performs no SHA-256 verification or finite-value preflight; Rust performs full whole-file SHA-256, per-tensor SHA-256, and FP32 finite checks.

| Size | Model Load: Python MLX vs Rust | Peak RSS: Python MLX vs Rust | Warm Median: Python MLX vs Rust | All checks passed | Result JSON |
| --- | ---: | ---: | ---: | --- | --- |
| `lm1b-small` | 508.1 ms vs **435.9 ms (1.17× faster)** | 1,682 MiB vs **877 MiB (0.52×, -805 MiB)** | 116.78 ms vs **101.67 ms (1.15× faster)** | Yes | [mlx-lm1b-small-vs-pymlx-v1.json](mlx-lm1b-small-vs-pymlx-v1.json) (`8c276827...`) |
| `lm1b-medium` | 1,026.6 ms vs **1,018.1 ms (1.01× faster)** | 3,366 MiB vs **1,726 MiB (0.51×, -1,640 MiB)** | 224.63 ms vs **209.23 ms (1.07× faster)** | Yes | [mlx-lm1b-medium-vs-pymlx-v1.json](mlx-lm1b-medium-vs-pymlx-v1.json) (`fda2990a...`) |
| `lm1b-large` | 1,993.6 ms vs 2,158.7 ms (0.92×) | 6,494 MiB vs **3,296 MiB (0.51×, -3,198 MiB)** | 402.98 ms vs **385.13 ms (1.05× faster)** | Yes | [mlx-lm1b-large-vs-pymlx-v1.json](mlx-lm1b-large-vs-pymlx-v1.json) (`c60d7563...`) |

**Takeaway:** The narrow 0.15–1.2% Metal RSS margin noted in Phase 4 is now a **48–49% (2×) RSS reduction vs Python MLX** across all three LM1B sizes (saving 3.12 GiB on LM1B-large), while cutting load latency **1.71×–2.37× on CUDA** and **1.82×–4.65× on Metal** with zero warm-inference regression and byte-identical outputs across all 9 profiles.

## Checks & Limitations

- Local checks passed on both platforms at `ee8769f`: `setpriv --inh-caps=-all ./tools/check.sh`, `setpriv --inh-caps=-all ./tools/check.sh --cuda` (`18` cases), and `./tools/check.sh --mlx` (`18` cases + `ncm-cli --features mlx` clippy/tests).
- Streaming verification assumes a trusted, non-concurrently-mutated artifact directory (same trust model as `Artifact::load` and `Artifact::load_inference`), and opens weight bundles with `O_NOFOLLOW`.
- Warm-cache page-cache residency applies to repeated alternating processes; cold-disk reads were not separately isolated. Whole-file SHA-256 remains inherently single-threaded over the largest bundle (`generator.safetensors` = 2.9 GiB on LM1B-large), which bounds the remaining ~2.1–2.4 s load floor on LM1B-large.
