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
- Warm-cache page-cache residency applies to repeated alternating processes; cold-disk reads were not separately isolated. Whole-file SHA-256 remains inherently single-threaded over the largest bundle (`generator.safetensors` = 2.9 GiB on LM1B-large), which bounds the remaining ~2.0–2.1 s load floor on LM1B-large.

---

## Pre-`v0.2.0` Gap-Audit Optimization & Foundation Bundle (`4d970fa` / `e7b8cfa`, 2026-11-22)

Following the [cross-phase gap audit](../2026-11-20-phase5-audit/cross-phase-gap-audit-v1.md), five additional improvements were implemented and verified on both CUDA (RTX 3060 Ti) and Metal (M1 Ultra) before tagging `v0.2.0`:

1. **Warm-inference & I/O pipeline (Milestone A):** added `GeneratorBackend::sampled_embedding_8192` (removing 11× per-level CUDA host syncs in `Lm1bSmallTeacher`), `TraceSink::wants_trace()` (skipping unused codec greedy argmax & `upload_indices(&penalized)` during inference), precomputed FP64 penalty tables (`lm1b_penalized_ids` + `select_xsum_tokens_cpu`), pipelined whole-file SHA-256 I/O in `ncm-io` (`validate_streaming_bundle` overlapping `pread` and SHA-256 over two 2 MiB scratch buffers), and cleared `cargo check --workspace --locked` warnings.
2. **XSum Metal seeded + positive CFG (Milestone B):** added `XsumTeacher::sample_seeded_cfg_cpu` and enabled positive CFG + seeded sampling on Metal (`ncm sample ... mlx --cfg 1 --experimental-positive-cfg-xsum --seed 42 --experimental-stochastic-xsum`).
3. **Self-contained LM1B v2 artifacts (Milestone C):** admitted `lm1b-{small,medium,large}-inference-v2` bundling hash-pinned `rope_cos`/`rope_sin` inside `inference.safetensors` (`lm1b_small_runtime_artifact.py --reference-rope`), verified byte-identical to `v1` + sidecar in isolated directories on both CUDA (`cd87171d...`) and Metal (`3f28b815...`).
4. **FP32 prefix K/V caching for XSum & WMT14 (Milestone D):** added `PrefixKvCache` / `prefill_prefix_cache` / `decode_target_with_prefix_cache`, enabled on XSum (all modes) and WMT14 (`cfg == 0.0`, preserving exact parity on CFG > 0), dropping `prefix_cache` and `prefix` before final vocabulary projection.
   - **CUDA corpus equivalence (`prefix_cache_equiv_probe`):** WMT14 **3,003/3,003 rows** (`0` code diffs, `0` ID diffs; `b=1` 1.61×, `b=4` 2.23×); XSum **1,024 rows + CFG=1 + seed=42** (`0` diffs across all modes; `b=1` 3.40×, `b=4` 4.20×).
   - **Metal corpus equivalence (`prefix_cache_equiv_probe`):** WMT14 **256 rows** (`0` code/ID diffs; `b=1` 1.21×, `b=4` 2.00×); XSum **512 rows + CFG=1 + seed=42 + CFG=1&seed=42** (`0` diffs across all modes; `b=1` 2.50×, `b=4` 1.59×).
5. **Generic config-driven `local-ncm-v1` model foundation (Milestone E):** added `LocalNcmConfig` / `CodecConfig::validate_local`, `"local-ncm-v1"` profile in `ncm-io`, offline `LocalTokenizer`, `LocalNcmModel` (all 3 head kinds, prefix cache, greedy + seeded sampling), unified `ncm sample` support, and stdlib exporter `tools/reference/export_local_ncm.py`. Verified end-to-end on CPU, CUDA, and Metal.

### Cumulative `v0.2.0` vs Pre-Step-3 Baseline: Linux / CUDA (RTX 3060 Ti)

Protocol: `tools/bench/run_phase5_step3_bench.py --platform cuda --processes 3` (`llama-server.service` stopped; 3 alternating fresh processes, 30 warm trials per candidate; batch 1).
Summary JSON: [cuda-v020-summary-v1.json](cuda-v020-summary-v1.json) (SHA-256 `cd85f4a41cd8584b350dab0c04aa1705dd48b5bcdc3fdf58497d179a970ff86f`).
9/9 CLI byte-equivalence (`phase5_cli_equivalence.py --platform cuda`): **9/9 exact matches** against pre-Step-3 `baseline-ncm`.

| Profile | Load before -> `v0.2.0` (ms) | Load speedup | Host HWM before -> `v0.2.0` (MiB) | Warm median before -> `v0.2.0` (ms) | Warm speedup | CUDA pool peak (MiB, before / `v0.2.0`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sudoku-easy` | 203.2 -> 91.8 | **2.21×** | 605.1 -> 611.9 | 25.63 -> 25.65 | 1.00× | 168.8 / 168.8 |
| `tinystories` | 700.9 -> 295.7 | **2.37×** | 940.1 -> 815.3 (**-125**) | 41.21 -> 41.27 | 1.00× | 492.9 / 492.9 |
| `xsum` | 820.5 -> 356.1 | **2.30×** | 1,076.4 -> 895.5 (**-181**) | 155.75 -> **48.09** | **3.24×** | 632.7 / 686.1 |
| `wmt14` | 1,392.4 -> 598.2 | **2.33×** | 1,517.7 -> 818.4 (**-699**) | 58.47 -> **38.04** | **1.54×** | 1,050.2 / 1,050.2 |
| `lm1b-small` | 1,150.9 -> 467.8 | **2.46×** | 1,241.9 -> 803.2 (**-439**) | 42.95 -> **41.73** | **1.03×** | 838.7 / **820.2** |
| `lm1b-medium` | 2,185.7 -> 1,013.2 | **2.16×** | 2,080.8 -> 803.8 (**-1,277**) | 119.27 -> **118.45** | **1.01×** | 1,675.9 / **1,657.3** |
| `lm1b-large` | 4,306.1 -> 2,102.7 | **2.05×** | 3,637.6 -> 804.5 (**-2,833**, 0.22×) | 249.11 -> **247.98** | **1.005×** | 3,233.3 / **3,215.6** |

### Cumulative `v0.2.0` vs Pre-Step-3 Baseline: Apple Silicon / Metal (M1 Ultra)

Protocol: `tools/bench/run_phase5_step3_bench.py --platform mlx --processes 3` (`dotsetup.rmlx` idle/untouched; 3 alternating fresh processes, 30 warm trials per candidate; batch 1).
Summary JSON: [mlx-v020-summary-v1.json](mlx-v020-summary-v1.json) (SHA-256 `6619013f6a7a146579f815f7160d196ebebda733011f0abbe21bdb7781f36de2`).
9/9 CLI byte-equivalence (`phase5_cli_equivalence.py --platform mlx`): **9/9 exact matches** against pre-Step-3 `baseline-ncm`.

| Profile | Load before -> `v0.2.0` (ms) | Load speedup | Peak RSS before -> `v0.2.0` (MiB) | Warm median before -> `v0.2.0` (ms) | Warm speedup | MLX peak (MiB, before / `v0.2.0`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sudoku-easy` | 200.1 -> 92.8 | **2.16×** | 359.2 -> 197.2 (**-162**) | 53.46 -> 54.04 | 0.99× | 261.1 / 261.7 |
| `tinystories` | 670.4 -> 283.9 | **2.36×** | 1,015.9 -> 526.7 (**-489**) | 93.79 -> 93.68 | 1.00× | 606.9 / 606.6 |
| `xsum` | 1,039.7 -> 341.1 | **3.05×** | 1,584.7 -> 1,016.8 (**-568**) | 204.61 -> **84.64** | **2.42×** | 1,117.9 / **930.6** (-187) |
| `wmt14` | 2,012.2 -> 561.5 | **3.58×** | 1,561.7 -> 1,122.5 (**-439**) | 93.67 -> **78.35** | **1.20×** | 1,162.7 / **1,104.2** (-59) |
| `lm1b-small` | 1,088.5 -> 386.3 | **2.82×** | 1,674.4 -> 878.4 (**-796**) | 101.70 -> **96.26** | **1.06×** | 926.7 / 926.7 |
| `lm1b-medium` | 3,370.3 -> 927.9 | **3.63×** | 3,328.5 -> 1,726.7 (**-1,602**) | 210.02 -> **204.66** | **1.03×** | 1,806.4 / 1,806.4 |
| `lm1b-large` | 3,885.2 -> 2,018.3 | **1.93×** | 6,454.7 -> 3,298.6 (**-3,156**, 0.51×) | 385.41 -> **381.33** | **1.01×** | 3,414.5 / 3,413.8 |

### Updated Direct Head-to-Head vs Python MLX on Metal (`v0.2.0`, Batch 1)

| Size | Model Load: Python MLX vs Rust | Warm Median: Python MLX vs Rust | Warm p95: Python MLX vs Rust | Peak RSS: Python MLX vs Rust | MLX Peak: Python vs Rust | Result JSON |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `lm1b-small` | 516.0 vs **425.9 ms (1.21×)** | 120.19 vs **97.10 ms (1.24×)** | 122.42 vs **98.92 ms (1.24×)** | 1,683 vs **880 MiB (0.52×)** | 944 vs **927 MiB** | [mlx-vs-py-lm1b-small-v020.json](mlx-vs-py-lm1b-small-v020.json) (`c4e805e3...`) |
| `lm1b-medium` | 1,044.3 vs **983.1 ms (1.06×)** | 229.90 vs **204.56 ms (1.12×)** | 233.52 vs **208.12 ms (1.12×)** | 3,368 vs **1,729 MiB (0.51×)** | 1,832 vs **1,806 MiB** | [mlx-vs-py-lm1b-medium-v020.json](mlx-vs-py-lm1b-medium-v020.json) (`9175a46f...`) |
| `lm1b-large` | 2,025.5 vs 2,068.2 ms (0.98×) | 405.24 vs **382.60 ms (1.06×)** | 408.11 vs **386.44 ms (1.06×)** | 6,495 vs **3,298 MiB (0.51×)** | 3,448 vs **3,415 MiB** | [mlx-vs-py-lm1b-large-v020.json](mlx-vs-py-lm1b-large-v020.json) (`27257b4a...`) |
