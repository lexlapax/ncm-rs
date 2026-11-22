# v0.2.0 release notes

Source-only release following `v0.1.0` (`de2975a`), delivering Phase 5 Steps 2–4 plus the five-milestone cross-phase gap-audit bundle verified on both Linux/CUDA (RTX 3060 Ti) and Apple Silicon/Metal (M1 Ultra).

## Highlights

- **Verified streaming model loading & pipelined SHA-256 (byte-identical, zero numerical changes):**
  - `Artifact::load_inference_streaming` (`ncm-io`) opens safetensors bundles with `O_NOFOLLOW`, validates `codec.safetensors`, `generator.safetensors`, and `training.safetensors` concurrently with parallel per-tensor SHA-256 + finite-FP32 verification and double-buffered pipelined whole-file SHA-256, then streams weights one tensor at a time into CUDA or MLX (`visit_bundle_tensors`).
  - **Linux / CUDA (RTX 3060 Ti, 3 alternating fresh processes):** model load sped up **2.05×–2.46×** across all 7 profiles (`lm1b-small` 1,150.9 → 467.8 ms, `wmt14` 1,392.4 → 598.2 ms, `lm1b-large` 4,306.1 → 2,102.7 ms); host peak HWM dropped up to **-2,833 MiB (0.22×)** on `lm1b-large` (3,637.6 → 804.5 MiB).
  - **Apple Silicon / Metal (M1 Ultra, 3 alternating fresh processes):** model load sped up **1.93×–3.63×** across all 7 profiles (`xsum` 1,039.7 → 341.1 ms, `wmt14` 2,012.2 → 561.5 ms, `lm1b-medium` 3,370.3 → 927.9 ms, `lm1b-large` 3,885.2 → 2,018.3 ms); peak process RSS dropped **28–49%** (`lm1b-large` 6,454.7 → 3,298.6 MiB, **-3,156 MiB**).
- **FP32 prefix K/V caching for XSum & WMT14 (zero corpus diffs):**
  - Added per-layer `PrefixKvCache` (`prefill_prefix_cache` / `decode_target_with_prefix_cache`) in `ncm-model`, enabled for XSum (all modes) and WMT14 (`cfg == 0.0`, preserving exact parity on positive CFG), and dropped before final vocabulary projection.
  - **XSum warm latency:** **3.24× faster on CUDA** (`155.75 → 48.09 ms` at b=1; `4.20×` at b=4) and **2.42× faster on Metal** (`204.61 → 84.64 ms` at b=1, while cutting MLX device peak by **-187 MiB**). Verified `0` diffs across 1,024 CUDA rows and 512 Metal rows (CFG=0, CFG=1, seeded=42, and CFG=1+seed=42).
  - **WMT14 warm latency:** **1.54× faster on CUDA** (`58.47 → 38.04 ms` at b=1; `2.23×` at b=4; `0` MiB VRAM increase) and **1.20× faster on Metal** (`93.67 → 78.35 ms` at b=1; `2.00×` at b=4; `-59 MiB` MLX device peak). Verified `0` diffs across all **3,003/3,003** WMT14 rows on CUDA and 256 rows on Metal.
- **Warm LM1B & penalty optimizations (byte-identical):**
  - Added `GeneratorBackend::sampled_embedding_8192` (eliminating 11× per-level CUDA host syncs), `TraceSink::wants_trace()` (skipping unused codec greedy argmax and penalty index uploads during inference), and precomputed FP64 penalty tables (`lm1b_penalized_ids` + `select_xsum_tokens_cpu`).
  - Reduced LM1B CUDA VRAM peak by **18–19 MiB** across small/medium/large and improved warm latency on both CUDA and Metal.
  - **Head-to-head vs Python MLX on Metal (`lm1b-{small,medium,large}`, batch 1):** Rust beats Python MLX on **warm median (1.06×–1.24×), warm p95 (1.06×–1.24×), peak RSS (0.51×–0.52×, saving up to 3.12 GiB), and MLX peak** across all three sizes, beats unverified Python MLX load on small (`1.21×`) and medium (`1.06×`), and stays within 2% on 3.3 GiB `lm1b-large` (`2,068` vs `2,026 ms`) despite full dual SHA-256 + finite-FP32 verification.
- **XSum Metal seeded + positive CFG parity:**
  - Removed the `Seeded XSum Metal generation requires CFG 0` limitation via `XsumTeacher::sample_seeded_cfg_cpu`, matching CUDA CLI flags.
- **Self-contained LM1B v2 artifacts (`lm1b-{small,medium,large}-inference-v2`):**
  - `tools/reference/lm1b_small_runtime_artifact.py --reference-rope` now exports optional self-contained `v2` artifacts bundling hash-pinned `rope_cos`/`rope_sin` inside `inference.safetensors` (`ncm sample` auto-detects embedded tables while keeping `v1` + sidecar fully supported). Verified 100% byte-identical to `v1` + sidecar in isolated directories on CUDA and Metal.
- **Generic config-driven local NCM model foundation (`local-ncm-v1`):**
  - Added `LocalNcmConfig` / `CodecConfig::validate_local` (`ncm-core`), `"local-ncm-v1"` verified artifact profile (`ncm-io`), offline HuggingFace `LocalTokenizer` (`ncm-tokenizers`), `LocalNcmModel` (`ncm-model`, supporting `untied_affine`, `untied_linear`, and `shared_head_biases` heads, optional prefix padding/EOS, prefix K/V caching, greedy & deterministic seeded sampling), unified `ncm sample` CLI integration, and stdlib exporter [tools/reference/export_local_ncm.py](../tools/reference/export_local_ncm.py).
- **Documentation & developer foundation:**
  - Added [Developer Guide (`docs/DEVELOPING.md`)](DEVELOPING.md) covering library embedding, backend traits, and running custom local NCM checkpoints.
  - Updated [README.md](../README.md), [Usage Guide (`docs/USAGE.md`)](USAGE.md), [Architecture (`docs/ARCHITECTURE.md`)](ARCHITECTURE.md), [Artifact Contract (`docs/ARTIFACTS.md`)](ARTIFACTS.md), and [Testing (`docs/TESTING.md`)](TESTING.md).
  - Published the [Step 2 stage-breakdown profile](results/2026-11-20-phase5-profile/README.md), [v0.2.0 dual-platform optimization report](results/2026-11-20-phase5-optimization/README.md), [Phase 5 final audit](results/2026-11-20-phase5-audit/phase5-final-audit-v1.md), and [cross-phase gap audit](results/2026-11-20-phase5-audit/cross-phase-gap-audit-v1.md).

## Verification

- **Byte-identical CLI equivalence:** all 9 profiles matched pre-Phase-5 baseline == alias == unified `ncm sample` on both CUDA ([cuda-v020-summary-v1.json](results/2026-11-20-phase5-optimization/cuda-v020-summary-v1.json)) and Metal ([mlx-v020-summary-v1.json](results/2026-11-20-phase5-optimization/mlx-v020-summary-v1.json)).
- **Local checks:** `./tools/check.sh` (0 `cargo check --workspace --locked` warnings), `./tools/check.sh --cuda` (`18` cases + `sampled_embedding_8192` + `local_ncm`), and `./tools/check.sh --mlx` (`18` cases + `ncm-cli --features mlx` clippy/tests) all passed.
- **End-to-end feature verification (CUDA & Metal):** isolated `lm1b-small-inference-v2` byte-identity, `local-ncm-v1` export + `ncm inspect` + `ncm sample --text --decode` (greedy & seeded), XSum Metal `--cfg 1 --seed 42 --decode`, and prefix-cache corpus equivalence.
- **Unchanged scope & limitations:** FP32 only; OWT unsupported (weights unreleased upstream); all accepted Phase 2–4 quality/numerical exceptions remain as documented in `v0.1.0` and the cross-phase gap audit. Source-only release (no checkpoints, datasets, or prebuilt binaries).
