# v0.2.0 release notes — draft (awaiting tag approval)

Target release for [Phase 5 Steps 2–4](PHASE5.md) following the published `v0.1.0` release (`de2975a`).
This release adds verified streaming model loading with parallel SHA-256 verification (byte-identical outputs across all nine profiles on CUDA and Metal), publishes the dual-platform stage-breakdown profile and Phase 5 closeout audits, and refreshes both user-facing and developer-facing documentation. No `v0.2.0` tag or GitHub release has been created yet; tagging awaits user approval.

## Highlights

- **Faster verified model loading (byte-identical, zero numerical changes):**
  - `Artifact::load_inference_streaming` (`ncm-io`) opens safetensors bundles with `O_NOFOLLOW`, validates `codec.safetensors`, `generator.safetensors`, and `training.safetensors` concurrently with parallel per-tensor SHA-256 and finite FP32 checks using bounded 1 MiB scratch buffers, and streams weights one tensor at a time directly to CUDA or MLX (`visit_bundle_tensors`).
  - **Linux / CUDA (RTX 3060 Ti, 3 alternating fresh processes):** model load sped up **1.71×–2.37×** across all 7 profiles (`lm1b-small` 1,152.5 → 487.1 ms, `wmt14` 1,391.5 → 619.8 ms, `lm1b-large` 4,133.4 → 2,415.1 ms); host peak HWM dropped up to **-2,837 MiB (0.22×)** on `lm1b-large` (3,637 → 800 MiB). Warm inference latency (`0.997×–1.002×`) and CUDA pool peaks are unchanged.
  - **Apple Silicon / Metal (M1 Ultra, 3 alternating fresh processes):** model load sped up **1.82×–4.65×** (`xsum` 1,429.5 → 348.9 ms, `wmt14` 1,975.4 → 583.6 ms, `lm1b-small` 1,084.1 → 402.2 ms, `lm1b-medium` 4,571.2 → 982.6 ms, `lm1b-large` 3,863.8 → 2,125.9 ms); peak process RSS dropped **45–49%** (`lm1b-large` 6,455 → 3,296 MiB, **-3,158 MiB**).
  - **Head-to-head vs Python MLX on Metal (LM1B small/medium/large, batch 1):** Rust now uses **48–49% less peak RSS than Python MLX** (`0.51×–0.52×`) while beating unverified Python MLX load latency on `lm1b-small` (435.9 vs 508.1 ms) and `lm1b-medium` (1,018.1 vs 1,026.6 ms), and staying within 8% on 3.2 GiB `lm1b-large` despite full cryptographic whole-file + per-tensor SHA-256 and finite-FP32 verification.
- **Documentation & developer foundation:**
  - Added [Developer Guide (`docs/DEVELOPING.md`)](DEVELOPING.md) covering library embedding, backend traits, and adding new local NCM models/profiles.
  - Updated [Architecture (`docs/ARCHITECTURE.md`)](ARCHITECTURE.md), [Artifact Contract (`docs/ARTIFACTS.md`)](ARTIFACTS.md), [Testing (`docs/TESTING.md`)](TESTING.md), and [README.md](../README.md) to reflect the implemented Phase 5 system.
  - Published the [Step 2 stage-breakdown profile](results/2026-11-20-phase5-profile/README.md), [Step 3 optimization report](results/2026-11-20-phase5-optimization/README.md), [Phase 5 final audit](results/2026-11-20-phase5-audit/phase5-final-audit-v1.md), and [cross-phase gap audit](results/2026-11-20-phase5-audit/cross-phase-gap-audit-v1.md).

## Verification

- **Byte-identical CLI equivalence:** all 9 profiles matched pre-Phase-5 baseline == alias == unified `ncm sample` on both CUDA (`ee8769f`) and Metal (`ee8769f`).
- **Local checks:** `./tools/check.sh`, `./tools/check.sh --cuda` (`18` cases), and `./tools/check.sh --mlx` (`18` cases + `ncm-cli --features mlx` clippy/tests) all passed.
- **Unchanged scope & limitations:** FP32 only; OWT unsupported; all accepted Phase 2–4 quality/numerical exceptions remain as documented in `v0.1.0` and the cross-phase gap audit. Source-only release (no checkpoints, datasets, or prebuilt binaries).
