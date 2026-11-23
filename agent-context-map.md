# Agent context map

Last updated: 2026-11-23 UTC. Read first; load linked context lazily. Keep this file short:
current state, decisions, blockers and next actions. Put durable evidence in `docs/results/`.
The full pre-compaction history is archived in [docs/history/context-map-2026-11-18.md](docs/history/context-map-2026-11-18.md).

## Current state

- **Phase 7 (`v0.4.0` candidate, [docs/PHASE7.md](docs/PHASE7.md)) — COMPLETE on BOTH Linux/CUDA and Apple Silicon/Metal (2026-11-23):**
  - **Milestone 7A (`Q8`/`Q4` weight quantization) — COMPLETE on CUDA & Metal:**
    - `ComputePrecision::{Q8, Q4}` (`--precision q8|q4`) + `WeightBackend::upload_linear_weight_f32` (`ncm-core`/`ncm-model`), native `QTensor`/`QMatMul` (`Q8_0`/`Q4_0`) on CUDA and `mlx_rs::ops::quantize`+`quantized_matmul` (group-64 affine) on Metal; embeddings/norms/convs stay in BF16 (strict FP32 default preserved).
    - **Linux/CUDA ([cuda-quantization-v1.json](docs/results/2026-11-23-phase7/cuda-quantization-v1.json)):** `precision.rs` 4/4 passed; `lm1b-large` b=1 VRAM **3,215.6 → 1,025.1 MiB (`q8`, -68.1%, 199.0 ms / 1.24×) → 673.3 MiB (`q4`, -79.1%, 194.4 ms / 1.27×)**; `lm1b-large` b=16 fits in **975.4 MiB (`q4`) / 1,327.1 MiB (`q8`)**; `sudoku-easy` **128/128 `q8` / 126/128 `q4`**; `wmt14` `q8` **86.82%** (`q4` 57.7% disclosed); `tinystories` fluent in both `q8` & `q4`.
    - **Apple Silicon/Metal ([mlx-quantization-v1.json](docs/results/2026-11-23-phase7/mlx-quantization-v1.json)):** `precision.rs` 3/3 passed; `lm1b-large` b=1 MLX peak **3,414.4 → 1,131.4 MiB (`q8`, -66.9%, 544.2 MiB RSS) → 780.0 MiB (`q4`, -77.2%, 324.5 MiB RSS / -90.2%)**; `sudoku-easy` **128/128 `q8` / 127/128 `q4`**; `wmt14` `q8` **83.79%** (`q4` 57.1%); `tinystories` fluent in both `q8` & `q4`.
  - **Milestone 7B (Ollama-class CLI & Serving v2) — COMPLETE on CUDA & Metal:**
    - Added `LoadedModelInfo` + `GET /api/ps` (`crates/ncm-serve`), `ncm run MODEL [PROMPT]` (one-shot + hot-GPU REPL), `ncm pull (--list | MODEL)`, and dynamic micro-batching (`--max-batch`, default `8`) across all profiles.
    - **Linux/CUDA ([cuda-serve-v2-v1.json](docs/results/2026-11-23-phase7/cuda-serve-v2-v1.json)):** 8-concurrent-client micro-batching delivered **2.21× on `wmt14` FP16 (25.22 → 55.64 req/s, 3,560.8 tok/s, p50 306.4 → 141.1 ms)** and **1.64× on `lm1b-large` Q8 (4.95 → 8.10 req/s)**.
    - **Apple Silicon/Metal ([mlx-serve-v2-v1.json](docs/results/2026-11-23-phase7/mlx-serve-v2-v1.json)):** `ncm run wmt14 --precision q8` cold-to-translation **932.1 ms** (+80.8 ms 2nd REPL turn); 8-client micro-batching delivered **1.85× on `wmt14` FP16 (12.37 → 22.91 req/s, p50 639.5 → 237.2 ms)** and **1.97× on `lm1b-large` Q8 (2.31 → 4.55 req/s)**.
  - **Milestone 7C (Native Rust Fine-Tuning & Training) — COMPLETE on CUDA & Metal:**
    - Added `crates/ncm-train` (zero new external deps) + `ncm train` supporting RVQ/STE/EMA codec training, frozen-codec generator training & `--init-from` fine-tuning, Candle CPU/CUDA + MLX `value_and_grad` Metal autodiff, decoupled AdamW + grad clipping + warmup/cosine LR, atomic deterministic checkpoint save/resume, and direct verified `local-ncm-v1` export.
    - **Linux/CUDA ([cuda-train-v1.json](docs/results/2026-11-23-phase7/cuda-train-v1.json)):** `./tools/check.sh` (including `ncm-train/cpu`), CUDA Clippy, `ncm-train --features cuda` 7/7 (finite-diff gradient checks `< 2e-4`, EMA/dead-code, 5+5 vs 10-step checkpoint resume `< 1e-5`, CPU & CUDA roundtrips) passed. Live RTX 3060 Ti 4-level `[2,4,8,16]` training: codec **4.5654 → 0.0426 (8.76 ms/step)**, generator **3.4622 → 0.0011 (17.97 ms/step)** in **3.13s**, fine-tune **0.0011 → 0.0002**, `ncm sample` **8/8 rows (128/128 tokens = 100.0%)**, `ncm serve` exact match.
    - **Apple Silicon/Metal ([mlx-train-v1.json](docs/results/2026-11-23-phase7/mlx-train-v1.json)):** `./tools/check.sh --mlx` (18/18 Metal cases + `precision.rs` 3/3 + `ncm-train/cpu`), MLX Clippy (`-D warnings`), `ncm-train --features mlx` 4/4 passed. Live M1 Ultra 4-level `[2,4,8,16]` training: codec **4.5653 → 0.0425 (16.75 ms/step)** (matching CUDA to `0.0001`!), generator **3.4878 → 0.0010 (18.55 ms/step)** in **3.66s**, fine-tune **0.0010 → 0.0002**, `ncm sample` **8/8 rows (128/128 tokens = 100.0%)**, `ncm serve` exact match.
  - **Published docs & release readiness:** published [Phase 7 dual-platform report](docs/results/2026-11-23-phase7/README.md) and draft [docs/RELEASE-NOTES-v0.4.0.md](docs/RELEASE-NOTES-v0.4.0.md); refreshed `README.md`, `docs/QUICKSTART.md`, `docs/ARCHITECTURE.md`, `docs/PLAN.md`, and `docs/results/README.md`. **Pending user decision:** confirm bumping workspace to `0.4.0`, tagging `v0.4.0`, and publishing the GitHub release.

- **`v0.3.0` (Phase 6) released on Linux/CUDA and Apple Silicon/Metal (2026-11-22):**
  - Bumped workspace version to `0.3.0` (`Cargo.toml` + `Cargo.lock`), incorporated live newbie walkthrough UX polish (bind-before-preload with actionable port-in-use hint on `ncm serve`, dynamic column alignment on `ncm list`, primary alias priority over replay directories in `ModelRegistry`, `--plain` flag on `ncm sample`, trailing newline on HTTP JSON responses, and refreshed [docs/QUICKSTART.md](docs/QUICKSTART.md)), and finalized [docs/RELEASE-NOTES-v0.3.0.md](docs/RELEASE-NOTES-v0.3.0.md).
  - Plan in [docs/PHASE6.md](docs/PHASE6.md); full dual-platform results published in [docs/results/2026-11-22-phase6-precision-serving/README.md](docs/results/2026-11-22-phase6-precision-serving/README.md).
  - **Milestone 6A (`BF16`/`FP16` half precision) — COMPLETE on CUDA & Metal:**
    - `ComputePrecision::{Fp32, Bf16, Fp16}` + `WeightBackend::precision()`, `--precision` on `ncm sample`/`ncm serve`, strict FP32 default preserved, load-time FP16 range check (`> 65504.0`, cleanly rejecting TinyStories `vq.embeddings.0.weight` up to `2.48e10`), and eager MLX half-precision upload materialization.
    - **Linux/CUDA (RTX 3060 Ti, 8 GiB, [cuda-precision-v1.json](docs/results/2026-11-22-phase6-precision-serving/cuda-precision-v1.json)):** `lm1b-large` b=1 **246.2 → 167.7 ms (1.47× `fp16`)** & **3,215.6 → 1,667.7 MiB VRAM (-48.1%)**; `lm1b-large` **b=16 fits in 1,955.1 MiB VRAM** (`90.7 ms/seq` `fp16`); `sudoku-easy` **128/128 (100%)**; `xsum` `fp16` **29/32 exact texts (97.59% tokens)**; `wmt14` `fp16` **29/32 exact texts (98.93% tokens)**.
    - **Apple Silicon/Metal (M1 Ultra, [mlx-precision-v1.json](docs/results/2026-11-22-phase6-precision-serving/mlx-precision-v1.json)):** `./tools/check.sh --mlx` + `precision.rs` (2/2) passed; `lm1b-large` b=1 **3,414.4 → 1,798.4 MiB MLX peak (-47.3%)** & **3,297.8 → 323.4 MiB RSS (-90.2%)**; `lm1b-large` b=16 **2,113.3 → 1,838.3 ms (1.15× `fp16`)** & **-1.7 GiB MLX peak**; `sudoku-easy` **128/128 (100%)**; `wmt14` `fp16` **29/32 exact texts (97.71% tokens)**; `xsum` `fp16` **27/32 exact texts (92.51% tokens)**.
  - **Milestone 6B (Ollama & OpenAI-compatible serving + `ncm list` + `--prompt`) — COMPLETE on CUDA & Metal:**
    - Zero-new-dependency `crates/ncm-serve` (HTTP/1.1 + CORS + bounded 4 MiB body + SSE + NDJSON + `ModelRegistry`), `ncm list`, `ncm serve` (dedicated GPU worker thread + bounded LRU cache `--max-loaded`), and inline `--prompt TEXT` on `ncm sample`.
    - **Linux/CUDA ([cuda-serve-e2e-v1.json](docs/results/2026-11-22-phase6-precision-serving/cuda-serve-e2e-v1.json)):** warm XSum SSE **52.5 ms**, warm WMT14 Ollama chat **42.0 ms**, LRU switches (`xsum → wmt14 → lm1b-small`), 4× concurrent requests all `200 OK`, 404 unknown model.
    - **Apple Silicon/Metal ([mlx-serve-e2e-v1.json](docs/results/2026-11-22-phase6-precision-serving/mlx-serve-e2e-v1.json)):** `./tools/check.sh --mlx` + `ncm-cli --features mlx` clippy/tests passed; warm XSum SSE **101.4 ms**, warm WMT14 Ollama chat **87.9 ms**, LRU switches, 4× concurrent requests all `200 OK`.
  - **Milestone 6C (Newbie Quickstart & Docs) — COMPLETE:**
    - Added [docs/QUICKSTART.md](docs/QUICKSTART.md), `tools/reference/quickstart_export.py` (+ `test_quickstart_export.py` wired into `tools/check.sh`), [Phase 6 report](docs/results/2026-11-22-phase6-precision-serving/README.md), and draft [docs/RELEASE-NOTES-v0.3.0.md](docs/RELEASE-NOTES-v0.3.0.md); refreshed `README.md`, `docs/USAGE.md`, `docs/ARCHITECTURE.md`, `docs/DEVELOPING.md`, `docs/PLAN.md`, and `docs/results/README.md`. Verified 210 relative markdown links (0 broken) and full `./tools/check.sh`.
  - **Released:** annotated tag `v0.3.0` and source-only GitHub release published after dual-platform verification (`./tools/check.sh`, CUDA Clippy + release build, `./tools/check.sh --mlx`, and `ncm-cli --features mlx` Clippy/tests).

- **`v0.2.0` release ready & published on Linux/CUDA and Apple Silicon/Metal (2026-11-22):**
  - Bumped workspace version to `0.2.0` (`Cargo.toml` + `Cargo.lock`), finalized `docs/RELEASE-NOTES-v0.2.0.md` and `README.md`, tagging annotated `v0.2.0` and publishing source-only GitHub release with user approval.
  - All five cross-phase gap-audit items from [cross-phase-gap-audit-v1.md](docs/results/2026-11-20-phase5-audit/cross-phase-gap-audit-v1.md) implemented and dual-platform verified:
    - **Milestone A (warm-inference & I/O hygiene):** added `GeneratorBackend::sampled_embedding_8192` (removing 11× per-level CUDA host syncs in `Lm1bSmallTeacher`), `TraceSink::wants_trace()` (skipping unused codec greedy argmax and `upload_indices(&penalized)` during inference), precomputed FP64 penalty tables (`lm1b_penalized_ids` + `select_xsum_tokens_cpu`), pipelined whole-file SHA-256 I/O in `ncm-io` (`validate_streaming_bundle`), and cleared `cargo check --workspace --locked` warnings.
    - **Milestone B (XSum Metal seeded + positive CFG):** added `XsumTeacher::sample_seeded_cfg_cpu` and wired `sample_xsum_mlx` so seeded XSum Metal supports positive CFG scales (matching CUDA CLI flags; verified on M1 Ultra `prefix_cache_equiv_probe` `cfg1_seed42_id_diffs=0` + CLI `--cfg 1 --seed 42 --decode` deterministic SHA-256 `dee14e21...`).
    - **Milestone C (`lm1b-*-inference-v2` self-contained artifacts):** admitted `lm1b-{small,medium,large}-inference-v2` bundling hash-pinned `rope_cos`/`rope_sin` in `inference.safetensors` (`lm1b_small_runtime_artifact.py --reference-rope`), auto-routed by `ncm sample` while keeping `v1` + sidecar unchanged. Verified isolated `lm1b-small-inference-v2` (no sidecar) is **100% byte-identical** to `v1` + sidecar on both CUDA (`cd87171d...`) and Metal (`3f28b815...`).
    - **Milestone D (XSum & WMT14 prefix K/V caching):** added `PrefixKvCache` / `prefill_prefix_cache` / `decode_target_with_prefix_cache` in `ncm-model`, enabled for XSum (all modes) and WMT14 (`cfg == 0.0`, preserving exact parity on CFG > 0) and dropped before final vocab projection. Verified on RTX 3060 Ti (WMT14 **3,003/3,003 rows** `0` diffs, `b=1` 1.61× / `b=4` 2.23×; XSum **1,024 rows + CFG=1 + seed=42** `0` diffs, `b=1` 3.40× / `b=4` 4.20×) and M1 Ultra (WMT14 256 rows `0` diffs, `b=1` 1.21× / `b=4` 2.00×; XSum 512 rows + CFG=1 + seed=42 `0` diffs, `b=1` 2.50× / `b=4` 1.59×).
    - **Milestone E (`local-ncm-v1` config-driven local model foundation):** added `CodecConfig::validate_local`, `LocalNcmConfig`/`LocalGeneratorConfig`/`LocalHeadKind`, `"local-ncm-v1"` profile in `ncm-io`, offline `LocalTokenizer` in `ncm-tokenizers`, `LocalNcmModel` in `ncm-model` (with prefix cache, greedy + seeded sampling, all 3 head kinds), unified `ncm sample` support, and stdlib exporter `tools/reference/export_local_ncm.py` (+ `test_export_local_ncm.py`). Verified on CPU (`tests/local_ncm.rs`), CUDA, and Metal (`ncm inspect`, `--text --decode`, greedy, and deterministic `--seed 42`).
  - **Linux/CUDA full gate passed (2026-11-22, `llama-server.service` stopped and left off per user):**
    - `./tools/check.sh` and `./tools/check.sh --cuda` (18/18 CUDA operator cases, `sampled_embedding_8192`, `local_ncm`, `test_export_local_ncm.py`).
    - **9/9 CLI byte-equivalence** (`phase5_cli_equivalence.py --platform cuda`) matched pre-Step-3 `baseline-ncm` identically on all 9 profiles.
    - **3-alternating-process 7-profile CUDA benchmark** ([cuda-v020-summary-v1.json](docs/results/2026-11-20-phase5-optimization/cuda-v020-summary-v1.json), SHA-256 `cd85f4a4...`): load **2.05×–2.46× faster** (`lm1b-large` 4,306→2,103 ms), HWM **-125 to -2,833 MiB** (`lm1b-large` 3,638→804 MiB, 0.22×), warm latency **3.24× faster on XSum** (155.8→48.1 ms), **1.54× faster on WMT14** (58.5→38.0 ms, 0 MiB VRAM increase), and **faster + 18–19 MiB lower VRAM peak on LM1B small/medium/large**.
  - **Apple Silicon/Metal full gate passed on `macstudio` (2026-11-22, M1 Ultra, `dotsetup.rmlx` idle/untouched):**
    - `./tools/check.sh --mlx` (18/18 Metal cases) + `ncm-cli --features mlx` Clippy (`-D warnings`) & unit tests.
    - **9/9 CLI byte-equivalence** (`phase5_cli_equivalence.py --platform mlx`) matched pre-Step-3 `baseline-ncm` identically on all 9 profiles.
    - **3-alternating-process 7-profile Metal benchmark** ([mlx-v020-summary-v1.json](docs/results/2026-11-20-phase5-optimization/mlx-v020-summary-v1.json), SHA-256 `6619013f...`): load **1.93×–3.63× faster**, peak RSS **-28% to -49%** (`lm1b-large` 6,455→3,299 MiB, 0.51×), warm latency **2.42× faster on XSum** (204.6→84.6 ms, `-187 MiB` MLX peak), **1.20× faster on WMT14** (`-59 MiB` MLX peak), and **faster on LM1B small/medium/large**.
    - **Direct Rust vs Python MLX head-to-head** (`run_lm1b_small_mlx_timing.py`, batch 1): all checks passed for `lm1b-{small,medium,large}` (`c4e805e3...`, `9175a46f...`, `27257b4a...`) — Rust beats Python MLX on **warm median (1.06×–1.24×), warm p95 (1.06×–1.24×), peak RSS (0.51×–0.52×), and MLX peak** across all three sizes, plus faster load on small (`1.21×`) & medium (`1.06×`) and `0.98×` on 3.3 GiB large (despite full dual SHA-256 + finite-FP32 verification).
  - **Published docs & release:** updated [v0.2.0 optimization report](docs/results/2026-11-20-phase5-optimization/README.md), [cross-phase gap audit](docs/results/2026-11-20-phase5-audit/cross-phase-gap-audit-v1.md), `README.md`, `docs/USAGE.md`, `docs/ARTIFACTS.md`, `docs/ARCHITECTURE.md`, `docs/DEVELOPING.md`, and `docs/RELEASE-NOTES-v0.2.0.md`. Annotated tag `v0.2.0` and source-only GitHub release published on this commit.

- **Phase 5 complete on Linux/CUDA and Apple Silicon/Metal (2026-11-21):**
  - **Step 1 (`v0.1.0`):** published annotated tag `v0.1.0` at `de2975a` and source-only GitHub release (`https://github.com/lexlapax/ncm-rs/releases/tag/v0.1.0`).
  - **Step 2:** dual-platform stage-breakdown profile published ([docs/results/2026-11-20-phase5-profile/README.md](docs/results/2026-11-20-phase5-profile/README.md)).
  - **Step 3 (`ee8769f`):** implemented user-selected target (parallel SHA-256 + streaming verified weight loading via `Artifact::load_inference_streaming` / `visit_bundle_tensors` / `read_bundle_tensor`).
    - **CUDA (RTX 3060 Ti):** `./tools/check.sh --cuda`, 9/9 CLI byte-equivalence (`fdfd8d94...`), 3-process alternating benchmark ([cuda-step3-summary-v1.json](docs/results/2026-11-20-phase5-optimization/cuda-step3-summary-v1.json), SHA-256 `bfd5302f...`): **1.71×–2.37× faster model load**, up to **-2,837 MiB (0.22×) host HWM** on `lm1b-large`, warm latency `0.997×–1.002×`, CUDA pool peaks unchanged. `llama-server.service` restored (`active`).
    - **Metal (M1 Ultra, `macstudio`):** `./tools/check.sh --mlx`, `ncm-cli --features mlx` clippy/tests, 9/9 CLI byte-equivalence (`1e2a81e4...`), 3-process alternating benchmark ([mlx-step3-summary-v1.json](docs/results/2026-11-20-phase5-optimization/mlx-step3-summary-v1.json), SHA-256 `2131cb16...`): **1.82×–4.65× faster model load**, **-45% to -49% peak RSS** (`lm1b-large` 6,455→3,296 MiB), warm latency `0.995×–1.002×`. Direct Rust vs Python MLX head-to-head (`run_lm1b_small_mlx_timing.py`, batch 1) passes all checks for `lm1b-small/medium/large` with **0.51×–0.52× peak RSS** and faster load on small (`1.17×`) and medium (`1.01×`).
  - **Step 4:** published [Step 3 optimization report](docs/results/2026-11-20-phase5-optimization/README.md), [Phase 5 final audit](docs/results/2026-11-20-phase5-audit/phase5-final-audit-v1.md), [cross-phase gap audit](docs/results/2026-11-20-phase5-audit/cross-phase-gap-audit-v1.md), new [Developer Guide (`docs/DEVELOPING.md`)](docs/DEVELOPING.md), refreshed `README.md`, `docs/ARCHITECTURE.md`, `docs/ARTIFACTS.md`, `docs/TESTING.md`, `docs/PLAN.md`, `docs/results/README.md`, updated `docs/RELEASE-NOTES-v0.1.0.md`, and drafted `docs/RELEASE-NOTES-v0.2.0.md`.
  - **Pending user decision:** confirm whether to tag and publish `v0.2.0` now (source-only) or first address any items from [cross-phase-gap-audit-v1.md](docs/results/2026-11-20-phase5-audit/cross-phase-gap-audit-v1.md) (e.g., CUDA/Metal warm LM1B wins, generic local-model config loader, or LM1B-large verified load speed).

- **Phase 5 Step 2 complete on CUDA and Metal (2026-11-20):** Metal profiling passed on `macstudio` (`51046ab`, all 7 profiles × 2 batch sizes; [mlx-profile-summary-v1.json](docs/results/2026-11-20-phase5-profile/mlx-profile-summary-v1.json), SHA-256 `0514461ed7ecb61dcb0ad600c62d3b0dce642d15b8873628d21dfd1cb8f4c10a`). Published the combined dual-platform stage breakdown and ranked bottleneck table in [docs/results/2026-11-20-phase5-profile/README.md](docs/results/2026-11-20-phase5-profile/README.md).

- **Phase 5 Step 2 CUDA profiling complete (2026-11-20):** added zero-overhead `stage_tensor`/`stage_indices` hooks to `TraceSink`, `phase5_stage_profile` examples (`ncm-backend-cuda` and `ncm-backend-mlx`), and `tools/bench/run_phase5_profile.py`. Profiled all 7 profiles at batch 1 and largest measured batch on RTX 3060 Ti (`llama-server.service` stopped; [cuda-profile-summary-v1.json](docs/results/2026-11-20-phase5-profile/cuda-profile-summary-v1.json), SHA-256 `7941fd98ade8f37199862796d4660fc71a06b65969867d87c784be4df15996bc`). Key CUDA findings: `Artifact::load_inference` dominates model load at 80–88% across all profiles (`4,668 ms` of `5,345 ms` on LM1B-large; `1,296 ms` of `1,469 ms` on LM1B-small); warm inference is dominated by generator levels (`80–97%`), with LM1B-small b=16 showing `42.9 ms` (`11.4%`) in sampling+penalties and `32.4 ms` (`8.7%`) in codec decode. Rust fmt/clippy/CPU tests and CUDA example clippy passed; running Metal Step 2 profiling on `macstudio` next.

- **Phase 5 Step 1 complete on CUDA and Metal (2026-11-20):** fresh-clone macOS guide replay at `88d2956` passed on `macstudio` (extracted `docs/USAGE.md` blocks 3 and 17: `uv` + `requirements-macos.lock`, `.reference/mlx-native`, `ncm-cli --features mlx` SHA-256 `bd18bce000d843102b828e4f6ca7873f0cbfb78b5ffaec82cfbebd7ff3956723`, `./tools/check.sh --mlx` 18 cases, 6 manifest/sidecar SHA-256 checks, synthetic Sudoku + 4 text decoded examples; temp clone deleted; [report](docs/results/2026-11-19-phase5-cli/macos-guide-fresh-clone-v1.md)). Updated `docs/USAGE.md`, `README.md`, `docs/RELEASE-NOTES-v0.1.0.md`, and `docs/results/README.md`. Step 1 commit is ready for `v0.1.0` tagging upon user confirmation; proceeding to **Step 2** stage-breakdown profiling (CUDA first, then Metal).

- **Metal Step 1 CLI & text/decode checks passed (2026-11-20):** on `macstudio` (`a18a624`, M1 Ultra, `dotsetup.rmlx` idle/untouched), `./tools/check.sh --mlx` passed (493 macOS report tests, MLX clippy/test/doc, `mlx_probe` 18 cases). `phase5_cli_equivalence.py --platform mlx` matched `702168c` baseline == candidate alias == unified `ncm sample` byte-for-byte across all 9 profiles (`--features mlx` only); `phase5_cli_invalid_devices.py` rejected all 9 profiles. XSum/WMT14 experimental flags, `mlx-retry-padded`, LM1B sibling rotary sidecar and `--top-p-fp32`, all 6 macOS corpus/boundary encode+decode tests (11,334 XSum, 3,003 WMT14, 8,199/8,192 TinyStories), and Metal `--text`/`--decode` end-to-end equivalence passed ([report](docs/results/2026-11-19-phase5-cli/mlx-step1-cli-v1.md)). Updated `docs/USAGE.md` with complete Apple Silicon / MLX setup and examples (`17` shell blocks pass `bash -n`); fresh-clone Mac replay next.

- **macOS report-test portability fix (2026-11-20):** initial `macstudio` run at `fc52a57` built `ncm-cli --features mlx` cleanly (clippy `-D warnings` and `ncm-cli` tests passed), then failed two macOS report tests: `test_phase5_handoff_image_probe.py` (Linux `/proc/self/exe` stat) and `test_phase5_cli_cold_screen.py` (`/var` vs `/private/var` resolved path). Skipped the `/proc` test on non-Linux and compared resolved paths; re-running `./tools/check.sh --mlx` on `macstudio` next.

- **Post-reboot WMT14 model timing published (2026-11-20):** with llama-server stopped but three desktop GPU clients present (`533` samples), the `4f804fa` pool-instrumented WMT14 CFG=0 rerun (`reports/phase5-wmt14-post-reboot-perf-v1`, batches 1/4/16/32) passed independent output/stats/pool replay and was published in [wmt14-cuda-perf-summary-v2.json](docs/results/2026-11-19-phase5-cli/wmt14-cuda-perf-summary-v2.json) (SHA-256 `848bc5bdd3799815c2a22e10f1edfacaa49abaea0c674dd8681d51ecbaf075b3`) and [cuda-wmt14-performance-retest-v1.md](docs/results/2026-11-19-phase5-cli/cuda-wmt14-performance-retest-v1.md#post-reboot-rerun-with-rust-cuda-default-pool-counters). Rust/Python medians: 57.855/64.324, 165.064/178.549, 630.848/651.295, 1246.266/1323.332 ms. Default-pool high-water is not total native peak; Linux Step 1 evidence is complete for advancing to Metal Step 1 on `macstudio`.

- **GPU window (2026-11-20):** user authorized stopping llama-server; stopped `llama-server.service` (previously active, enabled; not disabled). Desktop GPU clients remain, so device exclusivity is not established; five `nvidia-smi dmon` samples showed 0% SM utilization. Do not stop other applications without approval. Restore the service when the measurement window ends.

- **Post-reboot CUDA corpus rerun (2026-11-20):** loaded module and NVML both 620.73.04; CUDA initialization succeeded. With user approval on shared GPU, the interrupted WMT14 CFG=1 direct text/decode command completed 3,003 rows / 751 batches against its pinned references. Summary SHA-256 `142773995eebf6c1c5927856d28541494635d62f7ec8f350942b300949c828ef`, ignored capture `reports/phase5-wmt14-cfg1-text-decode-reboot-v1/`; earlier partial capture retained. Independent replay matched inputs, codes, IDs, decoded strings and summary hashes for all batches; `setpriv --inh-caps=-all tools/check.sh` passed. Nonblind correctness only; exclusive performance and Metal remain pending. See [command and interrupted attempt](docs/results/2026-11-19-phase5-cli/cuda-decode-screen-v1.md#wmt14-cfg1-direct-textdecode-cli-attempt--interrupted).

- **Pre-reboot interruption (2026-11-20):** user confirmed an NVIDIA update was in progress and needed to reboot. The [direct WMT14 CFG=1 text+decode corpus attempt](docs/results/2026-11-19-phase5-cli/cuda-decode-screen-v1.md#wmt14-cfg1-direct-textdecode-cli-attempt--interrupted) stopped after 28 matching rows with `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`; loaded module was 620.70.06 and NVML 620.73.04. No machine change was made before reboot. Six runner tests and `setpriv --inh-caps=-all tools/check.sh` passed before reboot. The later complete rerun is recorded above.

- 2026-11-20: [WMT14 CFG=1 retained CUDA corpus decoding](docs/results/2026-11-19-phase5-cli/cuda-decode-screen-v1.md#wmt14-cfg1-retained-cuda-corpus-decode) matched pinned evaluator strings on 3,003 rows; CFG=0 regression, altered-reference negative control and `setpriv --inh-caps=-all tools/check.sh` passed. Host-side test only; no direct CLI corpus decode, new GPU inference, performance or Metal claim.

- 2026-11-20: [WMT14 CFG=1 CUDA text corpus](docs/results/2026-11-19-phase5-cli/cuda-text-input-v1.md#wmt14-cfg1-full-corpus-text-generation) matched the prior ID-input stdout on all 3,003 rows in 751 batches; pinned text IDs equaled frozen prefixes. Independent replay, five runner tests and `setpriv --inh-caps=-all tools/check.sh` passed. Nonblind shared-GPU correctness only, no decode/Metal/performance claim; raw texts and outputs local.

- 2026-11-20: [CPU handoff executable-identity investigation](docs/results/2026-11-19-phase5-cli/linux-lifecycle-env-investigation-v1.md#handoff-executable-identity-investigation) instrumented only fake workers, preserving fail-closed errors and recording fixed-identity booleans. Fifty handoffs did not reproduce the intermittent failure; cause remains unproven. Probe tests and `setpriv --inh-caps=-all tools/check.sh` passed. No production observer change; exclusive CUDA window requested from user.

- 2026-11-20: published [sanitized LM1B CUDA model timing trials](docs/results/2026-11-19-phase5-cli/lm1b-cuda-perf-trials-v1.json): 1,620 values across all nine measured cells, bound to five result/manifest hashes previously pinned in the aggregate, without inputs/outputs/paths/PIDs. Independent median/p95 and exporter replays, negative tests and `setpriv --inh-caps=-all tools/check.sh` passed. Shared-GPU model-only evidence, not release or Metal qualification; raw bundles remain local.

- 2026-11-20: published sanitized [XSum](docs/results/2026-11-19-phase5-cli/xsum-cuda-perf-trials-v1.json) (540) and [WMT14](docs/results/2026-11-19-phase5-cli/wmt14-cuda-perf-trials-v1.json) (720) shared-GPU model timings, no inputs/outputs/paths. Export replay and independent median/p95 checks passed; WMT14 values also match previously pinned measurements, while XSum raw per-file hashes were not precommitted. Negative tests and `setpriv --inh-caps=-all tools/check.sh` passed. Not release or Metal qualification; raw bundles remain local.

- 2026-11-20: published [sanitized TinyStories model timing trials](docs/results/2026-11-19-phase5-cli/tinystories-cuda-perf-trials-v1.json): 720 shared-GPU measured values without inputs/outputs/paths. [Report](docs/results/2026-11-19-phase5-cli/cuda-tinystories-performance-retest-v1.md#sanitized-per-trial-publication) discloses that original per-file timing hashes were not precommitted; published medians/p95 replay, exporter replay, negative tests and `setpriv --inh-caps=-all tools/check.sh` passed. Not release qualification; raw bundles remain local.

- 2026-11-20: [WMT14 CUDA default-pool diagnostic](docs/results/2026-11-19-phase5-cli/cuda-wmt14-pool-diagnostic-v1.md) adds scoped native pool high-water counters to the model timing example and collector/summarizer. One real-GPU batch-1 output matched the prior example; counter invariants, focused tests, old-summary byte replay, CUDA-feature Clippy and `setpriv --inh-caps=-all tools/check.sh` passed. A subsequent one-repetition shared-GPU collector smoke retained distinct Python/Rust counters and matched outputs; independent replay passed. A full check then failed a CPU handoff fixture with `ObservationMismatch` / unexpected live executable identity; focused and full retries passed, cause unproven. Full performance protocol, CLI/total-device peaks, exclusive GPU and Metal gates remain pending; no production inference change.

- 2026-11-20: [post-WMT14-boundary nine-profile CUDA routing rerun](docs/results/2026-11-19-phase5-cli/cuda-current-routing-v1.md#post-wmt14-boundary-nine-profile-rerun) matched pre-change, alias and unified stdout for each fixed request, with previous output hashes reproduced. Independent replay, five focused runner tests and `setpriv --inh-caps=-all tools/check.sh` passed. No Metal, quality or performance gate claim; captures local.

- 2026-11-20: [updated step 1 progress audit](docs/results/2026-11-19-phase5-cli/step1-progress-audit-v2.md) maps current scoped evidence and remaining CUDA, Metal, release, profiling and optimization gates; not a completion audit. Thirty relative file links and `setpriv --inh-caps=-all tools/check.sh` passed; no runtime change.

- 2026-11-20: [WMT14 CFG=1 CUDA alias corpus comparison](docs/results/2026-11-19-phase5-cli/cuda-wmt14-cfg1-quality-retest-v1.md#legacy-and-pre-change-aliases) matched the frozen Python capture on 3,003 rows; independent replay found byte-identical input/stdout across pre-change alias, current alias and prior unified CLI for all 751 batches. Extended runner tests and `setpriv --inh-caps=-all tools/check.sh` passed. Nonblind shared-GPU correctness evidence, not performance/Metal; raw captures local.

- 2026-11-20: on `06b6721`, `setpriv --inh-caps=-all tools/check.sh --cuda` passed, including 18 real-device operator cases; `reports/local-check-20261120T172804Z-4158547/cuda.json` is local/ignored. This is a CUDA smoke check, not a model-quality, performance or Metal gate. GPU was shared at inspection; release-quality exclusive-device measurement remains pending.

- 2026-11-20: [WMT14 CFG=1 unified CUDA CLI corpus](docs/results/2026-11-19-phase5-cli/cuda-wmt14-cfg1-quality-retest-v1.md#unified-cuda-cli-corpus-equivalence) matched the frozen Python capture on all 3,003 rows and 751 batches; independent replay, runner negative tests and `setpriv --inh-caps=-all tools/check.sh` passed. Nonblind ID-input retest on shared GPU; subsequent legacy full-corpus comparison is above, while performance and Metal remain pending; raw captures local.

- 2026-11-20: [WMT14 CFG=1 CUDA nonblind quality retest](docs/results/2026-11-19-phase5-cli/cuda-wmt14-cfg1-quality-retest-v1.md) regenerated the exact pinned Python capture, matched all 3,003 rows' 939,939 codes and 192,192 IDs with the fast backend, and reproduced passing full/held-out receipts in both scorer environments. The subsequent unified-CLI corpus comparison is recorded above; performance and Metal remain pending, and raw captures are local only. `setpriv --inh-caps=-all tools/check.sh` and report shell syntax passed; no runtime change.

- 2026-11-20: user chose WMT14 CFG option 1: 0 and 1 unflagged; other positive scales require `--experimental-cfg-wmt14` on unified and legacy routes. Help/migration docs and host rejection tests updated. [CUDA fixed-request screen](docs/results/2026-11-19-phase5-cli/cuda-wmt14-cfg-boundary-v1.md) matched the pre-change alias for scales 0/1/2 across both routes; unflagged 2 rejected. `setpriv --inh-caps=-all tools/check.sh` passed. Not full quality/performance or Metal qualification; Linux release gate remains open.

- 2026-11-20: extended unified-CLI rejection coverage for malformed, missing-profile and over-4-MiB dispatch manifests. Focused `setpriv --inh-caps=-all cargo test -p ncm-cli --test unified_rejections --locked` and full `setpriv --inh-caps=-all tools/check.sh` passed. Test-only, no inference or GPU qualification; the subsequent WMT14 CFG decision is recorded above, and release gates remain pending.

- 2026-11-20: refreshed `README.md`, `docs/USAGE.md` and draft release notes to reflect implemented empty-only LM1B text, the `c63b205` Linux guide replay and the then-pending WMT14 CFG decision/Mac gate. Relative file links (80), shell-block syntax and `setpriv --inh-caps=-all tools/check.sh` passed. No local or remote `v0.1.0` tag was observed; GitHub release lookup returned not found. Documentation only; no runtime or GPU validation.

- 2026-11-20: [WMT14 CFG release-surface audit](docs/results/2026-11-19-phase5-cli/wmt14-cfg-policy-audit-v1.md) found arbitrary positive scales enter both CLI routes without an experimental flag, while qualification is scoped to tested CFG settings. The subsequent decision and implementation are recorded above. This audit itself made no runtime change or GPU inference; `setpriv --inh-caps=-all tools/check.sh` passed.

- 2026-11-20: [LM1B default-sidecar/empty-text CUDA cold screens](docs/results/2026-11-19-phase5-cli/cuda-lm1b-text-cold-performance-v1.md) compared old ID, new ID, empty text and text+decode, 90 measured processes/condition per size; all fixed output hashes matched. Median paired decode increments were 7.745/7.858/8.885 ms (small/medium/large); causes unisolated. Sanitized export replay, tests and `setpriv --inh-caps=-all tools/check.sh` passed. Shared GPU, one row, no Python/native CLI peak/Metal; release gate pending.

- 2026-11-20: [host tokenizer-stage diagnostic](docs/results/2026-11-19-phase5-cli/linux-text-tokenizer-stage-v1.md) measured decoder construction medians 15.317 ms XSum / 61.856 ms WMT14 on already loaded artifacts; encode/decode of the tested rows was much smaller. This supports, but does not prove, the cause of the cold text delta. Six release-mode test processes, sanitized-series replay, negative tests and `setpriv --inh-caps=-all tools/check.sh` passed after fixing an initial test-only Clippy warning and recollecting. No production inference change; no Metal or release performance gate.

- 2026-11-20: [XSum/WMT14 CUDA CLI text/decode cold screens](docs/results/2026-11-19-phase5-cli/cuda-text-cold-performance-v1.md) compare old ID, unified ID, text and text+decode across 90 paired fresh processes per condition; expected outputs matched. Median paired text−ID increases were 14.766 ms XSum and 74.469 ms WMT14 on one row; causes unisolated. Sanitized trial export replayed, offline tests and `setpriv --inh-caps=-all tools/check.sh` passed. Shared GPU, no native device peaks/Python/warm model/Metal; not release qualification.

- 2026-11-20: published [sanitized CUDA cold trial series](docs/results/2026-11-19-phase5-cli/cuda-cold-trials-v1.json): 90 paired measured time/RSS values per condition for seven profiles, no inputs, outputs or paths. Export audited retained captures; replay, independent aggregate verification, negative controls and `setpriv --inh-caps=-all tools/check.sh` passed. Full raw outputs/telemetry remain local, and these shared-GPU screens do not qualify release performance.

- 2026-11-20: [CPU lifecycle mismatch investigation](docs/results/2026-11-19-phase5-cli/linux-lifecycle-env-investigation-v1.md) reproduced a failed observation with the sampled child environment equal to a later collector read. A test-only post-exec marker separates the fake lifecycle composition test from startup: 30 consecutive fixture runs, focused observer/coordinator tests and `setpriv --inh-caps=-all tools/check.sh` passed. Production observer remains fail-closed and has no authenticated startup barrier; the race is unresolved. No runtime or machine policy change.

- 2026-11-20: [LM1B CUDA/Python repeated model timing](docs/results/2026-11-19-phase5-cli/cuda-lm1b-performance-retest-v1.md) covers small/medium batches 1/4/16 and large 1/4/8 with 3 processes and 30 trials; Rust generation median/RSS was lower, model-load intervals longer. Aggregate replay, negative controls and `setpriv --inh-caps=-all tools/check.sh` passed. Independent RNGs, shared GPU, noncomparable allocator scopes, local-only raw data and no changed-CLI/previous-Rust/Metal comparison leave release qualification pending.

- 2026-11-20: [LM1B CUDA CLI cohort routing](docs/results/2026-11-19-phase5-cli/cuda-lm1b-cli-cohort-v1.md) matched old and unified stdout and retained seed-2000 codes/IDs for 256 rows per size. Independent replay, runner tests and `setpriv --inh-caps=-all tools/check.sh` passed. This is one seed, not full CLI cohort or Metal; raw outputs remain local.

- 2026-11-20: XSum corpus-runner lookup now accepts a relocated checkout without changing pinned summary/capture identities; [two-batch relocated-root CUDA smoke](docs/results/2026-11-19-phase5-cli/cuda-text-input-v1.md#full-corpus-cuda-textid-generation-retest), negative controls and `setpriv --inh-caps=-all tools/check.sh` passed. This is not a fresh-clone or full-corpus rerun of the updated runner.

- 2026-11-20: [full-corpus CUDA text generation](docs/results/2026-11-19-phase5-cli/cuda-text-input-v1.md#full-corpus-cuda-textid-generation-retest) matched retained equivalent ID captures on all 11,334 XSum and 3,003 WMT14 rows; independent replay, runner tests and `setpriv --inh-caps=-all tools/check.sh` passed. [Selected batches](docs/results/2026-11-19-phase5-cli/cuda-text-input-v1.md#selected-multi-batch-cuda-textid-generations) also matched old alias and decoded text/ID paths. XSum source text is reconstructed from IDs; no performance or Metal claim, raw data remain local.

- 2026-11-20: [WMT14 CUDA/Python model performance retest](docs/results/2026-11-19-phase5-cli/cuda-wmt14-performance-retest-v1.md) used three alternating processes and 30 trials at batches 1/4/16/32; measured outputs matched and Rust had lower observed median latency and RSS. Aggregate replay and tests passed. First full check hit the known lifecycle `ObservationMismatch`; focused and full retries passed, cause unknown. Shared desktop, sampled GPU peaks, local-only raw bundle and no CLI/previous-Rust/Metal evidence prevent release qualification.

- 2026-11-20: [paired CUDA cold-request screens for original vs provenance-only Sudoku medium/hard re-exports](docs/results/2026-11-19-phase5-cli/cuda-sudoku-reexport-cold-v1.md) used the same binary and 90 measured processes per condition; fixed two-board outputs matched. Paired intervals include zero; shared desktop, sampled system GPU memory, small batches and no Python comparison prevent a release gate claim. Runner tests and `setpriv --inh-caps=-all tools/check.sh` passed; raw captures remain local.

- 2026-11-20: [LM1B BERT decoding](docs/results/2026-11-19-phase5-cli/cuda-decode-screen-v1.md#lm1b-retained-cuda-cohort-decoding) matched the pinned evaluator on all 61,440 retained CUDA cohort rows across small/medium/large. Reference replay, all three ignored tests, negative control and `setpriv --inh-caps=-all tools/check.sh` passed; this is not CLI cohort or Metal validation, and captures remain local.

- 2026-11-20: the CLI's artifact-bound decoder matched pinned evaluator text for retained CUDA outputs: [3,003 WMT14 rows](docs/results/2026-11-19-phase5-cli/cuda-decode-screen-v1.md#wmt14-retained-cuda-corpus-decode), [11,334 XSum and 1,280 TinyStories rows](docs/results/2026-11-19-phase5-cli/cuda-decode-screen-v1.md#xsum-and-tinystories-retained-cuda-output-decoding). The ignored corpus tests and `setpriv --inh-caps=-all tools/check.sh` passed; modified WMT14 and XSum references failed hash checks. No runtime changes or Metal validation; captures remain local.

- 2026-11-20: [XSum CUDA/Python model performance retest](docs/results/2026-11-19-phase5-cli/cuda-xsum-performance-retest-v1.md) used three alternating processes and 30 trials at batches 1/4/32; outputs matched for those prefixes and Rust had lower observed median latency and RSS. Both aggregate summaries replayed byte-for-byte and `setpriv --inh-caps=-all tools/check.sh` passed. Desktop GPU was shared; this is not changed-CLI, cold-load, previous-Rust or Metal qualification. Raw captures are local only. Remaining release gates and profiling remain open.

- **Phases 0–3 complete.** Phase 2 closed with a [scoped Apple immediate-memory exception](docs/results/2026-11-11-phase2-optimization/phase2-final-audit-v1.md).
  Phase 3 Sudoku is complete on CUDA and Metal ([report](docs/results/2026-11-14-phase3-sudoku/README.md)).
- **Phase 4 complete with scoped, user-accepted exceptions** (2026-11-19, [audit](docs/results/2026-11-19-phase4-audit/phase4-final-audit-v1.md), [contract](docs/PHASE4-TEXT.md), [handoff](RESUME.md)).
- **Phase 5 step 1 in progress** ([PHASE5.md](docs/PHASE5.md)): user chose release first, lean, and FP32 only (2026-11-19). The user approved proceeding with provenance-only Sudoku medium/hard re-exports (qualification still required), empty-only LM1B `--text` mapped to `[101]`, and explicit experimental flags on the legacy XSum alias. The XSum alias flag parser is implemented and host-tested; [CUDA fixed-request routing and experimental flag screens](docs/results/2026-11-19-phase5-cli/cuda-current-routing-v1.md) match the pre-change binary, while broader and Metal validation remain pending. LM1B empty-only text is implemented and screened for one CUDA request per size; broader CUDA and Metal validation remain pending; [full 2,000-board CUDA comparisons](docs/results/2026-11-19-phase5-cli/cuda-sudoku-reexport-corpus-v1.md) matched original and alternate Sudoku outputs, while Metal and performance qualification remain pending. `ncm sample` routes supported profiles to existing paths; ordinary GPU CLI features enable the prior experimental feature paths. A [tracked CUDA routing runner](docs/results/2026-11-19-phase5-cli/cuda-routing-screen-v2.md) matches baseline, alias and unified stdout for one request per profile, including all three Sudoku difficulties; a [decode screen](docs/results/2026-11-19-phase5-cli/cuda-decode-screen-v1.md) matches the pinned references on those text outputs. XSum/WMT14 `--text` has [corpus-exact Linux encoding and CUDA end-to-end screens](docs/results/2026-11-19-phase5-cli/cuda-text-input-v1.md); [LM1B's default rotary sidecar](docs/results/2026-11-19-phase5-cli/cuda-lm1b-rope-v1.md) preserves CUDA output without a command-line rope argument; Metal validation remains pending. TinyStories `--text` now accepts strings encoding to exactly one valid non-padding token, with a [CUDA screen](docs/results/2026-11-19-phase5-cli/cuda-text-input-v1.md#tinystories-text-input); LM1B `--text` accepts only empty strings mapped to `[101]`; [one CUDA screen per size](docs/results/2026-11-19-phase5-cli/cuda-lm1b-empty-text-v1.md#all-three-sizes) matches the equivalent ID input. [Linux export investigation](docs/results/2026-11-19-phase5-cli/linux-export-repro-v1.md) reproduced seven qualified manifests exactly; Sudoku medium/hard historical exporter is missing; provenance-only re-exports are approved for further qualification. [Setup guide](docs/USAGE.md) is incomplete; a [single-commit Linux fresh-clone replay](docs/results/2026-11-19-phase5-cli/linux-guide-single-commit-v3.md) covered all nine downloads/exports and synthetic examples, but full qualification and complete Mac preparation and verification remain pending. An [interim step 1 audit](docs/results/2026-11-19-phase5-cli/step1-interim-audit-v1.md) found the legacy XSum alias accepts unflagged experimental positional options despite the explicit-flag requirement; the approved explicit-flag change now has host and fixed-request CUDA coverage; broader equivalence, performance and release gate remain pending.

| Profile | Status |
| --- | --- |
| TinyStories | Advanced under [user-accepted exceptions](docs/results/2026-11-15-phase4-tinystories/user-accepted-exceptions-v1.md) |
| XSum | Advanced under the [Metal](docs/results/2026-11-15-phase4-xsum/user-accepted-exception-v1.md) and [CUDA stochastic](docs/results/2026-11-15-phase4-xsum/user-accepted-exceptions-v1.md) exceptions |
| WMT14 | Scoped CUDA and Metal top-1 qualification ([evidence](docs/results/2026-11-16-phase4-wmt14/mlx-corpus-retry-v1.md#cli-and-profile-qualification)) |
| LM1B-small | **Qualified on CUDA and Metal:** 80-seed quality passes on both; timing passes on both, including Metal RSS after the [memory fix](docs/results/2026-11-16-phase4-lm1b/small-metal-timing-v1.md#after-the-memory-fix-all-checks-pass) |
| LM1B-medium/large | **Lean bar met on CUDA and Metal:** teacher-forced parity ([CUDA](docs/results/2026-11-16-phase4-lm1b/medium-large-cuda-parity-v1.md), [Metal](docs/results/2026-11-16-phase4-lm1b/medium-large-metal-v1.md)), 80-seed quality ([CUDA](docs/results/2026-11-16-phase4-lm1b/cohort-campaign-v1.md#prospective-cuda-results-medium-and-large), [Metal](docs/results/2026-11-16-phase4-lm1b/cohort-campaign-v1.md#prospective-metal-results-medium-and-large)), CLI generation, and timing faster than Python ([CUDA](docs/results/2026-11-16-phase4-lm1b/medium-large-cuda-timing-v1.md), [Metal](docs/results/2026-11-16-phase4-lm1b/medium-large-metal-v1.md#latency-and-memory-against-python-mlx)) |
| OWT | Config-only coverage done: pinned config validated (`OwtConfigOnly`), artifacts rejected, synthetic operators at OWT shapes match naive references ([coverage](docs/results/2026-11-19-phase4-owt/config-operator-coverage-v1.md)). Weights unreleased |

- **LM1B-small evidence:** CUDA fixed-history traces, greedy and CLI checks pass. Repeated CUDA
  timing beats pinned Python in median, p95 and process RSS at batches 1, 4 and 16; Rust model load
  is slower ([performance](docs/results/2026-11-16-phase4-lm1b/small-cuda-capacity-screen-v1.md#repeated-cuda-timing-and-allocator-measurements)).
  The frozen five-seed entropy interval failed on CUDA and every observed-seed retest; the
  failures stay on record ([quality](docs/results/2026-11-16-phase4-lm1b/small-heldout-cuda-v1.md)).
  The approved **80-seed CUDA gate passes**: perplexity +0.44 [-0.27, 1.15], entropy +0.00073
  [-0.00032, 0.00180] ([result](docs/results/2026-11-16-phase4-lm1b/cohort-campaign-v1.md#prospective-cuda-result)).
  Metal fixed-history decisions and IDs match CUDA; draw-level differences are covered by
  user decision 2 below.
  The independent [Python MLX reference](docs/results/2026-11-16-phase4-lm1b/small-controlled-cuda-v1.md#metal-development-reference-for-complete-requests)
  is `tools/reference/lm1b_small_mlx_reference.py`.

- **Metal results (small):** the 80-seed gate passes, with perplexity +0.29 [-0.41, 0.97] and entropy +0.00103
  [-0.00007, 0.00213] ([result](docs/results/2026-11-16-phase4-lm1b/cohort-campaign-v1.md#prospective-metal-result)).
  After the memory fix (`57e5ee6`, output byte-identical), timing passes median, p95 and RSS at batches 1/4/16
  (13/15/28% faster; RSS margin 0.4–0.6%) ([timing](docs/results/2026-11-16-phase4-lm1b/small-metal-timing-v1.md)).
- **macstudio is always available** for testing; recheck its state before use.

## User decisions (2026-11-18)

Recorded in [user-decisions-v1.md](docs/results/2026-11-16-phase4-lm1b/user-decisions-v1.md):

1. The fresh **80-seed quality protocol** is the LM1B gate (same scorer, margins and bootstrap).
   Seeds are pre-registered in [cohort-seeds-v1.json](docs/results/2026-11-16-phase4-lm1b/cohort-seeds-v1.json)
   (SHA-256 `c9800090de18f18d6719993f1ac36057b6cea479a26cc2e4b3115de6e4beeab1`). Do not change them after generation starts.
2. **Stop Metal-vs-CUDA draw-level parity work.** Those differences are a scoped exception; fixed-history
   decision/ID differences remain failures.
3. **Medium/large bar:** export plus identity checks, teacher-forced decision/ID parity at 1–2 shapes,
   CLI generation, 80-seed quality, one batch-sweep latency/memory comparison per device. No new
   numerical deep-dives unless one of these fails.
4. The map was compacted; history is archived.

## Next actions, in order

Run checks under `setpriv --inh-caps=-all`; the shell may carry an inheritable capability.

1. **Phase 5 step 1 (CUDA, then Metal):** complete unified `ncm sample` (LM1B defaults to a separately exported, hash-pinned sibling rotary file), confirm byte-identical output against the current commands; finish validating the `*-experimental` feature folding; validate `--decode` and finish `--text`
   for every text profile; XSum/WMT14 corpus encoding passes on Linux, CUDA end-to-end is screened, Metal remains pending; `docs/USAGE.md` verified from a fresh clone; README and release
   notes; then ask before tagging `v0.1.0`.
2. Steps 2–4 follow [PHASE5.md](docs/PHASE5.md): profile, user picks targets, targeted optimizations, close.
3. **Retained local inputs** after the 2026-11-19 cleanup: pinned upstream checkpoints and evaluators, one current
   artifact per profile, the reference environments, `reports/phase4-lm1b-level7-replay-v1/reference-rope-tables.safetensors`
   and `reports/phase4-wmt14-uncached-cfg-link-v1/` (Linux), and `reports/phase4-lm1b-metal-cli-rope-v1.safetensors` (Mac).
   Raw Phase 4 captures were deleted; the docs keep their hashes.

## Working rules carried forward

- Linux/CUDA first, then Metal, for each step. `ssh macstudio` is authorized; inspect the clean
  remote checkout and fast-forward pull pushed commits; do not reboot the Mac or interrupt `dotsetup.rmlx`.
- Keep `.pi/` untouched. Ignored `reports/` hold raw captures; tracked docs hold identities and conclusions.
- CUDA check: `NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 ./tools/check.sh --cuda`.

## Checks run for this update

- **TinyStories CUDA/Python repeated model timing:** three alternating processes × 5 warmups/30 trials at batches 1/4/32/128 favored Rust in measured median and RSS ([report](docs/results/2026-11-19-phase5-cli/cuda-tinystories-performance-retest-v1.md)). Desktop GPU activity was present; 1 Hz GPU samples are not native peaks. The tracked aggregate replayed byte-for-byte and `setpriv --inh-caps=-all ./tools/check.sh` passed. This is model-probe timing, not CLI cold load, previous-Rust comparison, an isolated GPU or the release performance gate.

- **LM1B-large CUDA 80-seed retest:** nonblind frozen-registry campaign passed PPL/entropy mean and 95% interval margins; all 160 scores were upstream-exact and per-seed metrics reproduced the original prospective records ([report](docs/results/2026-11-19-phase5-cli/cuda-lm1b-large-quality-retest-v1.md)). Summary replay and `setpriv --inh-caps=-all ./tools/check.sh` passed. Performance and Metal remain pending; no full CLI cohort equivalence is implied.

- **LM1B-medium CUDA 80-seed retest:** nonblind frozen-registry campaign passed PPL/entropy mean and 95% interval margins; all 160 scores were upstream-exact and per-seed metrics reproduced the original prospective records ([report](docs/results/2026-11-19-phase5-cli/cuda-lm1b-medium-quality-retest-v1.md)). Summary replay and `setpriv --inh-caps=-all ./tools/check.sh` passed. Large, performance and Metal remain pending; no full CLI cohort equivalence is implied.

- **LM1B-small CUDA 80-seed retest:** nonblind frozen-registry campaign passed both PPL/entropy mean and 95% interval margins; all 160 score records are upstream-exact, and compact per-seed metrics plus comparison reproduce the original prospective records ([report](docs/results/2026-11-19-phase5-cli/cuda-lm1b-small-quality-retest-v1.md)). Summary replay and `setpriv --inh-caps=-all ./tools/check.sh` passed. Medium/large, performance and Metal remain pending; no full CLI cohort equivalence is implied.

- **LM1B nonblind retest mode:** campaign tooling now explicitly labels CUDA repeats of the original frozen 80-seed registry as `retest`, pins the registry and ordered cohorts, and fails on quality-margin failures without changing scorer, margins, bootstrap or earlier records ([contract](docs/results/2026-11-16-phase4-lm1b/cohort-campaign-v1.md#reproduce)). Three offline negative controls and `setpriv --inh-caps=-all ./tools/check.sh` pass. No new GPU quality or performance evidence yet; run CUDA retests before Metal.

- **XSum CUDA top-1 quality:** nonblind full/held-out retests passed the frozen quality rule on 11,334 rows; all IDs/text and metrics match Python, with the prior single row-2690 intermediate-code tie retained. The unified CLI matched every Rust capture row in the frozen 356 batches ([report](docs/results/2026-11-19-phase5-cli/cuda-xsum-quality-retest-v1.md)). `setpriv --inh-caps=-all ./tools/check.sh` passed. Raw evaluation-derived captures remain ignored locally; performance and Metal are pending.

- **WMT14 CUDA base quality:** regenerated pinned Python and current optimized Rust captures on all 3,003 rows; codes/IDs matched exactly, and both independent scorers reproduced the passing full/held-out frozen receipts. The actual unified CLI also matched Python on all 751 frozen batches ([report](docs/results/2026-11-19-phase5-cli/cuda-wmt14-quality-retest-v1.md)). `setpriv --inh-caps=-all ./tools/check.sh` passed. Nonblind retest; prior CFG=1 failure remains, and performance and Metal remain pending.

- **TinyStories CUDA five-seed quality:** regenerated and scored 256 samples per official seed per runtime against pinned Python; frozen PPL/entropy margins pass in a nonblind retest, and all Rust codes/IDs match pre-change and current CLI chunks ([report](docs/results/2026-11-19-phase5-cli/cuda-tinystories-quality-retest-v1.md)). The qualified streaming-scorer caveat and initial missing-`regex` failure are recorded; `setpriv --inh-caps=-all ./tools/check.sh` passed. This is not performance or Metal evidence.

- **TinyStories CUDA five-seed routing:** pre-change, alias and unified CLI outputs matched byte-for-byte for 128 rows at each official seed ([report](docs/results/2026-11-19-phase5-cli/cuda-tinystories-five-seed-routing-v1.md)). `setpriv --inh-caps=-all ./tools/check.sh` passed. This is not the 256-per-seed evaluator quality gate, Python RNG parity, performance or Metal.

- **CUDA Sudoku CLI corpus regression:** pre-change `generate` and current `sample` matched byte-for-byte on all 6,000 boards across the three original artifacts; scores against retained dataset references matched Phase 3 ([report](docs/results/2026-11-19-phase5-cli/cuda-sudoku-baseline-corpus-v1.md)). Medium/hard current captures were reused from the preceding re-export screen; `setpriv --inh-caps=-all ./tools/check.sh` passed. This is not a new direct Python comparison, performance or Metal evidence.

- **LM1B empty text all sizes:** small, medium and large matched equivalent `[101]` input byte-for-byte, including decoded text, for one CUDA request each ([report](docs/results/2026-11-19-phase5-cli/cuda-lm1b-empty-text-v1.md#all-three-sizes)). `setpriv --inh-caps=-all ./tools/check.sh` passed. This is not other batch sizes, quality, performance or Metal evidence.

- **Sudoku re-export CUDA corpus:** original and alternate medium/hard artifacts produced byte-identical codes and IDs on all 2,000 boards per difficulty; scores against dataset references matched the prior Phase 3 metrics ([report](docs/results/2026-11-19-phase5-cli/cuda-sudoku-reexport-corpus-v1.md)). `setpriv --inh-caps=-all ./tools/check.sh` passed. This is not a new direct Python comparison, performance or Metal qualification.

- **Linux single-commit guide replay:** all 15 documented Linux shell blocks passed in a fresh clone at `c63b205`, including nine downloads/manifest checks and `tools/check.sh --cuda` (18 operator cases); inspected worktrees and clone were removed ([report](docs/results/2026-11-19-phase5-cli/linux-guide-single-commit-v3.md)). The local default `setpriv --inh-caps=-all ./tools/check.sh` also passed. The known offline Python `_ssl` import failure remains documented. This is not quality/performance, Sudoku re-export qualification or Mac/Metal verification.

- **Resumed Linux fresh-clone guide:** after the missing-directory failure and CUDA check retry, the clone was fast-forwarded to the fix and guide blocks 4–16 passed, including fresh downloads/exports of all nine profiles and synthetic examples ([report](docs/results/2026-11-19-phase5-cli/linux-guide-resumed-v2.md)). The inspected worktrees and clone were removed; `setpriv --inh-caps=-all ./tools/check.sh` passed. This is a resumed two-commit replay, not uninterrupted single-commit guide verification, quality/performance or Metal qualification.

- **Fresh-clone check regression:** full Linux guide attempt at `740eb3b` failed because `reports/` was absent when four report tests ran. `tools/check.sh` now creates it; default local check and patched fresh-clone `--cuda` retry passed (18 operator cases). [Evidence](docs/results/2026-11-19-phase5-cli/linux-fresh-reports-fix-v1.md). The clone was later fast-forwarded to the fix and its remaining guide blocks replayed; the inspected clone was then removed. No uninterrupted single-commit guide or Metal pass is claimed.

- **Sudoku medium/hard guide fresh-download slice:** extracted the draft guide block and ran it in the existing Linux checkout; fresh pinned downloads and both alternate manifest checks passed, as did separate download verification ([evidence](docs/results/2026-11-19-phase5-cli/linux-export-repro-v1.md#mediumhard-draft-guide-replay-with-fresh-downloads)). The inspected exporter worktree was removed and `setpriv --inh-caps=-all ./tools/check.sh` passed. This is not a fresh-clone, model-quality, performance or Metal gate.

- **Current CUDA routing:** the tracked runner matched pre-change, alias and unified output for nine fixed profiles; three additional XSum cases matched for positive CFG, stochastic sampling and both ([report](docs/results/2026-11-19-phase5-cli/cuda-current-routing-v1.md)). A first run rejected an unexpected candidate binary hash before writing output; rebuilding the CUDA-feature binary restored its identity. `setpriv --inh-caps=-all ./tools/check.sh` passed. These are fixed-input screens, not corpus, quality, performance or Metal qualification.

- **Sudoku medium/hard draft export recipe:** replayed its offline worktree/export sequence with retained pinned inputs; both alternate manifests matched the approved candidate hashes and exports reported bit-exact tensor roundtrips ([evidence](docs/results/2026-11-19-phase5-cli/linux-export-repro-v1.md#mediumhard-draft-guide-replay-with-retained-inputs)). All 16 guide shell blocks pass `bash -n`; `setpriv --inh-caps=-all ./tools/check.sh` passed. Fresh downloads, full fresh-clone replay, quality/performance and Metal remain pending.

- **Sudoku re-export CUDA screen:** retained original and alternate medium/hard artifacts matched byte-for-byte in CLI output for the existing two-row fixtures ([report](docs/results/2026-11-19-phase5-cli/cuda-sudoku-reexports-v1.md)). `setpriv --inh-caps=-all ./tools/check.sh` passed. The earlier provenance comparison and this screen do not establish full corpus, performance, fresh-clone or Metal qualification.

- **LM1B empty-only text:** parser/unit, CUDA-feature CLI tests and CUDA-feature all-target Clippy pass; one CUDA small-model text/ID request including decode matched byte-for-byte and a nonempty prompt was rejected ([screen](docs/results/2026-11-19-phase5-cli/cuda-lm1b-empty-text-v1.md)). `tools/check.sh --cuda` passed (18 operator cases). An earlier default `tools/check.sh` failed lifecycle `capture_error:ObservationMismatch`; the subsequent CUDA check's full host suite passed, cause unknown. Initial CUDA CLI test could not find cuDNN; retry with its reference library path passed. No broader profile, performance, quality or Metal qualification is implied.

- **XSum alias experimental opt-in:** positive positional CFG and a positional seed now require their respective explicit experimental flags; qualified top-1 syntax remains unchanged. `cargo test -p ncm-cli`, `setpriv --inh-caps=-all ./tools/check.sh`, and the documented `setpriv` CUDA check passed (18 operator cases). These parser and operator checks do not establish XSum inference equivalence, quality, performance or Metal correctness.

- **Release decisions recorded:** user approved the three choices above; `setpriv --inh-caps=-all ./tools/check.sh` passed. Documentation only; no new GPU qualification or runtime changes.

- **Cold aggregate capture audit:** added a [reproducible local audit](docs/results/README.md#phase-5-release-candidate-in-progress) of the seven compact aggregates against pinned raw-result/GPU CSV hashes, 1,470 captured stdout files, trial identities and independently recomputed statistics. All seven passed; four tests with tampered/duplicate/invalid/symlink negative controls and `tools/check.sh` pass. This checks retained captures, not binary source attestation, benchmark validity, quality, GPU isolation or release qualification.

- **Phase 5 results index and aggregates:** indexed the scoped release-candidate reports and committed [compact cold-screen aggregates](docs/results/2026-11-19-phase5-cli/cuda-cold-aggregates-v1.json). Independently recomputed seven profiles' 90-sample statistics from retained raw records and verified result/telemetry hashes, binary identities and all 210 per-screen output hashes; index links and `tools/check.sh` pass. Raw captures remain ignored/local, not durably published; these exploratory numbers do not satisfy quality/performance or release gates. Documentation/data only; no runtime impact.

- **CUDA LM1B-medium/large cold-request screens:** each size's 210 baseline/candidate processes produced identical stdout; measured median ms were 2727.627/2727.322 (medium) and 4851.750/4856.100 (large), with no meaningful improvement established ([report](docs/results/2026-11-19-phase5-cli/cuda-lm1b-medium-large-cold-screen-v1.md)). Report syntax/links and `tools/check.sh` pass. These active-desktop, batch-1 cold screens are not persistent inference, native device peaks, Python comparison, quality, Metal or the release performance gate.

- **CUDA LM1B-small cold-request screen:** extended the cold runner with a hash-checked explicit rotary table; 210 baseline/candidate processes produced identical stdout. Measured medians were 1578.016/1578.826 ms and median peak RSS 1209.566/1206.109 MiB ([report](docs/results/2026-11-19-phase5-cli/cuda-lm1b-cold-screen-v1.md)). Nine runner tests, report syntax/links and `tools/check.sh` pass. This active-desktop, single-request cold screen is not persistent inference, native device peaks, Python comparison, quality, Metal or the release performance gate.

- **CUDA XSum/WMT14 cold-request screens:** extended the runner for top-1 requests; each profile's 210 baseline/candidate processes produced identical stdout. Measured median ms were 1285.126/1283.963 (XSum) and 1837.986/1834.103 (WMT14), with no meaningful improvement established ([report](docs/results/2026-11-19-phase5-cli/cuda-xsum-wmt14-cold-screen-v1.md)). Seven runner tests, report syntax/links and `tools/check.sh` pass. Active-desktop cold-process timing is not persistent inference, native device peaks, Python comparison, quality, Metal or the release performance gate.

- **Invalid-device CLI screen:** the CUDA-feature unified CLI rejected one deliberately invalid device for all nine qualified profile manifests, with exit 1, empty stdout and explicit errors ([evidence](docs/results/2026-11-19-phase5-cli/cuda-routing-screen-v2.md#invalid-device-checks-with-qualified-artifacts)). Two runner negative controls, report syntax/links and `tools/check.sh` pass. This covers one invalid name on Linux, not all device/option combinations, inference or Metal.

- **TinyStories multi-token decode:** the artifact-bound CLI decoder matched pinned upstream on 137 synthetic rows including split/partial UTF-8, EOS/pad and 128 fixed-seed random sequences ([evidence](docs/results/2026-11-19-phase5-cli/cuda-decode-screen-v1.md#synthetic-tinystories-multi-token-decode)). Two references match byte-for-byte; output-preflight, ignored artifact-bound test and `tools/check.sh` pass. This is host-side synthetic coverage, not generated-corpus, GPU or Metal qualification.

- **TinyStories single-token decode:** artifact-bound CLI decoder matched the existing pinned upstream reference for all 8,192 vocabulary IDs, with EOS/pad handled under evaluator policy ([evidence](docs/results/2026-11-19-phase5-cli/cuda-decode-screen-v1.md#tinystories-single-token-vocabulary-decode)). The ignored real-artifact test, report syntax/links and `tools/check.sh` pass. This does not cover multi-token sequences, generated corpus, GPU inference or Metal.

- **Synthetic decoder boundaries:** artifact-bound XSum/WMT14 `Decoder::add_texts` matched pinned Python evaluator decoding for 135 synthetic rows per profile, including EOS, padding, model-only IDs and fixed-seed random IDs ([evidence](docs/results/2026-11-19-phase5-cli/cuda-decode-screen-v1.md#synthetic-xsum-and-wmt14-decode-boundaries)). Two reference captures per profile matched; output-preflight test, the ignored artifact-bound test and `tools/check.sh` pass. This is host-side synthetic coverage, not corpus, GPU inference or Metal validation.

- **Updated equivalence runner CUDA rerun:** the hardened runner compared baseline, alias and unified stdout for all nine fixed requests; hashes and row counts match the previous screen ([evidence](docs/results/2026-11-19-phase5-cli/cuda-routing-screen-v2.md#updated-runner-cuda-rerun)). Five runner tests and `tools/check.sh` pass. This repeats the existing fixture scope with the updated runner and newer candidate, not corpus, quality, performance or Metal validation.

- **CUDA TinyStories cold-request screen:** extended the cold runner for a fixed seed-42 TinyStories request; 210 fresh baseline/candidate processes produced identical stdout, with measured medians 1103.932/1103.759 ms and median peak RSS 908.039/907.695 MiB ([report](docs/results/2026-11-19-phase5-cli/cuda-tinystories-cold-screen-v1.md)). Six runner tests, report syntax/links and `tools/check.sh` pass. This active-desktop screen includes startup/loading, not persistent-process inference, native device peaks, Python comparison, other profiles, Metal or the performance gate.

- **Interim step 1 audit:** mapped the release-candidate requirements to scoped artifacts and open gates ([audit](docs/results/2026-11-19-phase5-cli/step1-interim-audit-v1.md)); observed legacy XSum positive-CFG/seed calls reach input loading without experimental flags while unified calls reject them. No inference was run. Audit links and `tools/check.sh` pass; GitHub release lookup timed out, so remote publication status is unknown. This is not a final Phase 5 audit or release gate.

- **CLI equivalence runner output safety:** reject live/dangling output symlinks before resolving the destination; five offline tests and `tools/check.sh` pass. The earlier nine-profile CUDA screen retains its original runner identity ([evidence](docs/results/2026-11-19-phase5-cli/cuda-routing-screen-v2.md)); the subsequent updated-runner rerun is recorded above.

- **CUDA overlength text:** one synthetic 1,101-token input per XSum/WMT14 was tested against its pinned Python-truncated IDs: baseline alias, candidate alias, unified IDs and unified text matched byte-for-byte; decoded ID/text paths matched and preserved IDs/codes. The [documented commands](docs/results/2026-11-19-phase5-cli/cuda-text-truncation-v1.md) were replayed with fresh outputs. Report shell blocks/links and `tools/check.sh` pass. This is not corpus generation, quality, performance or Metal evidence.

- **Text truncation/padding boundaries:** seven synthetic inputs per profile, including overlength, empty, Unicode and special strings, matched the pinned Python XSum/WMT14 encoders in an artifact-bound Rust test ([evidence](docs/results/2026-11-19-phase5-cli/cuda-text-input-v1.md#synthetic-truncation-and-padding-boundaries)). Three references per profile were byte-identical; output preflight and `tools/check.sh` pass. This checks encoding only, not model acceptance, GPU generation or quality.

- **Unified CLI rejection integration:** [ten synthetic-manifest cases](docs/results/2026-11-19-phase5-cli/cuda-routing-screen-v2.md#cli-rejection-integration-checks) exercise actual CLI rejection diagnostics and no stdout for unsupported profile/option combinations and two LM1B device/flag cases. Default and `cuda-cudnn` test builds, CUDA CLI all-target Clippy and `tools/check.sh` pass. All cases stop before artifact loading or inference; this is not a complete device matrix or GPU validation.

- **Integrated fresh-clone Linux guide replay:** the CUDA build/check, tokenizer setup, seven fresh downloads/qualified exports, both rotary references and synthetic Sudoku/text examples passed in one new clone; its inspected worktrees and clone were removed ([evidence](docs/results/2026-11-19-phase5-cli/linux-guide-integrated-replay-v1.md)). `tools/check.sh --cuda` reported 18 passing operator results; 15 guide shell blocks and links pass syntax/link checks. Sudoku medium/hard, full Mac guide, broader parity and quality/performance gates remain pending.

- **Synthetic text examples:** added [draft CUDA commands](docs/USAGE.md#synthetic-text-examples--cuda-development-screen) for TinyStories, XSum, WMT14 and LM1B-small. Four one-row requests passed with retained qualified artifacts and decoded text ([scope](docs/results/2026-11-19-phase5-cli/cuda-synthetic-text-examples-v1.md)); 14 guide shell blocks, links and `tools/check.sh` pass. They are smoke tests, not fresh-clone, Metal, parity, quality or performance evidence.

- **Fresh-clone Linux Sudoku guide replay:** extracted and followed the guide's CUDA build, `tools/check.sh --cuda` (18 operator results), checkpoint download, historical export and synthetic `sample`/`generate` blocks; the qualified manifest and identical output passed. The inspected clone was removed ([evidence](docs/results/2026-11-19-phase5-cli/linux-guide-sudoku-replay-v1.md)); 13 guide shell blocks and links pass syntax/link checks. Only one profile was replayed, not the full Linux or Mac guide, quality or performance gate.

- **Fresh-clone Linux tokenizer slice:** followed the guide's upstream clone, tokenizer venv, pinned network fetch and conversion from a new clone; all four tokenizer manifests and payloads matched the tracked lock, and the inspected temporary clone was removed ([evidence](docs/results/2026-11-19-phase5-cli/linux-fresh-clone-tokenizers-v1.md)). Added the exercised hash check to the guide; thirteen shell blocks, links and `tools/check.sh` pass. Full setup replay, model execution and Metal remain pending.

- **LM1B sidecar output safety:** offline reference exporter now exclusively creates its output, rejecting existing files and live/dangling symlinks, with three dependency-free tests added to `tools/check.sh`. A fresh Linux reference export exactly matches the pinned sidecar and the prior bytes ([evidence](docs/results/2026-11-19-phase5-cli/cuda-lm1b-rope-v1.md)); `tools/check.sh` passes. This is not new GPU, quality or fresh-clone evidence.

- **WMT14 and LM1B export recipes:** added [Linux draft commands](docs/USAGE.md#linux-wmt14-and-lm1b-development-replays) for WMT14, all LM1B sizes and the shared rotary sidecar. Retained inputs reproduced the four qualified manifest hashes and sidecar hash ([scope](docs/results/2026-11-19-phase5-cli/linux-export-repro-v1.md#wmt14-and-lm1b-guide-command-replays)); twelve shell blocks, document links and `tools/check.sh` pass. No fresh-clone, Mac, inference or quality gate is implied; complete guide replay and Sudoku medium/hard decision remain pending.

- **XSum export recipe:** added the pinned rotary preparation and historical exporter to the [draft guide](docs/USAGE.md#linux-xsum-development-replay); retained inputs reproduced both the rotary payload and qualified manifest ([scope](docs/results/2026-11-19-phase5-cli/linux-export-repro-v1.md#xsum-guide-command-replay)). Nine guide shell blocks, two rotary tests and `tools/check.sh` pass. No fresh-clone, Mac, inference or quality gate is implied; remaining exporter recipes and complete guide replay are pending.

- **TinyStories export recipe:** added a historical exporter and qualified manifest check to the [draft guide](docs/USAGE.md#linux-tinystories-development-replay); retained sources verified and the export replay matched the qualified manifest ([scope](docs/results/2026-11-19-phase5-cli/linux-export-repro-v1.md#tinystories-guide-command-replay)). This did not use a fresh clone or download; the full guide and remaining exporters are still pending. Eight guide shell blocks pass `bash -n`; `tools/check.sh` initially failed the intermittent clock-handoff test, then its four focused tests and a full retry passed (cause unknown).

- **CUDA CLI cold-request screen:** [Sudoku-easy report](docs/results/2026-11-19-phase5-cli/cuda-sudoku-cold-screen-v1.md) compares baseline vs unified CLI over 90 alternating fresh processes per condition (plus warmups), with byte-identical outputs. Medians 555.115/554.145 ms and median peak RSS 587.965/587.773 MiB; no meaningful improvement established. Earlier timed-wait screens had biased latency and are invalid. This includes load/process startup, not warm in-process inference, and lacks native device peaks, Python comparison, other profiles and Metal. Five runner tests and default `tools/check.sh` pass. The Phase 5 performance gate remains pending.

- **TinyStories vocabulary text differential:** pinned upstream and Rust agree on acceptance and admitted ID for 8,199 decoded-vocabulary/boundary texts (8,064 accepted, 135 rejected); rejected full ID sequences are not compared. [Reference generator and reproduction](docs/results/2026-11-19-phase5-cli/cuda-text-input-v1.md#tinystories-vocabulary-differential), ignored artifact-bound Rust test and two generator-preflight tests pass. CLI feature tests/Clippy and two final `tools/check.sh --cuda` runs pass; an earlier unrelated clock-handoff observation mismatch failed and is recorded. No new GPU quality/performance evidence is implied.

- **Fresh-clone Linux Sudoku-easy slice:** a new remote clone installed the Linux reference lock, built CUDA CLI, downloaded pinned sources, reconstructed the historical exporter and qualified manifest, compared synthetic sample/alias output, and passed `tools/check.sh --cuda` (18 GPU cases). The clone and inspected worktree were removed. [Evidence](docs/results/2026-11-19-phase5-cli/linux-fresh-clone-sudoku-v1.md); other profiles, documentation replay and Metal remain pending. Corrected the earlier easy-exporter SHA transcription, checked documentation links and shell syntax, and ran default `tools/check.sh`.

- **Bounded checkpoint downloads:** added [fetch_checkpoints.py](tools/reference/fetch_checkpoints.py), reusing existing exporter pins with explicit byte budget, streamed size/hash checks, plan and verify modes. All nine retained profile sources verify; a fresh 162,378,392-byte Sudoku-easy network download matches retained files, and its export's non-provenance bytes match the qualified artifact ([screen](docs/results/2026-11-19-phase5-cli/checkpoint-download-screen-v1.md)). Seven downloader tests and final `tools/check.sh` pass. The first full check failed an unrelated lifecycle `ObservationMismatch`; a focused retry and two full reruns passed, cause unknown. Full fresh-clone workflow remains pending.

- **Setup guide draft:** [docs/USAGE.md](docs/USAGE.md) documents current toolchain, pinned-source and download layout, with incomplete export and fresh-clone steps explicitly marked. Eleven small checkpoint-source files (nine configs and two TinyStories BPE files) fetched from the pinned revision and matched exporter hashes; no weights downloaded in that screen. Markdown file/heading links resolve, shell blocks pass `bash -n`, and `tools/check.sh` passes. Full downloads, exporter recipe and Linux/Mac fresh-clone verification remain pending.

- **TinyStories text input:** encode with the artifact tokenizer; reject empty, padding and multi-token strings rather than truncating. The pinned upstream tokenizer and Rust agree on three representative inputs; CUDA baseline/alias/ID/text output matches for a three-row request, with decode matching too. [Screen](docs/results/2026-11-19-phase5-cli/cuda-text-input-v1.md#tinystories-text-input). CLI tests (12 passed; artifact-dependent TinyStories test also run), feature Clippy, full XSum/WMT14 corpus-encoding retest and `tools/check.sh --cuda` pass. Metal and performance evidence remain pending.

- **README and release-note draft:** updated the profile matrix and linked scoped evidence; release and fresh-clone gates remain pending. Corrected the earlier TinyStories fixed-start claim: the model accepts one non-padding prefix ID; CUDA seed 42 with `[[2]]` matches baseline/alias/unified and differs from `[[0]]` at 639/641 codes ([screen](docs/results/2026-11-19-phase5-cli/cuda-text-input-v1.md#tinystories-prefix-correction)). README/draft relative links resolve and `tools/check.sh` passes; local tag and GitHub release `v0.1.0` are absent.

- **Sudoku provenance comparison:** searched every local and Mac Git blob and 2,171 bounded local files without locating the historical exporter. A [tracked verifier](tools/reference/compare_artifact_provenance.py) directly compares every non-provenance payload byte and verifies hashes, inventories and only the specified provenance change; it passed for both retained medium/hard re-exports. Six verifier tests and `tools/check.sh` pass. [Details](docs/results/2026-11-19-phase5-cli/linux-export-repro-v1.md#byte-level-comparison-of-alternate-exports). The user subsequently approved the provenance-only re-exports for further qualification; neither is yet qualified by this comparison alone.

- **Reproducible CLI screen:** [CUDA report](docs/results/2026-11-19-phase5-cli/cuda-routing-screen-v2.md) records a tracked runner passing nine fixed profile requests and four unit tests including negative controls; `tools/check.sh` passes (537 report tests). A read-only Mac checkout/unreachable-blob search did not find the missing historical Sudoku exporter; no Mac files changed or MLX test run.

- **Linux export reproducibility (investigation only):** [report](docs/results/2026-11-19-phase5-cli/linux-export-repro-v1.md) records seven exact manifest reproductions and the two Sudoku medium/hard provenance-only mismatches. A project-local Python 3.14.9 build-tree interpreter enabled old exporters; installation failed on `_ssl`, and no general Python setup or fresh clone is claimed. `tools/check.sh` passes. Original artifacts and pins unchanged.

- **LM1B rotary sidecar and help:** [CUDA screen](docs/results/2026-11-19-phase5-cli/cuda-lm1b-rope-v1.md): default sidecar, override and baseline outputs match for small/medium/large with and without decode; missing and wrong-hash sidecars are rejected. CLI tests (11 passed), CLI Clippy and `tools/check.sh --cuda` pass. `ncm --help` and `ncm sample --help` describe current options and experimental paths; `tools/check.sh` also passed. Metal, fresh-clone setup, and text-input decision are pending.

- **Phase 5 text input:** XSum 11,334/11,334 and WMT14 3,003/3,003 Rust-encoded padded prefixes match pinned reference rows; first-row CUDA generations match ID input both with and without decoding. [Evidence and reproduction](docs/results/2026-11-19-phase5-cli/cuda-text-input-v1.md). CLI tests (10 passed, corpus test separately run), CLI Clippy and `tools/check.sh --cuda` pass. TinyStories/LM1B `--text` and Metal remain pending.

- **Phase 5 text decoding:** CUDA CLI tests (10 passed), CLI all-target Clippy and `tools/check.sh --cuda` pass, including 18 real CUDA operator cases. [CUDA decode screen](docs/results/2026-11-19-phase5-cli/cuda-decode-screen-v1.md) matches evaluator decoding on one retained generated row for each text profile, with codes/IDs unchanged. Corpus-wide and Metal evidence remains pending.

- **Phase 5 CLI routing:** CLI binary tests (9 passed) and all-target Clippy with `cuda-cudnn`; `setpriv --inh-caps=-all ... ./tools/check.sh --cuda` passes, including 18 real CUDA operator cases. [CUDA screen](docs/results/2026-11-19-phase5-cli/cuda-routing-screen-v1.md) compares baseline, alias and unified output for one request per profile. CLI link/test required `LIBRARY_PATH=$PWD/.reference/phase4-cudnn-probe-lib` and `LD_LIBRARY_PATH=$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib`; the first attempt without those failed to link `-lcudnn`. Metal and performance checks are pending.

- **Phase 5 plan (docs only) and cleanup:** wrote `docs/PHASE5.md`. Deleted untracked data not needed for Phase 5 on both
  machines: raw reports, superseded artifacts, build caches, old `.reference` snapshots and logs. Afterwards
  `tools/check.sh --cuda` and `tools/check.sh --mlx` pass, and the LM1B CLI samples and decodes on both GPUs.
- **Phase 4 audit (docs only):** [audit](docs/results/2026-11-19-phase4-audit/phase4-final-audit-v1.md) maps every
  profile to tokenizer, teacher-forced, end-to-end, quality and performance evidence on both GPUs; all relative links
  resolve. LM1B tokenizer, codec and license bytes are identical across sizes. Default `tools/check.sh` passes.
- **Metal, all LM1B sizes (no code changes):** Mac at `30349ae`: `tools/check.sh --mlx`, five real-Metal sampler tests
  (with `lm1b-experimental`) and feature Clippy pass. Small output is byte-identical before and after the memory fix
  (`0a3ce34` vs `30349ae`, batches 1/4/16, both samplers). Medium and large Metal teacher traces pass 51/51 against
  the strict CUDA references with zero argmax and ID mismatches. The CLI generates at all three sizes. 80-seed Metal
  campaigns pass: medium perplexity +0.17 [-0.39, 0.71], entropy +0.00068 [-0.00036, 0.00171]; large +0.10
  [-0.39, 0.60], entropy +0.00087 [-0.00030, 0.00205]; Python rescoring identical, summaries replay. Metal timing passes
  median, p95 and RSS in every cell for small, medium and large ([Metal report](docs/results/2026-11-16-phase4-lm1b/medium-large-metal-v1.md)).
- **Medium/large CUDA quality and timing (no code changes):** 80-seed campaigns pass. Medium: perplexity
  +0.07 [-0.48, 0.61], entropy +0.00026 [-0.00084, 0.00137]. Large: perplexity +0.32 [-0.18, 0.81], entropy
  +0.00080 [-0.00040, 0.00199]. All 160 scores per size are `upstream_exact`, and summaries replay byte for byte.
  Timing: Rust median is 16/9/4.5% faster for medium at batches 1/4/16 and 12/6/5% faster for large at 1/4/8, with
  lower RSS and GPU memory. Pinned Python runs out of memory at large batch 16 on the 8 GiB card, so the large
  sweep uses batch 8 (the campaign batch); the failed run is retained.
  `setpriv --inh-caps=-all ./tools/check.sh --cuda` passes with the GPU idle (530 report tests, including
  `test_real_writers_and_owned_process_samples_compose`, which had failed only while the GPU was busy).
- **LM1B medium/large step 2 (CUDA parity):** `Lm1bSize` (profile, config hash, layers, heads, width) drives
  the teacher, CLI (`ncm sample-lm1b`), and CUDA whitelists (efficient attention heads 16/20, fused rotary
  12/16/20). Both CUDA backends pass 51/51 teacher-forced tensors with zero argmax and ID mismatches
  at all three sizes against matching references: the cuDNN backend against strict, and the efficient backend
  against evaluator settings (default attention dispatch, convolution TF32). The efficient path against strict fails by
  design: level-zero reconstruction equals PyTorch with convolution TF32 exactly.
  `setpriv --inh-caps=-all ./tools/check.sh --cuda` passes.
- **OWT coverage:** `ncm_core::OwtConfigOnly` validates `tests/fixtures/owt-config-v1.json` (pinned upstream
  YAML) and always refuses execution. `ncm-io` rejects OWT profiles, and `owt_shapes.rs` checks area/linear
  resampling, RoPE and masked attention at OWT's 1,025-position, 16-head shape.
- **Campaign and timing by size:** `campaign.py --size`, both cohort runners and the CUDA timing screen
  (`run_lm1b_small_capacity_screen.py --size`) accept all three sizes. A small rehearsal still reproduces the
  archive exactly, and a medium rehearsal completes every stage. The default check passes. Medium and large reuse
  the pre-registered seed lists, which are fresh for those models.
- **LM1B medium/large step 1:** the exporter takes `--size`; medium and large artifacts (manifests `2fea4a25…`,
  `86676fed…`) are deterministic across two exports, and `ncm-io` has separate per-size budgets plus a new test.
  The generalized CPU probe loads all three sizes. `setpriv --inh-caps=-all ./tools/check.sh` passes.
  Next: make the teacher read layers, heads and width from config; audit CUDA fused kernels for width
  assumptions; extend the trace tools, campaign and CLI by size.
- **Metal RSS fix (user chose "try one targeted fix"; since verified on Metal, see top entry):** host FP64
  penalties now read decoder logits through `visit_f32_rows` in blocks of two rows
  (`LM1B_PENALTY_ROWS`). On MLX each block is a separate exact GPU copy (multiply by one), so the CPU
  never touches the full logits buffer. The default (CUDA/CPU) implementation slices one host copy.
  Values and order are unchanged. Hypothesis being tested: Rust RSS growth equals the CPU-touched
  decoder logits. Checks: a new CPU row-visit test passes, and `setpriv --inh-caps=-all ./tools/check.sh --cuda`
  passes. A Metal row-visit test was added to `lm1b_categorical_fp32.rs`.
- Metal 80-seed campaign and full Metal timing ran at `0a3ce34`; `tools/check.sh --mlx` passed on the Mac
  at `0a3ce34`. The Mac candidate worktree was removed after it was confirmed identical to the commit.
- **Metal sampler change:** a short Metal screen of the old default (host FP64 draws) was 34–47%
  slower than Python MLX, because every level's logits were read to the host and the GPU was
  synced after each level. Metal now defaults to a device FP32 inverse CDF
  (`sample_seeded_device_fp32`, trait hook `lm1b_categorical_8192_fp32`) with non-finite rows
  counted on the device and checked in one read after all levels. `--host-fp64` keeps the old
  sampler. Host FP64 penalties now borrow MLX logits (`visit_f32`). Screen, not evidence (2
  warmups, 5 trials): Rust 0.101/0.185/0.432 s against Python 0.117/0.231/0.658 s at batches
  1/4/16; RSS 1,658/1,705/1,881 MiB against Python ~1,681–1,763 MiB, so batch 16 memory may still fail.
  Checks: four new real-Metal hook tests and the existing top-p tests pass, and Mac Clippy is clean.
  The refactored host path reproduces pre-change Metal outputs exactly at batches 1/4/16, and the
  CUDA rehearsal still reproduces the archive. `setpriv --inh-caps=-all ./tools/check.sh --cuda` passes.
- Metal tooling added, Metal runs pending: the campaign gains a Metal mode that imports the
  published Python CUDA captures by receipt hash and Mac-made Rust captures by pinned binary hash.
  New MLX examples `lm1b_small_cohort_campaign` and `lm1b_small_mlx_timing_probe`, Python probe
  `tools/reference/lm1b_small_mlx_timing_probe.py` (reference forward pass with on-device
  `mx.random.categorical`, as the TinyStories baseline does; the exact reference sampler is a
  pure-Python loop and not a fair speed baseline) and runner `tools/bench/run_lm1b_small_mlx_timing.py`.
  On Linux: synthetic Metal-import self-test (accepts a valid import, rejects a wrong binary hash,
  wrong receipt hash and Metal options in CUDA mode); CUDA-mode rehearsal still reproduces the
  archive exactly; `setpriv --inh-caps=-all ./tools/check.sh` passes. The MLX code is not yet compiled on Apple.
- Prospective 80-seed CUDA campaign passed; the summary replays byte for byte. No code changed.
- The 80-seed campaign tooling is now tracked in `tools/reference/lm1b_campaign/` plus a CUDA example runner.
  A rehearsal with the tracked tool on observed seeds 42/0 reproduces the archived records and comparison
  exactly. Feature Clippy and `setpriv --inh-caps=-all ./tools/check.sh` pass (533 report tests).
  [Evidence](docs/results/2026-11-16-phase4-lm1b/cohort-campaign-v1.md).
- `.claude/settings.json` (25 read-only Claude Code permission rules) is committed at the user's request. JSON validated; no code changed.

## Task routing

| Need | Load |
| --- | --- |
| Phase 2 contract / acceptance / remediation | [final dual-platform acceptance](docs/results/2026-11-10-phase2-acceptance/README.md), [docs/PHASE2.md](docs/PHASE2.md), [plan](docs/PLAN.md), [research and performance extension](docs/PHASE2-PERFORMANCE.md), [initial research/PoC](docs/results/2026-11-10-phase2-performance-research/README.md) |
| Approved additional Phase 2 optimization | [five-step plan, research and pending evidence checklist](docs/PHASE2-OPTIMIZATION-REVIEW.md) (implementation authorized; no new speedup result) |
| Shared runtime and model | [core](crates/ncm-core/src/lib.rs), [config](crates/ncm-core/src/config.rs), [model](crates/ncm-model/src/lib.rs), [blocks](crates/ncm-model/src/blocks.rs), [codec](crates/ncm-model/src/codec.rs) |
| CUDA adapter / known native fixes | [adapter](crates/ncm-backend-cuda/src/codec.rs), [vectorized resampling](crates/ncm-backend-cuda/src/resample.rs), [pool policy](crates/ncm-backend-cuda/src/runtime.rs), [operators](crates/ncm-backend-cuda/src/ops.rs), [remediation](docs/results/2026-11-10-phase2-cuda-remediation/README.md) |
| Phase 2 oracle and complete CUDA results | [oracle audit](docs/results/2026-11-10-phase2-oracle/README.md), [codec report](docs/results/2026-11-10-phase2-codec/README.md) |
| Tokenizer identities, safety and parity | [tokenizer report](docs/results/2026-11-10-phase2-tokenizers/README.md), [source lock](tools/reference/tokenizer-sources.lock.json), [canonical lock](tools/reference/tokenizers-v1.lock.json) |
| Fixtures / offline tools | [fixtures](tests/fixtures/README.md), [reference README](tools/reference/README.md) |
| Artifact trust/limits | [docs/ARTIFACTS.md](docs/ARTIFACTS.md), [ncm-io](crates/ncm-io/src/lib.rs), [runtime checks](docs/results/2026-11-10-phase2-runtime/README.md) |
| Architecture / upstream caveats | [architecture](docs/ARCHITECTURE.md), [research](docs/RESEARCH.md) |
| Checks / reporting / compatibility | [testing](docs/TESTING.md), [reporting](docs/REPORTING.md), [upstream policy](docs/UPSTREAM.md) |
| MLX codec / current parity evidence | [adapter](crates/ncm-backend-mlx/src/codec.rs), [Apple parity report](docs/results/2026-11-10-phase2-mlx-codec/README.md), [shared trained capture](tests/support/trained_codec.rs) |
| Phase 2 Linux acceptance / benchmarks | [Linux gate](docs/results/2026-11-10-phase2-linux/README.md), [summary tool](tools/bench/phase2_summary.py), [runner](tools/bench/run_phase2_linux.py) |
| Historical baselines | [results index](docs/results/README.md), [Apple Phase 1](docs/results/2026-11-10-mlx-phase1/README.md) |

## Pins, machines and caches

- Upstream NCM: `c70274273f489abc73832b2b98584a8483b6b203`.
- HF checkpoints: `hazyresearch/ncm-checkpoints` at
  `d342ce8aaea58830eef3b8ad473664673e41f692`. OWT weights remain absent.
- Candle `=0.11.1`; default workspace needs no GPU SDK. Explicit CPU diagnostic
  and CUDA paths never silently fall back. Rust floor **1.99.1** is tested;
  previous 1.88 failed Candle AVX-512 intrinsics, not all intermediate versions.
- Linux: RTX 3060 Ti 8 GiB, driver 615.78.08, CUDA 13.4. Command:
  `NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 ./tools/check.sh --cuda`.
  Default GCC 16 is not the tested CUDA host compiler.
- Linux Torch reference: `.reference/venv`, Python 3.14.9, Torch 2.15.0+cu132;
  `tools/reference/requirements-linux.lock`. Clean source: `.reference/upstream-ncm`.
- Sudoku source: `artifacts/upstream/sudoku/easy`; canonical inspected bundle on
  both machines: `artifacts/canonical/sudoku-easy-qualified`. Its name alone does
  not confer inference qualification. Original file/tensor hashes are committed
  under `docs/results/2026-11-10-sudoku-artifacts/raw/`.
- Tokenizer source: `artifacts/upstream/tokenizers-v2`; canonical:
  `artifacts/canonical/tokenizers-qualified`. Separate `.reference/tokenizers-venv`
  uses `requirements-tokenizers.lock` (Transformers 4.57.6, Tokenizers 0.22.2,
  regex 2025.9.18 / Unicode 16). Do not replace the Torch environment with it.
- Rust HF Tokenizers `=0.22.2`, only Oniguruma feature; no HTTP/remote code. Candle's
  pre-existing Tokenizers 0.23.2 remains separate. Tiny byte BPE uses locked
  fancy-regex 0.17.0 and Unicode-16 regex-syntax tables. Asset licenses are retained.
- Mac: M1 Ultra 64 GiB, macOS 26.7.3, Xcode 27.2, Rust 1.99.1. Noninteractive SSH:
  `export PATH=/opt/homebrew/bin:$PATH`. Optional Xcode Metal compiler is missing;
  do not change global Xcode. Tested project-local prebuilt native MLX route works.
- MLX bindings: mlx-rs 0.34.0 / mlx-sys 0.7.0 Git
  `adf439bf5361d8a02955de7aaf651b0c45900945`; native MLX 0.32.2, mlx-c
  `a341b4925024b88b2c593468f16e12f5e17315da`. `.reference/mlx-venv` is offline-only;
  native include/share/lib are copied into `.reference/mlx-native`.
- Mac candidate testing used a separate worktree `.reference/phase2-mlx-candidate`
  with the main checkout's native prefix and target cache. Main checkout is updated
  only by pulling pushed commits. Preserve unrelated work; existing CPU work was
  left running, builds limited to two jobs. Benchmark acceptance must assess noise.
- Mac build: `MLX_SYS_PREBUILT="$PWD/.reference/mlx-native"`,
  `RUSTFLAGS="-C link-arg=-Wl,-rpath,$MLX_SYS_PREBUILT/lib"`,
  `RUSTDOCFLAGS="$RUSTFLAGS"`, `CARGO_BUILD_JOBS=8`, then `./tools/check.sh --mlx`.
  Tested wheel needs macOS 26.2+; absolute rpath is a spike setup, not packaging.

