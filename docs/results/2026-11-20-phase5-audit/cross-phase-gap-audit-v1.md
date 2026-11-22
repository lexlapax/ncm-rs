# Cross-Phase Implementation, Quality, and Performance/Memory Gap Audit (Phases 0–5)

**Date:** 2026-11-21
**Purpose:** Honest end-to-end review across Phases 0–5 after completing Phase 5 Step 3 (`ee8769f`), identifying every remaining implementation limitation, accepted quality/numerical exception, and performance or memory cell where Rust is slower than, higher-memory than, or only narrowly ahead of Python. Use this ledger to scope any follow-up point releases (`v0.2.1` / `v0.3.0`) or future phases.

## 1. Performance & Memory vs Python Summary (Post-Step 3)

All warm-inference cells use the latest controlled dual-platform reports (3 alternating fresh processes, 5 warmups, 30 trials per process). Model-load and peak-RSS columns reflect Step 3 (`ee8769f`) improvements.

| Profile | CUDA Warm Latency (Rust vs PyTorch) | CUDA Memory (Rust vs PyTorch) | Metal Warm Latency (Rust vs Python MLX) | Metal Memory & Load (Rust vs Python MLX) | Gap / Margin Assessment |
| --- | --- | --- | --- | --- | --- |
| **Sudoku** (easy/med/hard) | Faster at all batches (b1: 25.7 vs 33.9 ms [-24%]; b128: 1,645 vs 1,744 ms [-5.7%]) | RSS 608 vs 1,725 MiB (-65%); GPU peak 1,700 vs 5,096 MiB at b128 | Faster at all batches (b1: 54.2 vs 82.2 ms [-34%]; b128: 1,542 vs 1,780 ms [-13%]) | Step 3 HWM 197 vs 403 MiB (-51%); MLX peak lower at all batches | **Clean win.** Small b128 CUDA margin (5.7%) is compute-bound in FP32 GEMM/attention. Note: Phase 3 Metal post-run MLX cache/active snapshot was slightly higher at b1/b4 (disclosed in Phase 3 report). |
| **TinyStories** | Faster at all tested batches (b1..b128) | Lower RSS (808 MiB post-Step 3) & GPU peak | Faster at all tested batches | Step 3 HWM 526 MiB (-48% vs pre-Step 3 1,016 MiB) | **Clean win** after Step 3. Historical Metal latency-tail exception from Phase 4 remains on record. |
| **XSum** (top-1, CFG 0) | Faster at b1/b4/b32 | Lower RSS (888 MiB post-Step 3) & GPU peak | Faster at b1/b4/b32 | Step 3 HWM 1,016 MiB (-36% vs pre-Step 3 1,584 MiB); load 4.10× faster | **Clean win** after Step 3 (eliminates earlier Phase 4 short-screen Metal RSS parity concern). |
| **WMT14** (top-1, CFG 0/1) | Faster at b1/b4/b16/b32 (b1: 57.9 vs 64.3 ms [-10%]; b32: 1,246 vs 1,323 ms [-5.8%]) | Lower RSS (808 MiB post-Step 3, -47% vs pre-Step 3) & GPU peak | Faster at all tested batches | Step 3 HWM 1,122 MiB (-28% vs pre-Step 3 1,561 MiB); load 3.38× faster | **Clean win.** Note: CUDA WMT14 post-reboot pool peak is default-pool scoped, not total driver peak. |
| **LM1B-small** | Faster at b1/b4/b16 | Lower RSS (796 MiB post-Step 3) & GPU peak | Faster at b1/b4/b16 (b1: 101.7 vs 116.8 ms [-13%]; b16: -28%) | **Load: 435.9 vs 508.1 ms (1.17× faster than Python!)**; **RSS: 877 vs 1,682 MiB (0.52×)** | **Clean win on every metric** including verified load time and RSS. |
| **LM1B-medium** | Faster at b1/b4/b16 (b1: -16.3%, b4: -9.1%, **b16: -4.5%**) | Lower RSS (797 MiB post-Step 3 vs PyTorch 3,936 MiB = **0.20×**) & GPU peak (2,528 vs 5,576 MiB at b16) | Faster at b1/b4/b16 (b1: -6.9%, b4: -7.2%, b16: -19.8%) | **Load: 1,018.1 vs 1,026.6 ms (1.01× faster)**; **RSS: 1,726 vs 3,366 MiB (0.51×)** | **Narrow CUDA b16 warm margin (4.5%).** Memory & load strongly favor Rust on both GPUs. |
| **LM1B-large** | Faster at b1/b4/b8 (b1: -12.2%, b4: -6.4%, **b8: -4.8%**); Python OOMs at b16 | Lower RSS (800 MiB post-Step 3 vs PyTorch 7,127 MiB = **0.11×**) & GPU peak (3,680 vs 5,154 MiB at b8) | Faster at b1/b4/b16 (b1: -4.5%, b4: -8.3%, b16: -15.6%) | **Load: 2,158.7 vs 1,993.6 ms (0.92× — 8% slower due to full 3.2 GiB SHA-256 verification)**; **RSS: 3,296 vs 6,494 MiB (0.51×)** | **Two minor gaps:** (1) CUDA b8 warm margin is 4.8%; (2) Metal load is ~165 ms slower than unverified Python MLX due to single-threaded whole-file SHA-256 over 2.9 GiB `generator.safetensors`. |

### Concrete Performance Opportunities Identified in Step 2 (Unselected in Phase 5, Ready for `v0.2.1`)

1. **CUDA LM1B per-level host syncs + unused codec greedy argmax (`crates/ncm-model/src/generator.rs`, `crates/ncm-model/src/codec.rs`):**
   - `sample_seeded_fully_device` currently downloads sampled codes to host at every level (11 device→host syncs) and `decode_latent_outputs` computes greedy argmax on `[B, 128, 30528]` even though LM1B immediately discards those greedy IDs and runs its own penalized token selector.
   - Expected gain: ~10–25 ms per batch-16 request (~1–3% at b16, ~10–15% at b1) with **byte-identical output**, widening the CUDA LM1B-medium/large warm latency lead.
2. **Metal LM1B host FP64 penalties + unused codec greedy argmax:**
   - Step 2 measured `82.3 ms` (`17.4%` of `lm1b-small` b=16 warm time) in CPU FP64 token penalties plus unused greedy argmax.
   - Expected gain: skipping unused greedy argmax is byte-identical; parallelizing or device-accelerating penalties could save 50–80 ms on Metal (requires quality re-gate if FP32).
3. **LM1B-large single-threaded whole-file SHA-256 (~165 ms behind unverified Python MLX):**
   - Per-tensor SHA-256 is already parallelized and finishes hidden behind the single-threaded whole-file SHA-256 over 2.9 GiB `generator.safetensors`. Because the artifact manifest schema v1 pins a single sequential whole-file SHA-256, beating unverified Python MLX on 3.2 GiB files without changing the manifest format would require either an opt-in `--skip-file-hash-if-all-tensors-verified` policy (only after checking that header + contiguous tensor extents cover 100% of file bytes, with header hash verified) or a v2 sharded/chunked Merkle hash manifest. Do **not** weaken integrity silently.
4. **FP16/BF16 / Quantization & Prefix K/V Caching (Future Phase):**
   - All models currently run strictly in FP32 without general prefix K/V caching (per Phase 5 scope). Enabling prefix K/V caching for XSum (512-token prefix recomputed at all 9 levels) and WMT14 (96-token prefix recomputed at all 10 levels) plus BF16/FP16 is the largest architectural speedup available (~2–4× warm inference), requiring separate quality gates.

## 2. Quality & Numerical Exceptions Across Phases 2–5

Every accepted exception remains explicitly documented and unchanged:

| Phase / Profile | Recorded Exception | Impact & Potential Fix for Point Release |
| --- | --- | --- |
| **Phase 2 (Codec, Metal)** | Immediate B4 live-memory snapshot `1.058744×` prior Rust and B4 encode MLX active peak `1.20127×` prior Rust ([Phase 2 final audit](../2026-11-11-phase2-optimization/phase2-final-audit-v1.md)). | Historical Phase 2 codec-only baseline; superseded operationally by Step 3's 45–49% whole-process RSS drop. |
| **Phase 3 (Sudoku, CUDA)** | 1 intermediate code near-tie out of 6,000 boards (medium board 85, margin `9.54e-6`), **0 final decoded-ID differences** ([Phase 3 report](../2026-11-14-phase3-sudoku/README.md)). | Floating-point tie boundary; zero user-visible output difference. |
| **Phase 4 (TinyStories, Metal)** | Metal teacher-forced logit values cross the strict CUDA numerical band (all decisions and IDs match); one earlier latency-tail failure ([TinyStories exceptions](../2026-11-15-phase4-tinystories/user-accepted-exceptions-v1.md)). | Metal FP32 kernel accumulation order differs slightly from CUDA math SDPA; decisions/IDs match. |
| **Phase 4 (XSum, Metal & CUDA)** | 15 Metal top-1 output differences out of 11,334 rows (ROUGE/PPL match) ([Metal exception](../2026-11-15-phase4-xsum/user-accepted-exception-v1.md)); CUDA stochastic bounds ([CUDA exceptions](../2026-11-15-phase4-xsum/user-accepted-exceptions-v1.md)). | Accepted under Phase 4; XSum Metal seeded + positive CFG remains explicitly rejected (`Seeded XSum Metal generation requires CFG 0`). |
| **Phase 4 (WMT14, Metal)** | Retained Metal intermediate differences on corpus retry; `mlx-retry-padded` remains an explicit experimental device option ([WMT14 qualification](../2026-11-16-phase4-wmt14/mlx-corpus-retry-v1.md)). | Qualified top-1 BLEU/chrF pass; retry path is opt-in. |
| **Phase 4 (LM1B, CUDA & Metal)** | Original 5-seed entropy gate failed; user-approved **80-seed gate passed** on CUDA and Metal for small/medium/large. Metal/CUDA stochastic draws differ from each other and from PyTorch RNG ([LM1B decisions](../2026-11-16-phase4-lm1b/user-decisions-v1.md)). | RNG stream independence is by design; statistical quality passes at 80 seeds. |
| **Phase 5 (Sudoku medium/hard exporter)** | Original historical exporter script for Sudoku medium/hard is unresolved; current exporter produces provenance-only re-exports matching all tensor weights and 2,000-board outputs ([export investigation](../2026-11-19-phase5-cli/linux-export-repro-v1.md)). | Tensor weights and outputs are byte-identical; only metadata provenance fields differ from the historical Phase 3 export. |

## 3. Implementation & Foundation Gaps (For Adding Local Models)

To make `ncm-rs` an even stronger foundation for running new local NCM models in future point releases:

1. **Profile-specific hardcoded shapes in `ncm-model`:**
   - `Codec` and `Generator` (Sudoku) are driven dynamically by `CodecConfig` and `GeneratorConfig`, whereas `TinyStoriesTeacher`, `XsumTeacher`, `Wmt14Teacher`, and `Lm1bSmallTeacher` currently pin exact config JSON / SHA-256 fixtures and hardcode layer counts/widths/vocab sizes (except `Lm1bSize` which already parameterizes layers/heads/width across small/medium/large).
   - **Follow-up (`v0.3.0`):** generalize a unified `TextGeneratorConfig` + `NcmModel` enum so user-trained or newly released local NCM checkpoints with arbitrary valid `(layers, heads, width, levels, vocab_size, head_kind)` load without adding a bespoke struct per checkpoint.
2. **LM1B rotary sidecar bundling:**
   - LM1B currently requires an external sibling `lm1b-reference-rope-v1.safetensors` file (or `--reference-rope`). Bundling `rope_cos`/`rope_sin` inside `inference.safetensors` (like `xsum-inference-v2`) in a future `lm1b-*-inference-v2` profile would make LM1B artifacts completely self-contained.
3. **XSum Metal seeded + positive CFG:**
   - Currently fails closed (`Seeded XSum Metal generation requires CFG 0`); straightforward to implement if XSum stochastic CFG on Metal is needed.
4. **OpenWebText (OWT):**
   - Config validation (`OwtConfigOnly`), no-EOS-cut policy, and synthetic operator shapes are implemented and tested; runtime execution awaits upstream checkpoint release.
5. **Serving / Multi-request concurrency / Streaming:**
   - Single-process CLI and serialized per-model `request_lock` only (by Phase 5 design). A library-level session/worker pool, cancellation, and streaming API remain deferred.
6. **Minor build-manifest lints:**
   - Default CPU-only `cargo check` emits two harmless `unused_dependencies` warnings (`ncm-core` in `ncm-backend-cuda` when no feature is active, and `ncm-model`/`ncm-backend-cuda` in `ncm-cli` when no backend feature is active). Gating or cleaning those in a small maintenance PR will keep feature-off builds warning-free.
7. **Linux CPU lifecycle observer race (`tools/bench` test-only):**
   - Intermittent `ObservationMismatch` in fake-worker supervisor tests (`linux-lifecycle-env-investigation-v1.md`) does not affect production Rust inference, only an offline benchmark supervisor test helper.

---

## 4. Resolution Ledger (Pre-`v0.2.0` Bundle, `4d970fa` / `e7b8cfa`, 2026-11-22)

The user elected to address all five actionable gap-audit items prior to tagging `v0.2.0`. All five are now implemented and verified on both Linux/CUDA (RTX 3060 Ti) and Apple Silicon/Metal (M1 Ultra); see the [updated optimization report](../2026-11-20-phase5-optimization/README.md#pre-v020-gap-audit-optimization--foundation-bundle-4d970fa--e7b8cfa-2026-11-22):

| Audit Item | Status | Verification Summary |
| --- | --- | --- |
| **1. Warm LM1B wins, SHA-256 I/O pipeline & Cargo warnings** | **Resolved** | `sampled_embedding_8192` removes 11× per-level CUDA host syncs; `TraceSink::wants_trace()` skips unused codec greedy argmax & penalty index uploads (-18 to -19 MiB CUDA VRAM peak; faster warm LM1B on CUDA & Metal); pipelined 2 MiB double-buffered whole-file SHA-256 cuts LM1B-large load to `2,102.7 ms` CUDA (2.05×) and `2,018.3 ms` Metal (within 2% of unverified Python MLX while verifying full SHA-256 + finite FP32); `cargo check --workspace --locked` emits 0 warnings. |
| **2. XSum Metal seeded + positive CFG** | **Resolved** | Added `XsumTeacher::sample_seeded_cfg_cpu` and wired `sample_xsum_mlx`. Verified on M1 Ultra (`prefix_cache_equiv_probe` 512 rows + `cfg1_seed42_id_diffs=0`; CLI `--cfg 1 --seed 42 --decode` deterministic SHA-256 `dee14e21...`). |
| **3. Self-contained LM1B v2 artifacts** | **Resolved** | Added `lm1b-{small,medium,large}-inference-v2` bundling hash-pinned `rope_cos`/`rope_sin` in `inference.safetensors` (`lm1b_small_runtime_artifact.py --reference-rope`). Verified 100% byte-identical to `v1` + sidecar in isolated directories on CUDA (`cd87171d...`) and Metal (`3f28b815...`). |
| **4. FP32 prefix K/V caching for XSum & WMT14** | **Resolved** | Added `PrefixKvCache` (`prefill_prefix_cache` + `decode_target_with_prefix_cache`), dropping cache before vocab projection. **CUDA:** WMT14 3,003/3,003 rows `0` diffs (`1.54×` warm, `0` MiB VRAM delta); XSum 1,024 rows `0` diffs (`3.24×` warm, `155.8 -> 48.1 ms`). **Metal:** WMT14 256 rows `0` diffs (`1.20×` warm, `-59 MiB` MLX peak); XSum 512 rows `0` diffs (`2.42×` warm, `204.6 -> 84.6 ms`, `-187 MiB` MLX peak). |
| **5. Config-driven local NCM foundation (`local-ncm-v1`)** | **Resolved** | Added `LocalNcmConfig` / `CodecConfig::validate_local`, `"local-ncm-v1"` artifact profile, offline `LocalTokenizer`, `LocalNcmModel` (all 3 head kinds, prefix cache, greedy + seeded sampling), unified `ncm sample` routing, and stdlib exporter `tools/reference/export_local_ncm.py`. Verified on CPU (`tests/local_ncm.rs`), CUDA, and Metal. |

**Remaining deferred items (intentionally out of scope for `v0.2.0`):** OpenWebText execution (weights unreleased upstream), lower-precision FP16/BF16/quantized inference, multi-request concurrent serving/streaming APIs, and the offline benchmark supervisor test-only race (`linux-lifecycle-env-investigation-v1.md`). All historical Phase 2–4 quality/numerical exceptions remain explicitly documented above.
