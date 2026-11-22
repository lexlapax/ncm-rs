# Test Strategy and Acceptance Gates

Status: **Phases 0–5 complete on both Linux/CUDA (RTX 3060 Ti) and Apple Silicon/Metal (M1 Ultra).**
All nine released FP32 profiles have passed tokenizer parity, teacher-forced numerical parity, end-to-end official quality gates (with explicitly documented Phase 2–4 user-accepted exceptions), 9-profile CLI byte-equivalence (`baseline == alias == sample`), `--text`/`--decode` parity, fresh-clone setup replays, and Step 3 alternating load/RSS benchmarks. See the [Phase 4 audit](results/2026-11-19-phase4-audit/phase4-final-audit-v1.md), [Phase 5 audit](results/2026-11-20-phase5-audit/phase5-final-audit-v1.md), and [cross-phase gap audit](results/2026-11-20-phase5-audit/cross-phase-gap-audit-v1.md).

## Layers of Evidence

| Layer | Coverage | Where It Runs |
| --- | --- | --- |
| **Config & artifact (`ncm-core`, `ncm-io`)** | Strict bounds, unknown/duplicate fields, profile/task matching, byte budgets, whole-file & per-tensor SHA-256, streaming & copied loaders, nonfinite FP32 rejection, symlink/`O_NOFOLLOW` rejection, OWT refusal | CPU (`./tools/check.sh`) |
| **Tokenizers (`ncm-tokenizers`, `ncm-cli`)** | Sudoku, TinyStories Byte BPE, GPT-2 (XSum), Qwen3 (WMT14), BERT (LM1B): Unicode, special tokens, padding/truncation, exact corpus encode/decode against pinned references | CPU unit tests + artifact-bound corpus tests on Linux & macOS |
| **Operators & model preflight** | Norms, split-half RoPE, bidirectional/prefix masked attention, conv/transposed conv, exact `area1d`/`linear1d`, stable tie-breaking selection, categorical sampling, FP64 penalties, zero GPU allocation on preflight failure | CPU fixtures every check; 18 real-GPU operator cases via `--cuda` / `--mlx` |
| **Teacher-forced parity** | Layer/level logits, codes, reconstructions, and final IDs against pinned Python CUDA references (`TraceSink`) | Both GPUs per profile |
| **CLI routing & equivalence** | `tools/bench/phase5_cli_equivalence.py` (9 profiles byte-identical across baseline, alias, unified `ncm sample`), `phase5_cli_invalid_devices.py`, experimental flag gates, `--text`/`--decode` | Both GPUs on CLI or loader changes |
| **Official quality protocols** | Sudoku (6,000 boards), XSum (11,334 rows, ROUGE-1/2/L + PPL), WMT14 (3,003 rows, sacreBLEU + chrF), TinyStories (5 seeds, GPT-2-large PPL + entropy), LM1B (approved 80-seed gate, GPT-2-large PPL + entropy) | Both GPUs for any output-affecting change or release |
| **Performance & memory** | Alternating fresh-process comparisons (≥ 3 fresh processes per side, warmups + pooled trials, median, p95, host peak RSS/HWM, native GPU/MLX counters) per [docs/REPORTING.md](REPORTING.md) | Both GPUs (Linux/CUDA first, then macOS/MLX) |
| **Fresh-clone setup** | Replaying `docs/USAGE.md` commands from a clean clone on Linux and macOS | Per release / setup change |

## Numerical & Correctness Policies

- **Elementwise comparison:** `abs(actual - reference) <= atol + rtol * abs(reference)` with separate finite-value and mask-position checks. Starting operator budget is FP32 `atol=1e-5, rtol=1e-4`; end-to-end tolerances are calibrated per profile and never silently widened to hide regressions.
- **FP32 baseline:** production inference is strictly FP32 with TF32 disabled where configurable. Masks contain `-inf` intentionally (`attn_mask_for_masking` is bit-checked explicitly in `ncm-model`, while all other F32 weights must be finite).
- **Resampling & convolutions:** `area1d` matches PyTorch `floor(i * in / out)..ceil((i+1) * in / out)` bin ranges; `linear1d` matches `align_corners=False` half-pixel mapping; `ConvTranspose1d` is tested with asymmetric kernels.
- **Sampling & ties:** `select_last` returns the first index on exact ties. Categorical samplers use deterministic SplitMix64 streams (independent of PyTorch's RNG; stochastic quality is gated statistically via fixed seed cohorts). Final XSum and LM1B sequential penalties use FP64 accumulation.
- **No silent fallback:** unavailable devices, unsupported profiles (OWT), unflagged experimental options (`--experimental-positive-cfg-xsum`, `--experimental-stochastic-xsum`, `--experimental-cfg-wmt14`, `--top-p-fp32`, `mlx-retry-padded`), and malformed inputs fail closed with exit code 1.

## Local Validation Commands

GitHub build CI is intentionally disabled; the metadata-only weekly upstream watcher remains enabled. Run local checks before committing:

```sh
# Default CPU suite (Rust 1.99.1+, fmt, clippy, workspace tests, docs, Python report tests):
./tools/check.sh

# Linux / CUDA real-device suite (RTX 3060 Ti tested configuration):
NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 ./tools/check.sh --cuda

# Apple Silicon / Metal real-device suite (requires MLX_SYS_PREBUILT):
export MLX_SYS_PREBUILT="$PWD/.reference/mlx-native"
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$MLX_SYS_PREBUILT/lib"
export RUSTDOCFLAGS="$RUSTFLAGS"
./tools/check.sh --mlx
```

- Always validate **Linux/CUDA first, then macOS/MLX** on real target hardware; CPU checks never prove GPU correctness.
- Use platform-specific features (`--features cuda-cudnn` or `--features cuda` on Linux; `--features mlx` on macOS) rather than `--all-features`.
- Checks never download checkpoints or run SSH automatically.
