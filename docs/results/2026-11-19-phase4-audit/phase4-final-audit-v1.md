# Phase 4 final requirement-to-evidence audit

**Result: complete with explicitly accepted, scoped exceptions.** Every released text profile in the
[Phase 4 plan](../../PLAN.md#phase-4--released-text-profile-compatibility) has the gate's evidence on both
GPUs (RTX 3060 Ti CUDA and M1 Ultra Metal), or the user has accepted a recorded, scoped exception. OWT has the
config, synthetic and operator coverage the plan asks for while no checkpoint is released. Pinned upstream code
`c7027427…`, checkpoint revision `d342ce8a…`, artifacts, oracles, numerical tolerances and quality margins are
unchanged. Every earlier failure stays on record.

The gate: each profile gets tokenizer parity, teacher-forced numerical parity, end-to-end output checks and a
published quality/performance report on both GPUs, using the official quality protocols.

## Per-profile disposition

| Profile | Tokenizer | Teacher-forced parity | End-to-end and CLI | Quality | Performance (Rust vs Python) | Disposition |
| --- | --- | --- | --- | --- | --- | --- |
| TinyStories | Byte BPE parity ([tokenizers](../2026-11-10-phase2-tokenizers/README.md)) | CUDA [teacher probe](../2026-11-15-phase4-tinystories/cuda-teacher-probe-v1.md) and [numerical investigation](../2026-11-15-phase4-tinystories/cuda-numerical-investigation-v1.md); Metal decisions and IDs match, with values outside the band | CLI codes and IDs match the scored files on CUDA and Metal ([contract](../../PHASE4-TEXT.md#tinystories-cli-candidate)) | Official five seeds pass on CUDA ([held-out](../2026-11-15-phase4-tinystories/heldout-quality-v1.md)); Metal under the accepted exceptions | Faster with less memory on [CUDA](../2026-11-15-phase4-tinystories/cuda-strict-performance-v1.md) and [Metal](../2026-11-15-phase4-tinystories/mlx-generator-qk-rope-portable-python-v2.md) | Complete with [accepted exceptions](../2026-11-15-phase4-tinystories/user-accepted-exceptions-v1.md): Metal numerical band crossings and an earlier latency-tail failure |
| XSum | GPT-2 parity ([tokenizers](../2026-11-10-phase2-tokenizers/README.md)) | CUDA top-1 matches all Python output IDs ([ties](../2026-11-15-phase4-xsum/cuda-reference-ties-v1.md)) | Top-1 CLI on CUDA and Metal, including CFG fixtures | Full and held-out ROUGE match on CUDA; Metal texts, ROUGE and perplexity match | Faster with less memory on [CUDA](../2026-11-15-phase4-xsum/cuda-reference-tie-repeated-v1.md) and [Metal](../2026-11-15-phase4-xsum/mlx-borrowed-upload-repeated-v1.md); an earlier short Metal screen with higher Rust RSS at batches 1 and 4 is retained | Complete with accepted exceptions: [Metal's 15 output differences](../2026-11-15-phase4-xsum/user-accepted-exception-v1.md) and [CUDA stochastic bounds](../2026-11-15-phase4-xsum/user-accepted-exceptions-v1.md) |
| WMT14 | Qwen parity ([tokenizers](../2026-11-10-phase2-tokenizers/README.md)) | [CUDA teacher reference](../2026-11-16-phase4-wmt14/teacher-cuda-reference-v1.md) | CLI controls on both devices, with an explicit bounded Metal retry | Frozen BLEU/chrF checks pass on CUDA and Metal | Lower median, p95, RSS and allocator peak in every cell ([qualification](../2026-11-16-phase4-wmt14/mlx-corpus-retry-v1.md#cli-and-profile-qualification)) | Scoped top-1 qualification; retained Metal intermediate differences are recorded |
| LM1B-small | BERT decode parity with the pinned evaluator ([decode](../2026-11-16-phase4-lm1b/bert-decode-development-v1.md)) | CUDA traces pass ([Rust](../2026-11-16-phase4-lm1b/small-teacher-cuda-rust-v1.md)); Metal decisions and IDs match | `sample-lm1b` on CUDA and Metal, with `--decode` | 80-seed gate passes on [CUDA and Metal](../2026-11-16-phase4-lm1b/cohort-campaign-v1.md) | Faster with less memory on [CUDA](../2026-11-16-phase4-lm1b/small-cuda-capacity-screen-v1.md#repeated-cuda-timing-and-allocator-measurements) and [Metal](../2026-11-16-phase4-lm1b/small-metal-timing-v1.md#after-the-memory-fix-all-checks-pass) | Complete under [user decisions 1–2](../2026-11-16-phase4-lm1b/user-decisions-v1.md) |
| LM1B-medium, large | Same tokenizer bytes as small (identical `tokenizer.json`, manifest and license in all three artifacts) | 51/51 tensors with zero argmax and ID mismatches on [CUDA](../2026-11-16-phase4-lm1b/medium-large-cuda-parity-v1.md) and [Metal](../2026-11-16-phase4-lm1b/medium-large-metal-v1.md#teacher-forced-parity) | CLI generation on both devices | 80-seed gate passes on [CUDA](../2026-11-16-phase4-lm1b/cohort-campaign-v1.md#prospective-cuda-results-medium-and-large) and [Metal](../2026-11-16-phase4-lm1b/cohort-campaign-v1.md#prospective-metal-results-medium-and-large) | Faster with less memory on [CUDA](../2026-11-16-phase4-lm1b/medium-large-cuda-timing-v1.md) and [Metal](../2026-11-16-phase4-lm1b/medium-large-metal-v1.md#latency-and-memory-against-python-mlx) | Complete to the lean bar of [user decision 3](../2026-11-16-phase4-lm1b/user-decisions-v1.md#3-minimum-bar-for-lm1b-medium-and-lm1b-large) |
| OWT | — | — | — | — | — | Config, synthetic and operator coverage only, as planned ([coverage](../2026-11-19-phase4-owt/config-operator-coverage-v1.md)). Execution is refused, and the no-EOS-cut policy is validated. Weights are unreleased |

## Cross-cutting requirements

- **Artifacts and metadata:** each profile has a non-executable exported artifact with pinned manifest, tensor
  hashes and per-profile memory budgets. OWT artifacts are rejected. Sudoku profiles, limits and goldens are unchanged.
- **Unsupported behavior fails explicitly:** devices never fall back to each other, unsupported sampling options are
  rejected, and OWT inference always errors.
- **Order:** within each profile, Linux/CUDA evidence came first, then Metal.
- **Official protocols:** ROUGE (XSum), BLEU/chrF (WMT14), and generative perplexity with entropy from the pinned
  GPT-2-large evaluator (TinyStories, LM1B). LM1B uses the user-approved 80-seed protocol; the failed five-seed
  comparison stays recorded.
- **Disclosed limits:** desktop activity, telemetry and allocator-scope differences, and OOMs are disclosed: the
  original scorer at batch 32, and pinned Python at LM1B-large batch 16 on the 8 GiB card. Concurrent-request
  throughput was not measured. Rust Metal does not reproduce PyTorch's random stream, which the contract does not require.

## Checks at closure

- Linux at `30349ae`: `setpriv --inh-caps=-all ./tools/check.sh --cuda` passes (530 report tests, 18 CUDA operator
  cases). The default check passes at the closure commit.
- Mac at `30349ae`: `tools/check.sh --mlx` passes (18 Metal operator cases), as do the five LM1B real-Metal sampler
  tests and feature Clippy.

## What this does not claim

There is no OWT inference or quality, no positive-CFG XSum quality, no draw-for-draw Metal and CUDA equivalence,
no concurrent-throughput figure and no inference release. The `--top-p-fp32` Metal option remains experimental.
Phase 5 (optimization and inference release) is not started by this audit.
