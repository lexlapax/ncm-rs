# v0.1.0 release notes

Published on 2026-11-20 at commit `de2975a` (`https://github.com/lexlapax/ncm-rs/releases/tag/v0.1.0`, source-only) for [Phase 5 step 1](PHASE5.md#step-1--release-candidate-v010-no-numerical-changes)
(lean FP32 inference release, no numerical changes). All Step 1 CLI, text/decode, feature-folding,
and fresh-clone setup checks passed on both Linux/CUDA and Apple Silicon/MLX.

## Intended scope

- FP32 greedy Sudoku (easy, medium and hard); TinyStories sampling; XSum and WMT14 top-1;
  LM1B small, medium and large sampling. Explicit CUDA or Apple Silicon/MLX device selection;
  no device fallback. See the [profile matrix](../README.md#profile-and-evidence-matrix).
- A unified `ncm sample` surface, retaining profile-specific commands. `--decode` uses each
  artifact's pinned tokenizer. XSum and WMT14 accept `--text`; TinyStories accepts text encoding
  to exactly one non-padding token. LM1B accepts only empty text strings, mapped to `[101]` ID
  rows; nonempty prompts are unsupported. It uses a separately generated hash-pinned rotary sidecar.
- Source-only release: no checkpoints, evaluation data or prebuilt binaries. CUDA builds depend
  on local cuDNN and compute capability; MLX builds depend on matching native MLX libraries.
  Python remains an offline export/reference dependency, not a production inference dependency.

## Accepted exceptions and limits

- Sudoku CUDA had one intermediate near-tie code difference and no final-ID disagreement in
  the tested 6,000 boards ([report](results/2026-11-14-phase3-sudoku/README.md)). The prior
  Phase 2 Apple immediate-memory exception is documented in its
  [audit](results/2026-11-11-phase2-optimization/phase2-final-audit-v1.md).
- TinyStories' scoped Metal numerical-band crossings and historical latency-tail failure remain
  recorded ([decision](results/2026-11-15-phase4-tinystories/user-accepted-exceptions-v1.md)).
- XSum Metal's 15 enumerated code/raw-ID differences, with identical decoded texts and quality
  scores, remain an [accepted exception](results/2026-11-15-phase4-xsum/user-accepted-exception-v1.md).
  CUDA stochastic sampling has separate [accepted evidence and limitations](results/2026-11-15-phase4-xsum/user-accepted-exceptions-v1.md).
  Positive-CFG and stochastic XSum are experimental, not a general quality claim. The legacy
  `sample-xsum` alias now requires `--experimental-positive-cfg-xsum` for positive positional CFG
  and `--experimental-stochastic-xsum` for a positional seed; old unflagged experimental calls
  must add the corresponding flags. Qualified top-1 calls retain their positional syntax.
- WMT14's scoped top-1 qualification retains Metal intermediate differences and explicit
  bounded-retry behavior ([report](results/2026-11-16-phase4-wmt14/mlx-corpus-retry-v1.md#cli-and-profile-qualification));
  `mlx-retry-padded` is experimental. WMT14 CFG 0 and 1 remain unflagged; other positive scales require `--experimental-cfg-wmt14` on both commands. Existing unflagged non-0/1 calls must add that flag; arbitrary scales have no blanket quality claim ([audit](results/2026-11-19-phase5-cli/wmt14-cfg-policy-audit-v1.md)).
- LM1B's original five-seed entropy gate failed; the user-approved 80-seed protocol passes on both
  devices. Metal draw-level and floating-policy differences are a scoped exception; fixed-history
  decision/ID differences remain failures ([decisions](results/2026-11-16-phase4-lm1b/user-decisions-v1.md)).
  Metal `--top-p-fp32` remains experimental. Neither runtime reproduces the PyTorch RNG stream.
- FP32 only; reduced precision and quantization are deferred. OpenWebText inference is unsupported
  because weights are unreleased. No concurrent serving, streaming or cancellation is qualified.
- On the tested 8 GiB CUDA card the pinned Python LM1B-large baseline OOMed at batch 16; the
  comparison used batch 8 and Rust large batch 16 was not measured
  ([report](results/2026-11-16-phase4-lm1b/medium-large-cuda-timing-v1.md)). Model, batch, desktop
  activity and allocator scope affect capacity; accepted CLI batch limits are not memory guarantees.

## Step 1 verification summary

- **Linux/CUDA:** nine-profile baseline/alias/unified byte-identical routing ([routing](results/2026-11-19-phase5-cli/cuda-current-routing-v1.md)),
  WMT14 CFG boundary ([boundary](results/2026-11-19-phase5-cli/cuda-wmt14-cfg-boundary-v1.md)),
  full-corpus text encoding/generation and retained-output decoding ([text](results/2026-11-19-phase5-cli/cuda-text-input-v1.md), [decode](results/2026-11-19-phase5-cli/cuda-decode-screen-v1.md)),
  nonblind quality retests across all released profiles, post-reboot WMT14 model timing with default-pool counters ([timing](results/2026-11-19-phase5-cli/cuda-wmt14-performance-retest-v1.md)),
  and a single-commit fresh-clone Linux setup replay ([Linux guide replay](results/2026-11-19-phase5-cli/linux-guide-single-commit-v3.md)).
- **Apple Silicon/MLX:** `./tools/check.sh --mlx` (18 operator cases, 493 macOS report tests, MLX clippy/test/doc),
  nine-profile baseline/alias/unified byte-identical routing with `--features mlx` only, invalid-device
  rejections, XSum/WMT14 experimental flag boundaries, `mlx-retry-padded`, LM1B sibling rotary sidecar
  and `--top-p-fp32`, all six macOS corpus/boundary encode+decode tests, end-to-end Metal `--text`/`--decode`
  equivalence ([Metal step 1](results/2026-11-19-phase5-cli/mlx-step1-cli-v1.md)), and a fresh-clone macOS
  setup replay ([macOS guide replay](results/2026-11-19-phase5-cli/macos-guide-fresh-clone-v1.md)).
- **Publication gate:** ask before tagging `v0.1.0`, pushing the tag, and publishing the source-only GitHub release.
