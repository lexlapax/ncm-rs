# v0.1.0 release notes — draft, not released

This is a working draft for the [Phase 5 release candidate](PHASE5.md#step-1--release-candidate-v010-no-numerical-changes).
No local or remote `v0.1.0` tag was observed, and GitHub's release lookup returned not found on
2026-11-20. Linux/CUDA and Metal release gates, complete reproducible setup, and the remaining
decisions in the [context map](../agent-context-map.md) must be completed before requesting approval
to tag and publish. Historical Phase 3/4 evidence below is not a measurement of every current CLI path.

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
  `mlx-retry-padded` is experimental. The boundary for other CFG scales is [awaiting a decision](results/2026-11-19-phase5-cli/wmt14-cfg-policy-audit-v1.md); it is not a blanket quality claim.
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

## Pending before publication

- The Linux guide blocks were replayed from a fresh clone at `c63b205` ([evidence](results/2026-11-19-phase5-cli/linux-guide-single-commit-v3.md)); complete the Mac recipe and fresh-clone replay after the Linux gate. Sudoku medium/hard's original exporter identity remains unresolved and the approved re-exports need full qualification ([investigation](results/2026-11-19-phase5-cli/linux-export-repro-v1.md)).
- Finish CUDA command/option, quality and performance qualification, including the WMT14 CFG decision; then validate the current CLI, text/decode and performance on Metal. LM1B empty-text has a scoped [CUDA screen](results/2026-11-19-phase5-cli/cuda-lm1b-empty-text-v1.md#all-three-sizes), not a Metal pass. Record failures and do not infer a GPU pass from CPU tests.
- Update these notes and the context map with actual gate evidence. Ask before tagging and pushing
  `v0.1.0` and publishing the source-only GitHub release.
