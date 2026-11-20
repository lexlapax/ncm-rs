# v0.1.0 release notes — draft, not released

This is a working draft for the [Phase 5 release candidate](PHASE5.md#step-1--release-candidate-v010-no-numerical-changes).
The tag and GitHub release do not yet exist. Linux/CUDA and Metal release gates, reproducible
fresh-clone setup, and the remaining decisions in the [context map](../agent-context-map.md)
must be completed before requesting approval to tag and publish. Historical Phase 3/4 evidence
below is not a measurement of every current CLI path.

## Intended scope

- FP32 greedy Sudoku (easy, medium and hard); TinyStories sampling; XSum and WMT14 top-1;
  LM1B small, medium and large sampling. Explicit CUDA or Apple Silicon/MLX device selection;
  no device fallback. See the [profile matrix](../README.md#profile-and-evidence-matrix).
- A unified `ncm sample` surface, retaining profile-specific commands. `--decode` uses each
  artifact's pinned tokenizer. XSum and WMT14 accept `--text`; TinyStories accepts text encoding
  to exactly one non-padding token. LM1B text-input behavior remains unresolved; it accepts only
  `[101]` ID rows and uses a separately generated hash-pinned rotary sidecar.
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
  `mlx-retry-padded` is experimental.
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

- Complete and verify `docs/USAGE.md` from a fresh clone on Linux, then on Apple Silicon, including
  pinned downloads, exporters, manifest checks and examples. Sudoku medium/hard's original exporter
  identity remains unresolved ([investigation](results/2026-11-19-phase5-cli/linux-export-repro-v1.md)).
- Resolve LM1B `--text` behavior; finish command equivalence, text/decode and performance
  evidence on CUDA and then Metal. Record failures and do not infer a GPU pass from CPU tests.
- Update these notes and the context map with actual gate evidence. Ask before tagging and pushing
  `v0.1.0` and publishing the source-only GitHub release.
