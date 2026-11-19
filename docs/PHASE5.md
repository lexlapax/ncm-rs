# Phase 5 plan: lean inference release, then targeted optimization

User decisions, 2026-11-19: **release first, kept lean**, and **FP32 only**. Reduced precision (BF16/FP16)
and quantization are deferred to a later phase with their own quality gates. This narrows the broader
[Phase 5 outline](PLAN.md#phase-5--optimization-and-inference-release). Its gate still applies: the full
regression and quality suite passes, there is no unreviewed quality regression, and published before/after
reports show real gains. Within every step, Linux/CUDA comes first, then macOS/MLX.

## Starting point

Phase 4 closed with every released profile qualified on both GPUs, or under recorded exceptions
([audit](results/2026-11-19-phase4-audit/phase4-final-audit-v1.md)). Rust already beats pinned Python on
every profile, but some margins are thin: 4–16% median latency on CUDA for LM1B, and 0.15–1.2% process RSS
on Metal. The CLI has a separate `sample-*` command per profile. Three of them still need `*-experimental`
build features, and only LM1B can decode text. There is no release tag, and the README still describes Phase 3.

## Step 1 — release candidate v0.1.0 (no numerical changes)

1. **One CLI surface.**
   - `ncm sample ARTIFACT DEVICE INPUT [--seed N] [--cfg S] [--decode]` dispatches on the artifact manifest's
     profile. The existing `sample-*` commands remain as aliases.
   - Fold `wmt14-experimental`, `lm1b-experimental` and `lm1b-mlx-experimental` into the ordinary `cuda`,
     `cuda-cudnn` and `mlx` builds.
   - Options that are not qualified stay available only as explicitly named runtime flags, marked
     experimental in help and docs: `--top-p-fp32`, stochastic XSum, positive-CFG XSum, `mlx-retry-padded`.
   - Unsupported profile, option and device combinations fail with a clear error; OWT is refused.
   - **Gate:** for fixed seeds and inputs, every profile on each device produces byte-identical output through
     the new command, the aliases and the pre-change binary.
2. **Text in and out.**
   - `--decode` works for every text profile, using the artifact's pinned tokenizer.
   - `--text` input tokenizes, pads and truncates using each profile's pinned policy: XSum 512-token
     prefixes and WMT14 96-token sources.
   - **Gates:**
     - decoded text equals the pinned evaluator's decoding on retained outputs;
     - encoded and padded IDs equal every row of the converted corpora (XSum 11,334, WMT14 3,003);
     - `--text` and the equivalent ID input give identical generations.
3. **Reproducible setup documentation** (`docs/USAGE.md`).
   - Build per GPU: toolchain, cuDNN, compute capability, and the MLX native prebuilt.
   - Fetch the pinned checkpoints, export each artifact with its exporter, and check the expected manifest
     SHA-256.
   - Then run examples. Verify the document by following it from a fresh clone on each machine (the temporary
     clone is deleted afterwards).
4. **README and release notes.**
   - The supported matrix:
     - greedy Sudoku;
     - TinyStories;
     - XSum and WMT14 at top-1;
     - LM1B small, medium and large.
   - Links to the quality and performance evidence.
   - Every accepted exception and limitation:
     - Metal draw-level differences;
     - no PyTorch random-stream reproduction;
     - FP32 only;
     - OWT unsupported;
     - no concurrent serving;
     - the 8 GiB card's batch limits.
5. **Tag `v0.1.0`.** Publish a GitHub release with notes and source only. No weights. No prebuilt binaries:
   CUDA builds depend on the local cuDNN and compute capability, and MLX on the native prebuilt. Ask before
   tagging and pushing.

## Step 2 — profile real shapes (CUDA, then Metal)

For each profile, at batch 1 and at its largest measured batch, break a request into stages:
- model load;
- per-level generator time;
- codec decode;
- sampling and penalties;
- host readback.

Record process RSS and device peak at each stage. Publish a ranked bottleneck table in
`docs/results/<date>-phase5-profile/`. The user then picks one or two targets per GPU.

Candidates already visible in Phase 4 evidence:
- Rust model load is slower than Python on both GPUs; LM1B-small on Metal takes 1.1 s, against 0.5 s.
- Metal's RSS lead is small.
- CUDA LM1B's lead at the largest batch is 4–5%.

## Step 3 — targeted optimizations (chosen targets only)

- **Hypothesis:** each change states one and is measured with the [reporting protocol](REPORTING.md): three
  alternating fresh processes, 5 warmups, 30 trials, median, p95, process RSS and native device peak.
- **Output-identical changes:** pass the step 1 equivalence fixtures.
- **Output-changing changes:** rerun that profile's frozen quality gate on both devices:
  - LM1B's 80-seed campaign;
  - XSum and WMT14 corpus checks;
  - TinyStories' five seeds.
- **Caching changes:** pass cache-on/off equivalence.
- **No measured benefit:** the change is reverted, with the negative result recorded.

## Step 4 — close

Write a Phase 5 audit mapping each requirement to its evidence. If optimizations landed, tag `v0.2.0` with
before/after reports. Otherwise `v0.1.0` stands as the release.

## Out of scope for this phase

- BF16/FP16 and quantization.
- A general prefix K/V cache. The existing WMT14 cache stays as is.
- CUDA graphs, MLX compilation and new fused kernels, unless step 2 points to them; they would then go
  through step 3.
- A serving API, streaming, cancellation and multi-request isolation.
- OWT inference.
- Training.
- New upstream pins. The weekly upstream watcher continues.
