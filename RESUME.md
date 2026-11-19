# Phase 4 handoff

Last updated: 2026-11-19 UTC. The objective is to **complete Phase 4**, not just
one diagnostic. Start with `AGENTS.md`, `agent-context-map.md`, `git status`,
[the Phase 4 plan](docs/PLAN.md#phase-4--released-text-profile-compatibility)
and [the text-profile contract](docs/PHASE4-TEXT.md). Current source and
recorded evidence take precedence over this summary.

## User direction and working rules

Focus narrowly on Phase 4 feature correctness and completeness before
optimization. The user directed the work to the Mac, authorized SSH and all
Mac/Linux runs, and approved pushes needed for Mac testing. Do not reboot the
Mac or interrupt its existing `dotsetup.rmlx` service. Inspect the clean remote
checkout before a fast-forward pull; preserve unrelated work. The local
untracked `.pi/` is unrelated and must be preserved. Before each commit update
`agent-context-map.md`, run appropriate checks and review the staged diff.

Keep pinned upstream, artifacts, oracle, numerical tolerances and the frozen
quality protocol unchanged unless separately authorized. A previously observed
seed is a retest, not fresh held-out evidence. Do not claim the phase is
complete or hide failed results. The earlier 80-seed proposal is a proposed
protocol, not an approved replacement gate.

## Where Phase 4 stands

- TinyStories advanced under the user's [scoped exceptions](docs/results/2026-11-15-phase4-tinystories/user-accepted-exceptions-v1.md).
- XSum advanced under the user's [scoped Metal exception](docs/results/2026-11-15-phase4-xsum/user-accepted-exception-v1.md); retain the documented nine intermediate-code and six post-EOS raw-ID discrepancies. The separate [CUDA stochastic exception](docs/results/2026-11-15-phase4-xsum/user-accepted-exceptions-v1.md) also remains recorded.
- WMT14 top-1 has scoped CUDA and Metal qualification, including documented
  differences and retries; see [the Metal qualification](docs/results/2026-11-16-phase4-wmt14/mlx-corpus-retry-v1.md#cli-and-profile-qualification).
- LM1B-small is the active profile. CUDA's frozen five-seed held-out entropy
  interval fails; a Python-versus-Python control also fails, which indicates
  the test may be noisy but does not waive the candidate failure. See the
  [CUDA quality evidence](docs/results/2026-11-16-phase4-lm1b/small-heldout-cuda-v1.md)
  and [frozen protocol](docs/results/2026-11-16-phase4-lm1b/heldout-protocol-v1.md).
- LM1B-small Metal has fixed-history and greedy evidence but does not pass the
  controlled stochastic, numerical or quality gates. A separately qualified
  same-device reference is still needed. The earlier five-observed-seed Metal
  retests fail the entropy interval. See [Metal quality and convolution](docs/results/2026-11-16-phase4-lm1b/small-metal-observed-quality-v1.md)
  and [sampler evidence](docs/results/2026-11-16-phase4-lm1b/small-controlled-cuda-v1.md).
- LM1B-medium and large have checkpoint inventories but no qualified GPU
  runtimes. OpenWebText currently requires synthetic/config/operator coverage;
  checkpoint and quality work waits for released, validated weights.

## Latest LM1B-small Metal work

`sample-lm1b-small ... mlx ... --top-p-fp32` is an explicit experimental
sampler. It does not reproduce PyTorch RNG or establish CUDA parity; the default
Metal and CUDA samplers remain unchanged. The multirow candidate now uses a
CUDA-shaped FP32 prefix reduction. On 133 retained, deliberately chosen
boundary draws it differs from CUDA in 66 cases versus 82 for the prior MLX
scan. This targeted result is not a normal-use error rate or a gate pass.
The Rust probability output matches the independent MLX candidate bit for bit
on those rows. A deterministic two-row cutoff fixture agrees with CUDA; a
separate single-row test remains ignored because CUDA uses a different scan and
Metal fails it. A single-row comparison on 133 retained rows found the proposed
grouped scan worse than the existing MLX scan, so single-row behavior is not
changed.

The candidate's observed-seed quality retest still fails the entropy interval:
`[-0.00576046, 0.00316554]` versus the frozen `±0.0046` margin. Perplexity and
both point checks pass. This is an observed-seed retest, not a new held-out
pass. The candidate's two-row cold CLI timing screen and the unchanged prior
sampler are documented in the [controlled evidence](docs/results/2026-11-16-phase4-lm1b/small-controlled-cuda-v1.md).
Do not claim full performance qualification from this small screen.

## Next work, in order

1. Continue LM1B-small correctness: investigate the remaining Metal/CUDA
   sampler and convolution differences, especially the separate single-row
   CUDA scan. Use fixed-input, same-draw comparisons and qualify a same-device
   reference before drawing wider conclusions. Keep unsupported paths explicit.
2. Resolve the frozen CUDA and Metal quality failures with valid evidence.
   If a different protocol or exception is needed, present the concrete
   proposal and existing failed results; do not silently change the gate.
3. Complete LM1B-small end-to-end, quality and reproducible latency/memory
   qualification on both devices, including preceding regressions.
4. Implement and qualify LM1B-medium and then large with explicit memory
   budgets, correctness traces, output/quality checks and performance reports
   on CUDA and Metal. Keep unavailable or unqualified profiles explicit.
5. Audit Phase 4 requirements and existing accepted exceptions against current
   source and evidence, finish the required documentation review, and only then
   mark Phase 4 complete.

The ignored local `reports/` directories contain detailed captures, scripts and
logs; they are not portable Git artifacts. Use the linked tracked evidence for
identities and conclusions, verify local files before reuse, and reconstruct
missing reports from recorded procedures rather than assuming they exist.
