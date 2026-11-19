# Phase 4 handoff

Last updated: 2026-11-19 UTC. Phase 4 is now complete (see below); this handoff is kept for its history. Start with `AGENTS.md`, `agent-context-map.md`, `git status`,
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
- LM1B-medium and large pass the lean bar on CUDA: teacher-forced parity on both
  backends, 80-seed quality and timing against Python
  ([quality](docs/results/2026-11-16-phase4-lm1b/cohort-campaign-v1.md#prospective-cuda-results-medium-and-large),
  [timing](docs/results/2026-11-16-phase4-lm1b/medium-large-cuda-timing-v1.md)), and on Metal
  ([report](docs/results/2026-11-16-phase4-lm1b/medium-large-metal-v1.md)). LM1B-small Metal timing now passes
  RSS as well ([memory fix](docs/results/2026-11-16-phase4-lm1b/small-metal-timing-v1.md#after-the-memory-fix-all-checks-pass)).
  OpenWebText has config-only and synthetic operator coverage
  ([coverage](docs/results/2026-11-19-phase4-owt/config-operator-coverage-v1.md)); checkpoint and
  quality work waits for released, validated weights.

## Latest LM1B-small Metal work

The latest fixed-history Metal check uses development seed 11 and the pinned
rotary tables. The independent Python reference matches all draws at batch 15.
At batch 16 the composed path differs on 18 draws and matching native affine
layer normalization reduces this to 13. At batch 32 the affine reference
differs on 20 draws. The Python decoder matches the Rust final IDs on the
forced histories at all three sizes. Replaying the Rust fused Q/K RoPE kernel
in Python eliminates the sampled-choice differences at batches 16 and 32.
This isolates the kernel's numerical effect. The independent composed
reference's differences remain.

The batch-16 trace comparison is complete. With the same Rust code history,
pinned Python CUDA differs from Rust Metal in 465 values across 19 of 46
floating tensors under the existing tolerance. The first failing tensor is
the level-zero reconstruction. Generator and decoder argmaxes and all
2,048 penalty-adjusted token IDs agree. Replaying the same draws against
the recorded logits selects different codes at 30 of 10,240 positions
after level zero. The independent Python MLX affine trace meets the float
tolerance on all 46 tensors but retains its 13 draw differences. The RoPE
replay matches 45 tensors bit for bit and every sampled choice. See the
[large-batch results](docs/results/2026-11-16-phase4-lm1b/small-controlled-cuda-v1.md#large-batch-fixed-history-comparison)
for commands, identities and matching histories.

Phase 4 remains at LM1B-small. On 2026-11-18 the user stopped the
cross-device draw investigation, approved the 80-seed quality protocol and set
a lean bar for medium and large. See [the decisions](docs/results/2026-11-16-phase4-lm1b/user-decisions-v1.md).
The run outputs above are in `reports/phase4-lm1b-metal-large-batch-v1/` on
Mac and Linux.

The checked-in Python MLX development reference matches all 8,333 codes and
1,664 final IDs on four Mac requests. Q/K RMS normalization
aligned all 47 shared generator tensors on the two-row trace. Go to the
[reference command and evidence](docs/results/2026-11-16-phase4-lm1b/small-controlled-cuda-v1.md#metal-development-reference-for-complete-requests).

The single-row CUDA scan is reproducible on captured CUDA probabilities.
The remaining same-logit differences involve softmax exponentials, and
substituting the CUDA-shaped scan alone increases the targeted draw
differences on MLX probabilities. See [the stage replay](docs/results/2026-11-16-phase4-lm1b/small-controlled-cuda-v1.md#single-row-cuda-scan-and-softmax-replay).

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

Phase 4 is complete with scoped, user-accepted exceptions
([audit](docs/results/2026-11-19-phase4-audit/phase4-final-audit-v1.md)). Phase 5 has not
started; ask the user before beginning it.

The ignored local `reports/` directories contain detailed captures, scripts and
logs; they are not portable Git artifacts. Use the linked tracked evidence for
identities and conclusions, verify local files before reuse, and reconstruct
missing reports from recorded procedures rather than assuming they exist.
