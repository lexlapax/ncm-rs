# Phase 4 resume handoff

Last updated: 2026-11-18 UTC. Start by reading `AGENTS.md` and
`agent-context-map.md`, inspecting the current worktree, and checking the
linked Phase 4 contracts and evidence. This handoff summarizes state; the
source, recorded gates, and current machine state remain authoritative.

## Objective and constraints

Finish Phase 4, preserving functionality and all approved numerical, quality,
performance, and hardware gates. Follow the approved profile order and Linux
before Metal for each profile. The user asked to optimize the work without
weakening gates. Do not change pinned upstream, artifacts, schemas, oracle,
thresholds, or held-out protocol silently. Preserve failed evidence and do not
claim a phase or runtime gate from a bounded diagnostic. The user authorized
pushes needed for Mac work via pulls and approved both Mac and Linux runs.
Before every commit include an updated
`agent-context-map.md`, run appropriate checks, and inspect the staged diff.

## State at handoff

- The user approved [one-time XSum Metal advancement](docs/results/2026-11-15-phase4-xsum/user-accepted-exception-v1.md)
  with the nine intermediate codes and six post-EOS raw IDs listed in
  the retained comparison. LM1B-small is now the active profile.
- Preserve the unrelated untracked `.pi/`. Check `git status` for the
  current Linux head. The pre-softmax baseline is `8ba381f`.
- The XSum CUDA masked-softmax change passes `tools/check.sh --cuda`
  (524 report and data tests, 18 device cases). All 11,334 top-1 rows
  and all 1,280 previously observed seeded rows match retained CUDA codes
  and raw IDs exactly. The repeated comparison against pinned Python
  lowers median latency by approximately 19% and also lowers p95, RSS,
  and sampled GPU memory at batches 1/4/32. Against the rebuilt previous
  Rust binary on that same fixture the paired median reductions are
  1.70%, 1.76% and 1.62%, with identical sampled GPU peaks. The completed
  reports and reproduction commands are in
  [the XSum CUDA results](docs/results/2026-11-15-phase4-xsum/cuda-combined-numerics-v1.md).
- XSum Metal's exact full-corpus top-1 gate fails. On retained batch 04928,
  row 4937 differs by three codes and five raw IDs from pinned Python CUDA.
  Three newly tested six-part-head combinations produced outputs byte-identical
  to the earlier candidate on batches 04864 and 04928; they do not fix it.
  Details, earlier interventions, hashes, and limitations are in
  [the XSum diagnostic record](docs/results/2026-11-15-phase4-xsum/level1-cross-backend-diagnostic-plan-v1.md).
  Smaller Metal batches resolve row 4883's single ID mismatch through size 8
  but do not resolve row 4937. Split-scale attention, an ordered latent
  reduction and a combined normalization/projection/attention probe also
  did not resolve it. A pinned tokenizer and ROUGE replay matches Python CUDA
  on all 11,334 retained Metal decoded texts and per-row scores. The pinned
  perplexity tokenizer inputs and score masks match as well. An
  [independent pinned GPT-2 scoring run](docs/results/2026-11-15-phase4-xsum/cuda-combined-numerics-v1.md#full-corpus-metal-text-and-rouge-replay)
  on the retained Metal outputs matches all 277,595 scoreable tokens and
  the exact retained Python perplexity of `82.62335629362389`. The nine
  intermediate code differences and six post-EOS raw-ID differences
  still fail the unchanged exact gate.
  The [pinned Python batch-size control](docs/results/2026-11-15-phase4-xsum/mlx-full-top1-v1.md#pinned-python-free-generation-across-batch-sizes)
  reproduced the retained rows at batch 32. Running the same Python model
  at sizes 1–16 reproduces Metal row 4937 and sizes 1–2 also reproduce
  Metal row 10015. Both batch-32 input projections restore row 4937 in
  the CUDA batch-16 replay. The projection replay and the same-hidden
  head replay on Metal retain the failure.
  Combining the split head with captured CUDA hidden state restores the
  exact 32-row Metal batch in the adapted model. Three captured CUDA block
  outputs also work with that head, but other, later states retain the
  mismatch. The [joint replays](docs/results/2026-11-15-phase4-xsum/mlx-full-top1-v1.md#joint-metal-head-and-transformer-replays)
  depend on injected CUDA state for a known row.
  Avoid repeating these variations without new evidence.
- TinyStories advanced under [the user's scoped exceptions](docs/results/2026-11-15-phase4-tinystories/user-accepted-exceptions-v1.md).
  Keep those exceptions visible and retain the preceding regressions.
- WMT14 top-1 passes the scoped CUDA and Metal profile gates. The
  [CUDA results](docs/results/2026-11-16-phase4-wmt14/cuda-fast-argmax-plan-v1.md)
  cover the full pinned corpus and paired performance. The
  [Metal qualification](docs/results/2026-11-16-phase4-wmt14/mlx-corpus-retry-v1.md#cli-and-profile-qualification)
  covers both full modes, all seven timing cells, the exact CLI controls,
  and both final device suites. Metal row 632 retains 13 intermediate code
  differences with all raw IDs matching. The bounded retry recovers one
  batch in each mode. Guided batch four uses a final transformer checkpoint
  after block three. Both real-device regressions pass 18 operator cases.
- LM1B-small CUDA's frozen held-out quality comparison fails: the entropy
  confidence interval lower endpoint is -0.007118 versus -0.0046. Its five
  held-out seeds are observed; a rerun is a retest, not fresh held-out data.
  The Python self-control also fails but does not waive the candidate failure.
  See [results](docs/results/2026-11-16-phase4-lm1b/small-heldout-cuda-v1.md)
  and [frozen protocol](docs/results/2026-11-16-phase4-lm1b/heldout-protocol-v1.md).
  LM1B medium and large have [checkpoint inventories](docs/results/2026-11-16-phase4-lm1b/inventory-preparation-v1.md)
  but no qualified GPU runtime. OWT remains limited to the synthetic and
  configuration scope in the [Phase 4 contract](docs/PHASE4-TEXT.md),
  which records the missing released checkpoint.
- Linux validation scheduling was improved without removing tests: 524 report
  tests and 18 CUDA device cases passed. The observed report suite time fell
  from 167.35 to about 146.48 seconds in a single comparison. See
  [validation evidence](docs/results/2026-11-17-phase4-check-scheduling.md).
- The Linux `llama-server.service` was stopped with the user's approval for
  CUDA testing and should remain stopped until the user asks to restart it.
  Verify its current state before relying on this historical observation.
- Mac `dotsetup.rmlx` was temporarily stopped for the last bounded Metal
  screens with explicit approval and restored; launchd and its server process
  were verified running afterward. An initial asynchronous shutdown made the
  immediate restore fail; the service was recovered before a successful retry.
  This approval covered that screen and restoration; recheck availability and
  obtain any needed authorization before interrupting live services again.
  The Mac checkout fast-forwarded to `0b89cd1` for the
  [guidance CLI verification](docs/results/2026-11-15-phase4-xsum/mlx-cfg-probe-v1.md#cli-verification).
  Full Metal checks passed. Experimental Metal top-1 guidance through
  `sample-xsum` matched pinned outputs at scales 0/1/2 on retained
  fixtures. The eight-row scale-one cold screen showed 9.6% lower Rust
  peak RSS and a 3.4% higher median wall time than adapted Python MLX.
  The service was left running.
  The earlier six-part screens remain in the Mac's ignored
  `reports/phase4-xsum-six-refined-screen-v1/`. Inspect the checkout
  before any future fast-forward pull.
- The prior thread's goal status was reported as blocked after waiting for the
  Mac service decision; the user subsequently approved the bounded screen and
  it was completed. Do not mistake that status for Phase 4 completion.

## Next actions

1. Continue LM1B-small on Linux CUDA. The
   [repeated performance run](docs/results/2026-11-16-phase4-lm1b/small-cuda-capacity-screen-v1.md#repeated-cuda-timing-and-allocator-measurements)
   favors Rust in median, p95 and process RSS at all three measured sizes.
   Resolve the frozen entropy interval failure. The replay reproduced the
   original result byte for byte, with an interval of
   `[-0.007118, 0.002290]` against the frozen `±0.0046` margin.
   The [fixed-code and precision audit](docs/results/2026-11-16-phase4-lm1b/small-heldout-cuda-v1.md#fixed-code-cuda-decode-replay)
   isolates cross-decoder differences to Python's convolution TF32
   setting. Both common-decoder intervals remain below the bound.
   The original precision and scorer receipts remain the acceptance baseline.
   The [Rust cuDNN experiment](docs/results/2026-11-16-phase4-lm1b/small-heldout-cuda-v1.md#rust-cudnn-default-math-experiment)
   matches the original Python fixed-code outputs exactly. Its five
   observed-seed retest passes perplexity and still misses the entropy
   lower bound with `[-0.005636, 0.001367]`.
   The [controlled precision comparison](docs/results/2026-11-16-phase4-lm1b/small-controlled-cuda-v1.md#controlled-generation-at-evaluator-convolution-precision)
   moves the first differing code to level 5 and traces the first latent
   difference to level-zero residual convolution. The legacy cuDNN API
   reproduces Rust on the same input. The pinned evaluator uses the graph
   API. Compare that residual convolution with a bounded graph-API probe
   before changing runtime behavior.
2. Complete the remaining LM1B-small gates before medium and large in the
   approved order. Any new acceptance protocol needs explicit review.
3. For each changed runtime, run the applicable full checks and reproducible
   quality, performance, and memory comparisons on both real target devices.
   Update the context map, commit, and use the existing push authorization.

Linux and Mac ignored `.reference/`, `artifacts/`, `reports/`, and build trees
contain local inputs and raw evidence; they are not Git deliverables. Do not
overwrite them or assume they exist on another machine. Never disassemble or
use restricted vendor implementations; the existing record documents an
earlier inadvertent inspection and its cleanup.
