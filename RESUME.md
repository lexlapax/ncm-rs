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
pushes needed for Mac work via pulls. Before every commit include an updated
`agent-context-map.md`, run appropriate checks, and inspect the staged diff.

## State at handoff

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
  perplexity tokenizer inputs and score masks match as well. The nine
  intermediate code differences and six post-EOS raw-ID differences
  still fail the unchanged exact gate.
  Avoid repeating these variations without new evidence.
- TinyStories advanced under [the user's scoped exceptions](docs/results/2026-11-15-phase4-tinystories/user-accepted-exceptions-v1.md).
  Keep those exceptions visible and retain the preceding regressions.
- WMT14's CUDA [reference retest](docs/results/2026-11-16-phase4-wmt14/cuda-reference-candidate-retest-v1.md)
  matched the 3,003-row corpus and passed the unchanged scorer. The
  [repeated batch-16 CFG performance comparison](docs/results/2026-11-16-phase4-wmt14/cuda-uncached-cfg-eight-v1.md)
  still has a 1.58% slower Rust median. The original CFG failure remains
  recorded. Complete performance, memory and real Metal qualification in
  profile order.
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

1. Resolve the exact XSum Metal gate using the retained evidence. The
   shortened-batch and combined-arithmetic probes did not resolve it.
   Use a new evidence-based hypothesis or the pending explicit scope
   decision before advancing. Verify Linux first and run Metal only when
   the Mac is available.
2. Resolve the separate LM1B-small frozen quality failure without silently
   changing its protocol, and complete the remaining profile gates in the
   approved order. If a protocol decision is genuinely necessary, present the
   retained evidence and ask the user rather than declaring a failure passed.
3. For each changed runtime, run the applicable full checks and reproducible
   quality, performance, and memory comparisons on both real target devices.
   Update the context map, commit, and use the existing push authorization.

Linux and Mac ignored `.reference/`, `artifacts/`, `reports/`, and build trees
contain local inputs and raw evidence; they are not Git deliverables. Do not
overwrite them or assume they exist on another machine. Never disassemble or
use restricted vendor implementations; the existing record documents an
earlier inadvertent inspection and its cleanup.
