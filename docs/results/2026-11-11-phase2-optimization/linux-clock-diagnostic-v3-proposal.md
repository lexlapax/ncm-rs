# Linux clock diagnostic v3 — proposal, NOT APPROVED / NOT RUN

This is an optional follow-up to [diagnostic v2](linux-diagnostic-v2-results.md),
not an acceptance attempt. **Linux remains 72/78; Apple remains pending.**
No clock change, new benchmark, or permission test has been performed for this
proposal. Automatic continuation is not approval. The earlier “yes. go” authorized
the completed quiet-window v2 run, not this system-policy intervention.

## Hypothesis and scope

V2 retained approximately 0.177/0.198ms B1 block clusters in both native runtimes,
including CUDA-event spans, despite negligible observed SMT-sibling activity.
GPU command windows contained varying SM clocks, but one-second samples cannot
identify the clock during a block. A supported constant graphics-clock request
could discriminate whether the same spread persists under that intervention.
It is not established necessary or sufficient, and cannot explain CPU-only tails.

Use the unchanged full codec harness, not a new microbenchmark: **18 processes**,
six cyclic repetitions of accepted Rust/candidate Rust/Python, all B1/B2/B4 scopes,
checked loads, five warmups and thirty trials. Tokenizers are deliberately absent:
this experiment targets the recurring GPU block spread and cannot resolve their
acceptance cells. Report all scopes, not just B1. No production/dependency/fixture/
precision/retention changes or recompilation. Use the frozen v2 binary identities.

This is a single-condition diagnostic compared descriptively with historical v2,
**not** a contemporaneous randomized clock-policy A/B test. Neither collapsed
spread nor its persistence alone proves an exclusive cause. Cyclic runtime orders
balance positions, not all carryover pairs. No new acceptance threshold is created.

## Required authorization and launch blockers

Before preparing an executable launch:

1. Obtain explicit approval to request **1710 MHz graphics clocks on this GPU**
   and reset graphics clocks to **defaults** afterward. The read-only capability
   query lists 1710 MHz, but that does not prove lock permission or enforcement.
2. Establish that resetting to defaults is appropriate. `--reset-gpu-clocks` is
   not a general restoration of an unknown custom lock range. Current sampled
   clocks and deprecated application-clock fields do not establish the original
   requested policy. Require user confirmation of default policy; stop if another
   controller/custom policy is present or the original state is uncertain.
3. Obtain a new quiet 15-minute window and confirm no competing GPU-clock controller.
   Do not stop unrelated processes/services or change desktop configuration.
4. Implement and test a process-local cleanup supervisor **before any mutation**.
   This proposal is not a ready-to-run shell sequence. Tests must use fake commands,
   never a real lock: success, lock failure, comparator failure, timeout, interrupt,
   parent disappearance and cleanup failure. Record exact tests/results.
5. Verify noninteractive permission for both setting and resetting without changing
   credentials, sudo configuration, services, persistence mode or global tooling.
   If privileges/restoration cannot be arranged safely, stop and ask the user.

No memory-clock, power-limit, governor, voltage, fan, application-clock or boost-
policy changes are included. Never use GPU/device reset. The RTX 3060 Ti's known
PCI address is `00000000:01:00.0`; rebind address/UUID/name before launch rather than
assuming enumeration index zero still refers to it.

## Fixed collection and cleanup contract

After approval and successful supervisor tests, freeze a machine-readable manifest
with this protocol and supervisor hashes, user authorization, UTC time, device
identity, pre-change read-only policy output, repository/status/source/fixture/
payload/Python identities and exact subprocess commands. Recheck v2 native hashes,
accepted checkout cleanliness, CPU 2/3 topology and CPU 0 eligibility. Any mismatch
stops before mutation; do not opportunistically rebuild. Require fresh paths:

- `reports/phase2-linux-clock-diagnostic-v3/` (comparator output)
- `reports/phase2-linux-clock-diagnostic-v3-cpu.jsonl`
- `reports/phase2-linux-clock-diagnostic-v3-prelaunch.json`

The future supervisor must perform this fixed sequence:

1. Start CPU telemetry for exactly **900 seconds**, one-second intervals, workload
   CPU 2 / observer CPU 0; verify metadata and the first sample. Pin diagnostic
   launcher, GPU monitors and cleanup observer to CPU 0. No concurrent analysis,
   builds or debuggers during collection.
2. Arm cleanup **before attempting** the clock request. The cleanup obligation
   starts even if the set command reports failure, since partial application is
   possible. Record command output/return codes, with bounded command timeouts.
3. Request `nvidia-smi -i <verified-device> --lock-gpu-clocks=1710,1710` once.
   On failure, do not benchmark: perform cleanup, preserve evidence and stop.
4. Run the existing comparison runner once with `--cpu 2 --observer-cpu 0
   --repeats 6 --workloads codec`, the accepted checkout and saved baseline binaries.
   No replacement process, exclusion, extension or measurement retry.
5. Request `nvidia-smi -i <verified-device> --reset-gpu-clocks` immediately after
   comparator completion/failure, or at a **600-second lock deadline**, whichever
   is first. The supervisor must stop/reap only its own diagnostic children before
   deadline cleanup; interrupted reports remain incomplete. It must not terminate
   unrelated work. Parent disappearance must also trigger the armed cleanup.
6. If cleanup fails, make at most one bounded cleanup-only retry, retain both
   results and notify the user immediately if reset remains unconfirmed. Do not
   run more benchmarks or report restoration as successful. A supervisor cannot
   guarantee cleanup through host failure; disclose that limitation.
7. Record post-reset read-only state and the reset command result. Do not claim
   arbitrary prior-policy restoration, or infer it merely from a low idle clock.
   Wait for the CPU sampler's full 900 seconds; no favorable-result early stop.

The supervisor implementation must resolve races between normal completion and
watchdog cleanup and verify no child/lock remains active. Its tests and actual
cleanup evidence are launch/delivery requirements, not assumed capabilities.

## Analysis and stop conditions

Verify all 18 commands, report hashes/identities and outputs. Preserve copied Python
reports without counting them as independent processes. Run the existing full-codec
summarizer with `include_execution=True` and retain all per-process median/p95 and
CUDA-event spans, alongside every command's CPU/GPU telemetry and boundary flags.
Use the unchanged 10,000 paired whole-process resamples/seed 20261110 descriptively;
no transplant into acceptance. Retain all timings, including outliers and first-call
costs. Report B1 wall/event process ranges and max/min spread for each native
runtime, including comparison with all six v2 processes.

Report requested versus observed clocks, available clock-event reasons, power and
temperature. Successful set return code alone is not proof of fixed clocks. Missing
counters, observed deviations, insufficient one-second resolution, thermal/power
limits or incomplete telemetry constrain the interpretation; do not discard their
associated timing samples. Memory clocks remain dynamic and are a residual factor.

Outcome must be one of: incomplete/unenforced intervention; spread persists under
the observed intervention; or reduced spread without established exclusive cause.
Report cleanup separately, including any uncertainty. **Every outcome leaves 72/78
unchanged.** No further sampling until favorable. Define any subsequent acceptance
protocol separately, covering all 78 cells and both comparators with unchanged
limits. If no defensible next step exists, stop with evidence, limitations and the
specific user input needed. A separately specified read-only investigation remains
an alternative if clock-policy changes are declined.
