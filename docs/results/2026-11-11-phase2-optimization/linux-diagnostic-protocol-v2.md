# Fixed Linux scheduling diagnostic v2 — NOT RUN

This is a bounded investigation of the six failures in the retained **72/78**
Linux audit, not another acceptance attempt. It cannot replace that audit, close
Linux, or authorize Apple work. Execution awaits confirmation of a quiet Linux
window. No clock/governor/power/service changes or interruption of unrelated work
are authorized. The public loading API and production candidate remain unchanged.

## Question and limits

Do the B1 resident-block clusters and GPT-2/codec/Qwen tails recur while CPU/SMT
activity, frequency and system pressure are observed over the same command windows?
The existing data cannot establish their cause. In particular, sampled GPU clocks
cannot explain CPU-only tokenizer failures by themselves.

Source inspection found that the comparison runner previously left its launcher
and GPU observers unpinned. That is a **possible confound**, not evidence that these
processes ran on CPU 2/3 or caused any old failure. The new opt-in
`--observer-cpu 0` places the launcher and its GPU observer children on CPU 0;
benchmark children still receive explicit `taskset -c 2`. Invalid, unavailable,
or workload-SMT-sibling observer placements fail before output creation. The
launcher's original affinity is restored on exit. The default remains unpinned.

## Preparation checks (not benchmark evidence)

On 2026-11-11, `python -m unittest discover -s tools/bench -v` and
`CARGO_BUILD_JOBS=4 ./tools/check.sh` passed, including 48 reporting tests.
Six new tests in `tools/bench/test_linux_comparison_affinity.py` cover permitted
placement/rejections, missing/inconsistent topology, unchanged defaults,
restoration after success/failure, placement failure before children start, and
real Linux child inheritance plus workload override. The last test launches only
tiny Python children, not GPU work. Full GPU observer execution is still pending.
`python tools/bench/run_phase2_linux_comparison.py --help` exposes the new option.

Local check log: `reports/phase2-observer-preparation-check.log`, SHA-256
`6c6e1d7467887eb36a849b30a92aacf00c839d3b4b89a87adc75b74ffa1ea76d`.
These checks validate tooling only; no new performance or device-quality claim.

## Preflight — before any measurements

1. Confirm the quiet window with the user. Recheck CPU 2's SMT topology and allowed
   affinity; CPU 0 must remain outside its sibling set. Do not silently select
   another CPU if this check fails. Do not run audits, builds or debuggers during
   collection. No policy changes, process termination, or global tooling changes.
2. Check repository/worktree state. Keep accepted checkout
   `.reference/phase2-optimization-baseline` clean at `3307dff`; use its saved
   `-bin` directory. Compare both native benchmark hashes against
   `reports/phase2-optimization-controls-threeway-v1/environment.json`, and check
   the unchanged Python environments, production sources, fixtures and artifacts.
   A mismatch stops this protocol; do not rebuild opportunistically or reuse
   controls from another candidate. Any later rebuild uses an isolated target.
3. Save a machine-readable run manifest before launch: this protocol's hash,
   current commit/status/diff hash, runner and `sample_linux_cpu.py` hashes,
   both native executable hashes, fixture/artifact identities, Python source and
   package identities, exact commands, UTC time and CPU topology. Preserve it
   alongside the reports. The runner additionally records source heads, binary
   hashes, helper hash, actual launcher affinity and command intervals.
4. Require new paths below. If either exists, stop rather than overwrite or invent
   another run identifier. Preflight failures and partial captures are retained;
   no automatic retry is allowed under this protocol.

## One fixed collection

- **54 fresh workload processes:** six repeats × three runtimes × three workloads.
- Workloads in fixed order: `codec`, `gpt2`, `qwen3`. Collect their complete reports,
  including all codec batches/scopes and all checked loads, not just failing cells.
- Unchanged five warmups, thirty trials, fixtures, precision and output retention.
- Existing three cyclic runtime orders repeat twice. This balances each runtime's
  position but does **not** balance every predecessor pair or eliminate carryover.
  Do not claim this design separately identifies runtime-order causality.
- CPU 2 for workloads; CPU 0 for the launcher, GPU observers and CPU sampler.
- One 900-second, one-second-interval CPU capture. Start it first, confirm its
  metadata/first sample, then launch the comparison. Retain the full capture,
  including ambient intervals; wait for its completion. There is no early stop
  based on favorable timing, nor an extension based on an unfavorable result.

Commands (not yet executed; CPU sampler runs concurrently with comparison):

```sh
python tools/bench/sample_linux_cpu.py \
  reports/phase2-linux-diagnostic-v2-cpu.jsonl \
  --cpu 2 --observer-cpu 0 --seconds 900 --interval 1

python tools/bench/run_phase2_linux_comparison.py \
  .reference/phase2-optimization-baseline \
  .reference/phase2-optimization-baseline-bin \
  reports/phase2-linux-diagnostic-v2 \
  --cpu 2 --observer-cpu 0 --repeats 6 --workloads codec gpt2 qwen3
```

Keep launcher/stdout/stderr logs and all telemetry. Failure, interruption, fewer
than 54 completed commands, a missing completion record, or CPU telemetry that
does not bracket **every** command means the diagnostic is incomplete. Do not
silently retry, exclude a process, shorten the matrix, or prolong the sampler.
The 900 seconds is an observation bound, not permission to kill unrelated work;
if collection outlasts it, retain the evidence and report the coverage failure.

## Analysis and stop decision

After collection ends, verify all command/report hashes and identities. Use
`summarize_codec_load_control.summarize(root, 'cuda', include_execution=True)` and
`summarize_tokenizer_control.summarize(root, profile, include_load=True)` for both
profiles. These validate full outputs and preserve whole-process units. Retain
all scopes, pooled summaries, process medians/p95s and existing descriptive
10,000-resample intervals (seed 20261110); do not turn the 54 processes into an
acceptance gate or count copied Python reports as independent observations.

For **every** command, tabulate its UTC interval and overlapping CPU telemetry
intervals, CPU 2/3 activity, driver-reported frequency, pressure and sampling
lateness. Retain GPU telemetry and inspect B1 block wall/event spans. Include
quiet-looking and busy-looking intervals alike. Flag boundary overlap, missing
counters and insufficient sampling resolution rather than attributing a whole
one-second interval to an individual request. Do not select/filter trials using
these observations. Aggregate activity and sampled frequency are not request CPU
work or per-request clocks; temporal correlation alone does not establish cause.

Report exactly one of: a testable mechanism supported by new observations; no
recurrence without an established mechanism; unresolved recurrence; or incomplete
collection. Any outcome leaves the existing **72/78** acceptance result intact.
A future intervention or acceptance protocol needs a separate predeclared
rationale; no repeat-until-green loop, tolerance change or transplant of these
controls is permitted. If no defensible next intervention exists, stop with the
observations, limitations and specific user input needed. Apple remains blocked.
