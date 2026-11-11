# Fixed Linux diagnostic v2 — unresolved recurrence

**Investigation complete; expanded Phase 2 is not complete. Linux remains 72/78.**
The user confirmed the quiet window (“yes. go”). The single collection prescribed
by [protocol v2](linux-diagnostic-protocol-v2.md) ran on 2026-11-11. No retry,
sample exclusion, clock/governor/service change, unrelated-process termination,
production edit, or Apple measurement occurred. These controls are **not** inputs
to the acceptance audit.

## Collection and verification

- Prelaunch manifest saved at **17:01:55.922494 UTC**, before telemetry or workload
  launch. HEAD/origin were `fa040ed`; accepted checkout was clean at `3307dff`;
  tracked candidate state was clean (only `.pi/` untracked).
- Frozen accepted/candidate binaries matched the earlier 360-process controls;
  production sources, fixtures, manifest and payload hashes, clean upstream pin,
  Python versions and package lists matched. The launch manifest records exact
  commands, hashes, CPU topology, worktree state and protocol identity. Payload
  digest checks are not tensor-layout/finite checks or output qualification.
- **54/54** fresh processes completed: six repetitions × previous/candidate/Python
  × codec/GPT-2/Qwen. All complete scopes, five warmups and thirty trials retained.
  Workloads ran on CPU 2; launcher and GPU observers inherited CPU 0 affinity.
  CPU 2/3 remain SMT siblings. Six cyclic orders balance positions, not every
  predecessor/carryover pair.
- Commands ran **17:01:56.413501–17:06:47 UTC**. The sampler ran the full
  **17:01:55.959550–17:16:55.959652 UTC**, with **901 samples**, a completion record
  and 900.000667 seconds actual duration. Both launch return codes were zero.
  No audit/build/debugger ran during collection; analysis began after completion.
- Analysis verifies every exact command/order/cwd, report digest, source identity,
  copied Python report, warmup/trial count and complete output qualification.
  Native codec outputs pass the existing complete-report validator against Python;
  both tokenizer profiles have exact matching output hashes in every scope.
  This is benchmark-output qualification, **not** a new 642-trace capture.
- Every command is bracketed by CPU telemetry. `commands.tsv` tabulates all 54
  windows, overlapping sample indices, boundary flags, activity/frequency/lateness
  and report hashes. `analysis.json` additionally retains each overlapping interval,
  system pressure/counters, GPU samples and **all process/scope median/p95 values**.
  `summaries.json` covers all 13 codec and 4 scopes per tokenizer, both comparators,
  using the existing 10,000 whole-process bootstrap resamples, seed 20261110.
  Shared Python copies are not independent measurements.

## The six previously failing cells

These are descriptive diagnostic ratios against accepted Rust, **not new gate
results**. Full native/Python summaries and every other scope are archived.

| Cell | Median ratio | p95 ratio | Relevant 95% interval upper |
| --- | ---: | ---: | ---: |
| B1 resident decoder block | 1.10076 | 1.01313 | median **1.12231** |
| GPT-2 encode | 0.97905 | 0.96607 | p95 0.98741 |
| GPT-2 decode | 0.99906 | 0.96421 | p95 1.04395 |
| GPT-2 decode, skip specials | 1.01204 | 0.97724 | p95 1.02759 |
| Codec checked load | 0.99897 | 0.99963 | p95 1.01170 |
| Qwen checked load | 0.98474 | 0.99564 | p95 1.01116 |

The five CPU/load-tail failures did not recur under these observations. That does
not establish their old cause, waive the retained failures, or permit transplanting
these results into the acceptance matrix. The B1 block's approximately 0.177/0.198ms
clusters recur in **both** native implementations, including CUDA-event spans:

| Repeat | Accepted wall median ms | Candidate wall median ms | Accepted event median ms | Candidate event median ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.197452 | 0.177653 | 0.193536 | 0.174080 |
| 2 | 0.196904 | 0.198512 | 0.192512 | 0.194560 |
| 3 | 0.177657 | 0.179009 | 0.174080 | 0.175104 |
| 4 | 0.196223 | 0.198751 | 0.192000 | 0.194560 |
| 5 | 0.176795 | 0.198779 | 0.173056 | 0.194560 |
| 6 | 0.175890 | 0.198527 | 0.172032 | 0.194560 |

Process-cluster occupancy differs (accepted three fast/three slow; candidate two
fast/four slow). Neither the favorable trials nor matching-source arguments justify
excluding a process. Event spans include device-stream scheduling/launch gaps;
they do not prove pure kernel execution or a particular clock cause.

## Telemetry interpretation

Across the 292 **unique** one-second intervals overlapping commands, aggregate
CPU activity median was 6.006% (range 3.202–7.846%). CPU 2's median was 100%, including
the workload itself. CPU 3's median was zero and maximum 0.9901%; the recurring
block clusters therefore do not require sustained observed sibling contention in
this run. This does not retroactively diagnose earlier runs.

CPU 2 used `intel_pstate`/`powersave`; driver-reported frequency varied. Per-command
ranges, including boundary intervals and low samples, are retained rather than
filtered. No pressure/frequency counters were missing in command-overlap intervals.
Sampler lateness median/max was **0.558/3.584ms** over all 901 samples. One-second
read intervals cannot attribute activity/frequency to a sub-millisecond request.
System pressure is not request CPU work and low aggregate load is not proof of
zero contention.

GPU samples show changing SM clocks (including 1710 and 1950–1980MHz during codec
commands), but both timing clusters occur in windows with overlapping sampled
clock ranges. Command-level telemetry does **not** identify the clock during the
B1 block. `nvidia-smi` timestamps use local `US/Pacific` (alias
`America/Los_Angeles`, UTC−08:00 on this date), checked via `/etc/localtime`; analysis
converts them to UTC and retains the original CSV. No per-request clock or causal
claim is made.

## Disposition and next decision

The predeclared outcome is **unresolved recurrence**. Isolating observers and
confirming the quiet window did not eliminate B1 block bimodality. No production
regression or environmental cause is established by these data.

A separate, bounded GPU-clock intervention is a plausible next discriminating
experiment: hold a supported clock constant for all runtimes and test whether
both native timing clusters persist. This is a **hypothesis**, not proof that
locking is necessary/sufficient, nor an explanation for CPU-only tails. It needs
explicit user approval for the temporary system-policy change, capability/original-
state inspection, an exact predeclared protocol and reliable restoration before
execution. No clock change is approved by the quiet-window confirmation and none
has been performed. A new quiet window would also be needed. If clock control is
not approved/available, use a separately specified read-only, finer-grained GPU
investigation rather than repeating this collection until favorable.

Any future acceptance attempt needs a separate evidence-backed protocol retaining
both comparators, all 78 cells, unchanged bands and negative evidence. Apple work
and final goal completion remain blocked on Linux acceptance.

## Reproduction and checks

[Evidence manifest](linux-diagnostic-v2-evidence.json) binds the compact archive,
all members and analysis sources. The archive contains complete raw reports,
telemetry, prelaunch manifest, the frozen **pre-run** protocol, local launch/analysis
scripts, summaries and logs; no executables, weights or large trace tensors.
The launch script is historical evidence: **do not rerun it**.

From a clean checkout, extract into a new colonless working directory:

```sh
mkdir -p reports
tar -xJf docs/results/2026-11-11-phase2-optimization/raw/linux-diagnostic-v2.tar.xz -C reports
python reports/phase2-linux-diagnostic-v2-analyze.py
python -m unittest discover -s tools/bench -v
```

Analysis reproduces `analysis.json`, `summaries.json` and `commands.tsv` exactly
without GPUs, executables, weights or Python model packages. All **48 reporting
tests** pass, and `CARGO_BUILD_JOBS=4 ./tools/check.sh` passes. An initial offline analyzer edit shadowed its imported `activity`
function and raised `UnboundLocalError`; the import was renamed before successful
analysis/reproduction. No measurements were repeated or altered for this tooling
error. This evidence-only change does not claim a new CUDA/Metal runtime check.
