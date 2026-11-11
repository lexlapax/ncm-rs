# Read-only Linux trace investigation — proposal, NOT RUN

Alternative to the optional [clock intervention](linux-clock-diagnostic-v3-proposal.md).
No GPU-clock, power, governor, service, credential or global-tooling changes.
Preparation does not authorize collection or establish a quiet window. Linux
remains **72/78**; this is not acceptance and does not unblock Apple.

## Question and evidence gap

Can an instrumented B1 resident-block timing difference be localized to kernel
execution versus allocation/launch/synchronization intervals? The
[source inspection](completion-audit.md#offline-resident-block-timing-inspection)
and [pool inspection](linux-diagnostic-v2-results.md#offline-pool-counter-and-retained-trace-follow-up)
do not answer this. Event spans include host-submission gaps; aggregate pool
counters do not measure allocation latency. One-second GPU telemetry cannot
identify per-block clocks. No result here would establish an exclusive clock or
CPU cause, or resolve the tokenizer/load acceptance cells.

The existing CUPTI text includes host API intervals/correlation IDs and kernel
intervals/device/context/stream IDs. Offline inspection of the three retained
historical `codec-diagnostic-block4{,-b,-retained}-cupti.txt` captures found exactly
36 elapsed-time query windows each, with two ordered `cuEventRecord` calls before
each query. DRIVER interval durations match end minus start, and no dropped-record
message occurs. This only supports feasibility of boundary recognition on those
old B4 captures. It does not validate a parser, prove current B1 attribution, or
permit transferring their allocation diagnosis to v2.

## Fixed proposed collection

Use the unchanged frozen v2 accepted/candidate **codec_bench** executables, their
matching worktrees, pinned artifact and `codec-bench-v1.json`. Run the complete
harness without `--profile-block4`; no recompile or workload changes. Trace via the
existing local CUPTI injection library only in the instrumented child's environment.
No inherited injection in controls, launcher or observers. No Python benchmark is
needed for this native-path diagnostic; both comparators remain mandatory for a
later, separate acceptance attempt.

Exactly **12 fresh processes**: three per implementation per condition. Let AP/CP
mean accepted/candidate plain, and AT/CT mean accepted/candidate traced. Fixed order:

1. AP, CP, AT, CT
2. CP, AT, CT, AP
3. AT, CT, AP, CP

This is a bounded three-repeat cyclic design, not complete position/carryover
balance. Do not adapt order or process count to observed timings. Retain every
scope, first call, five warmups and thirty trials provided by the existing harness.
All full report outputs must qualify using the existing validators and pinned
qualified reference outputs. Instrumented results are diagnostic only, even if
faster. Plain controls estimate instrumentation effects for this collection; do
not replace the retained acceptance controls or v2 processes with them.

Workloads CPU2, launcher/observers CPU0; recheck CPU2/3 SMT topology and permissions.
Use the existing read-only GPU telemetry and a full 900-second CPU capture. No
concurrent builds, analysis or debuggers. Bound the entire workload sequence to
600 seconds, each child to 120 seconds, each raw trace to 1 GiB and total raw
capture to 6 GiB; require at least 12 GiB free before launch. These are abort
limits, not promised workload sizes or timings. A runner must enforce them while
streaming, stop/reap only owned children, preserve partial files and never retry a
measurement. The sampler still completes 900 seconds unless interrupted or unsafe.
A timeout/size limit is an incomplete investigation, not grounds to raise limits
and rerun within this protocol.

Fresh paths: `reports/phase2-linux-readonly-trace-v1/`, sibling
`phase2-linux-readonly-trace-v1-prelaunch.json` and
`phase2-linux-readonly-trace-v1-cpu.jsonl`. Bind exact commands/order, executable,
source, artifact/payload/workload, Python-validator, CUPTI library/source/native
library, telemetry-tool identities and user availability confirmation before launch.
Record process status, stderr, environment differences, disk limits and failures.
Preserve v2 and every existing baseline. Large raw traces remain local until scope,
compression, provenance and publication size have been reviewed.

## Trial attribution contract — must be implemented and tested first

The pinned harness performs one cold timed call, then B1 encode/decode/reconstruct/
block, each with one first call plus 35 loop calls. Therefore the first **145**
elapsed-time queries precede B1 stage profiling. Queries **110–145**, one-based,
are the B1 block: 110 first call, 111–115 warmups, 116–145 retained trials. The
full harness has 2,008 queries: `1 + 3 * (4 * 36 + 35 * 15)`. The 15 stage queries
per reconstruction are not ordinary timed-call windows and must not be silently
assigned as such. These counts come from source, not a verified full CUPTI capture.
Recheck both bound source versions; any changed ordering or extra query is a blocker.

Before interpreting a new capture, the analyzer must:

- Verify complete reports, all expected query counts, no dropped records, valid
  monotonic intervals/durations and no duplicate ambiguous correlation IDs.
- Sort host records by timestamp, not printed interleaving. Require exactly two
  ordered event-record calls between successive queries in the first 145 windows.
  Validate single host-thread attribution; do not guess when multiple threads or
  extra records make pairing ambiguous. Preserve the first call/warmups too.
- Correlate every selected kernel to its host launch and device/context/stream.
  Reject ambiguous, missing or cross-window matches. Check completed kernel
  intervals against the surrounding synchronized window. Do not infer stream
  identity from a kernel name or ignore copies/other activity in a window.
- Report per-trial kernel names/counts, duration sum, interval union and hull;
  report launch, allocation and synchronization API intervals separately. Sums
  can overlap. Hull minus union means time not covered by the selected kernels,
  **not proven device idleness or CPU overhead**. Keep copies and unclassified
  activity explicit. Event-record host intervals are not GPU event timestamps.
- Join the 30 retained block windows to report trials by the validated sequence,
  not by nearest duration. Publish all windows/processes and instrumentation
  overhead relative to the paired plain condition. Keep diagnostic wall/event
  spans distinct from kernel-duration sums and acceptance latency.

Required offline tests: shuffled/interleaved print order; overlapping kernels;
multiple streams/threads; duplicate/missing correlations; missing/extra boundaries;
invalid durations; dropped records; truncated captures; incorrect scope/query
counts; empty windows and copies. Verify the mapping against the retained B4
captures as a separately labeled historical fixture, not evidence of B1 behavior.
A runner additionally needs fake-child success/failure/timeout/size-limit/interrupt
and owned-child cleanup tests. No real benchmark is a substitute for these tests.

## Offline tooling progress — boundary structure only

`tools/bench/cupti_trace_boundaries.py` now validates DRIVER interval syntax,
correlation uniqueness, single-process/thread attribution, query/event counts and
the ordered two-event prefix. It sorts timestamps rather than trusting print order.
Layouts distinguish the historical 36-window B4 captures from the source-derived
2,008-query / 145-prefix full codec schedule. Output includes source-assumed labels,
input/tool hashes and explicit scope limitations; output creation is exclusive.
It **does not parse or validate kernel/copy activity**, match launches, qualify
benchmark outputs, prove capture completeness, or authorize execution.

Fourteen tests in `tools/bench/test_cupti_trace_boundaries.py` cover the synthetic
full schedule and malformed/missing/duplicate/overlapping boundaries, wrong layouts,
multiple processes/threads, dropped-record messages, shuffled print order and CLI
output guards. All three published historical B4 trace hashes and 36-window layouts
verify; each is rejected as a full B1-layout capture. This boundary-only checkpoint
did not implement kernel attribution or a bounded collection runner; see below
for subsequent offline kernel-helper progress.

Validation: `CARGO_BUILD_JOBS=4 ./tools/check.sh` passes, including **62 reporting
tests**. The first full check failed an existing child-RSS test: parsing large
archive fixtures in the suite process raised its inherited pre-exec RSS floor
(`76410880 not greater than 86949888`). Archive validation now runs in a separate
child; the existing resource assertion is unchanged. The corrected full check
passes. This is a test-isolation fix, not a benchmark retry or memory improvement.
Both local logs are retained:

- `reports/phase2-readonly-boundary-preparation-v1-check.log` (failed), SHA-256
  `6e36da8a179d69a520f35b32cfe18d4028fa62061e56a037fc034a848c31a59c`.
- `reports/phase2-readonly-boundary-preparation-v2-check.log` (passed), SHA-256
  `7a2bb8bc7265cebeed1ee7b170c310a85bd37bd3518191af58f3c8c573dcf060`.

Reproduce the focused offline checks with
`python -m unittest discover -s tools/bench -p test_cupti_trace_boundaries.py -v`.
No GPUs, model weights, ignored traces or Python model packages are required.

## Offline kernel/launch helper — still not a complete analyzer

`tools/bench/cupti_kernel_windows.py` correlates kernels with supported DRIVER **or
RUNTIME** launches and preserves every source-assumed block window (first/warmup/
retained). It rejects missing/duplicate correlations, graph kernels, ambiguous
multiple device/context/stream identities, invalid intervals and cross-window work.
Output contains per-kernel and per-host-API intervals, kernel sum/union/hull and
uncovered hull time. Uncovered time is not proven GPU idle time or CPU overhead.
Copy, memory, overhead and other timestamped activity headers remain explicitly
**unattributed**, not silently subtracted or explained. Copy metadata/correlation,
report output qualification and semantic trial/report binding remain pending;
`activity_attribution_complete` is always false.

The initial DRIVER-only implementation rejected valid historical fixtures. Direct
inspection found each capture's 972 kernels correlate to 720 `cuLaunchKernel`,
216 `cudaLaunchKernel_v7000` and 36 `cudaLaunchKernelExC_v11060` calls. The latter
two are RUNTIME records, not missing launches. Both recorded variants are now
handled and covered by regression tests; unknown/ambiguous launches still fail.
No capture was filtered or repeated. This is an offline parser correction, not
an explanation for the current B1 timing clusters.

Fourteen additional tests cover overlap/union arithmetic, shuffled activity blocks,
missing/duplicate/runtime correlations, metadata/graph rejection, stream/boundary
ambiguity, empty windows, copy/unknown-activity preservation, dropped/truncated
captures, source bindings and exclusive output. Synthetic full-layout coverage
selects all 36 B1 block windows; this is not a real full-harness B1 trace. All
three hash-verified historical B4 captures map all **972 kernels to 36 windows**
each. Large fixture work stays in child processes to avoid the earlier suite-RSS
interaction. No kernel timing or gap aggregate here is a new performance result.

Commands actually run:

- `python -m unittest discover -s tools/bench -p test_cupti_kernel_windows.py -v`:
  14 tests pass. Local log `reports/phase2-readonly-kernel-preparation-tests-v1.log`,
  SHA-256 `8a5f6c66a32bd5f9b762f9dd72ce7bfe73e2b410dcbc1a5b507a0526ed247afe`.
- `CARGO_BUILD_JOBS=4 ./tools/check.sh`: passes, including **76 reporting tests**.
  Final local log `reports/phase2-readonly-kernel-preparation-check-v2.log`, SHA-256
  `90cd97d7c26fe07c1cacb44a97f0ff086e19d3bc39c632775d00b5c9fedf9976`.

## Offline declared report/trial binding

`tools/bench/bind_cupti_trials.py` adds an ordinal join, not nearest-duration
matching. A strict declaration supplies layout, exact trace/report hashes, the
complete expected report source inventory and the expected trace process ID.
The helper verifies those declarations, the reviewed benchmark/stage source
hashes, pinned artifact/workload hashes, runtime/precision/profiling mode, complete
ordered 12-scope inventory and positive finite first/30-trial timing arrays.
Unknown source layouts fail explicitly. The current harness source is supported;
the harness version in the two older B4 captures is not silently accepted.

All 36 block windows remain present. The first-call timings and 30 retained
wall/event pairs join by source-derived index. **Warmup timings were never
serialized by the unchanged harness**: their trace windows remain, but the
joined `recorded_timing` is null, not fabricated or zero. No report is relabeled
as uninstrumented: the joined result is always diagnostic, even when a full-harness
native report's own `diagnostic_profiling` field is false under external injection.

A hash declaration is not independent proof that a particular binary produced
both files. The output explicitly leaves `launch_provenance_verified`,
`benchmark_outputs_qualified` and `activity_attribution_complete` false. Actual
same-child launch receipts, executable/source provenance, existing full-output
qualification and copy attribution remain mandatory. The declaration is an
integrity contract to be supplied by reviewed evidence, not a trust certificate.

Nine tests cover hash/PID/source/protocol mismatches, scope order/completeness,
invalid/nonfinite timings, warmup absence, source hashes, exclusive outputs and
synthetic B1 joins. The hash-verified archived retained-B4 trace/report pair joins
its first call and every retained timing exactly; archive parsing stays in a child.
This is historical fixture coverage, not new B1 measurement or parity evidence.

Commands actually run:

- `python -m unittest discover -s tools/bench -p test_bind_cupti_trials.py -v`:
  9 tests pass. Local log `reports/phase2-readonly-binding-tests-v1.log`, SHA-256
  `f21cee638fa49eb167d97a77f429c69c301268205fea1a4d90fb28ecf7cb57bf`.
- `CARGO_BUILD_JOBS=4 ./tools/check.sh`: passes, including **85 reporting tests**.
  Local log `reports/phase2-readonly-binding-check-v1.log`, SHA-256
  `5c75f548852c238c74e3571c196d939079ce044ff8bc55755678b7775ad89de4`.

## Offline copy and memset correlation

The kernel-window helper now also parses supported `MEMCPY` / `MEMSET` records,
including bytes, direction/kinds or fill value, interval, correlation and
non-graph device/context/stream identity. It requires a corresponding supported
host API and matches selected memory calls/activities in both directions. Missing,
duplicate, ambiguous, cross-window or incompatible-stream records fail, as do
unsupported copy counts, directions, kinds, APIs and graph provenance. Supported
formats come from the inspected retained traces, not an assumed universal CUPTI
parser. Unknown formats do not silently fall back to timing-only attribution.

The three historical B4 captures each contain **396 HTOD copies and 72 memsets**
within the same 36 windows as their 972 kernels. Those operations were previously
preserved as unattributed headers; they are now separately correlated. This is
another reason kernel-hull gaps must not be labeled CPU overhead or device idle
time. It is not new current-B1 evidence or a diagnosis of its timing clusters.

Output preserves kernel-only metrics and adds memory activities and a combined
kernel/copy/memset interval union/hull. Overlap is not double-counted. Memory
bookkeeping, profiler overhead and other timestamped records remain explicitly
unattributed. `activity_attribution_complete` remains false; no counter proves
capture completeness, output quality or launch provenance. Equal-timestamp host
records now have deterministic tie ordering for offline reproduction.

Nine new regression tests (23 kernel-helper tests total) cover copy directions,
DRIVER/RUNTIME memsets, invalid size/duration/count/kinds, missing/duplicate/wrong
correlations, stream/graph/boundary rejection, unknown APIs and tie ordering. The
archived B4 tests verify all kernel/copy/memset counts; synthetic union tests retain
separate kernel and combined coverage. All original trace/report records remain
untouched. No collection or policy change occurred.

Commands actually run:

- `python -m unittest discover -s tools/bench -p test_cupti_kernel_windows.py -v`:
  23 tests pass. Local log `reports/phase2-readonly-memory-tests-v1.log`, SHA-256
  `7e5dc72b244110b38c6fa214162cceb7a20a900b1142b7bb6e4c8afea4d499b5`.
- `CARGO_BUILD_JOBS=4 ./tools/check.sh`: passes, including **94 reporting tests**.
  Final local log `reports/phase2-readonly-memory-check-v2.log`, SHA-256
  `652ee458d11a9bc9c74ac537d7e4e7b70b996b74f6d10da4bdfa4ae474bb9b00`.

## Diagnostic report-output qualification

`tools/bench/qualify_diagnostic_codec.py` now wraps the existing
`phase2_summary.codec` validation without changing that implementation, its FP32
budgets, goldens or acceptance reader. It accepts exactly three supplied native
reports with consistent declared source/protocol/profiling metadata. Duplicate
reports are rejected even when their JSON formatting differs. Distinct report
bytes still do not prove fresh independent processes or executable provenance.

The Python reference inputs are fixed to v2 repetitions 1–3 from one archived
copy set, verified against the published archive and member hashes. The helper
does not run Python inference, count the duplicate copied reports as extra
observations, regenerate an oracle, or use archived timings for acceptance.
Native harness layout/source declarations are checked separately; actual launch
provenance remains unverified. All validation is offline and standard-library only.

The reused validator checks all twelve codec scopes, input/code/output identity
and retained-element counts, first decode/reconstruct logits under unchanged
FP32 tolerances, and stage/load report structure. It does not establish new
operator/642-trace parity, individual unreported block/snapshot values, all-trial
floating outputs, or current hardware performance. The output retains only
quality/identity evidence, explicitly sets `diagnostic_only` and leaves
`performance_acceptance_evaluated` and `launch_provenance_verified` false.

Four new tests include a child-isolated archived three-report qualification, nine
invalid-report cases (source/protocol/scope/output/logit/timing/stage/load mutations
and duplicated data), frozen archive/member corruption, source-layout guards and
CLI overwrite protection. Profile-marked reports can undergo output qualification
but remain rejected by the unchanged acceptance file reader. No native report or
historical acceptance control was edited or replaced. These are tool regression
fixtures, not a repeated v2 collection or new diagnostic outcome.

Commands actually run:

- `python -m unittest discover -s tools/bench -p test_qualify_diagnostic_codec.py -v`:
  4 tests pass. Local log `reports/phase2-readonly-qualification-tests-v1.log`, SHA-256
  `d417b8b82db3e1693871dcee109e475fdeae5ae6e3e7d6e41519f3dcb27f8ed4`.
- `CARGO_BUILD_JOBS=4 ./tools/check.sh`: passes, including **98 reporting tests**.
  Local log `reports/phase2-readonly-qualification-check-v1.log`, SHA-256
  `01076309cbbddf0ef4f54a92998cef3f6ed0855c98698c11b5c5fa4780502b29`.

## Bounded child-process primitive — fake CPU commands only

`tools/bench/bounded_child.py` now supplies a Linux-only, single-threaded collector
primitive, **not an executable diagnostic launcher**. It creates a new owned
session, streams merged stdout/stderr to an exclusive file, enforces a byte cap
and child/remaining-sequence deadlines, and writes an exclusive JSON receipt with
exit/cleanup state and captured-byte hash. SIGINT/SIGTERM ownership is temporary
and restored before receipt serialization, so a late interrupt is not absorbed
after status computation. Errors, interrupts, overflow and unconfirmed cleanup fail;
there is no automatic retry. Environment contents are not copied into receipts.

The direct child's PID remains waitable with `waitid(WNOWAIT)` until the final
owned-group signal. The collector never signals that numeric group after reaping,
and refuses to signal if child ownership becomes unconfirmed. Default SIGCHLD
and exclusive child-reaping ownership are prerequisites. Only the direct child is
waited/reaped; a descendant remaining in its group is stopped, not falsely claimed
as directly reaped. Cleanup/drain waits are separately bounded and included in
receipt elapsed time; this is not a hard-real-time or host-failure guarantee.

A successful receipt establishes only bounded capture and direct-child exit,
**not workload/output qualification or executable identity**. Aggregate trace
quota/free-space checks, the fixed 12-job manifest/order, observer affinity,
telemetry, frozen binary/payload checks and integrated provenance verification are
still absent. Abrupt collector SIGKILL/host failure and descendants escaping its
session are not contained by this primitive. Whole-collection supervision and
those boundaries must be resolved before launch; this is not a substitute for the
clock proposal's separately required restoration supervisor.

Twelve tests run only fake Python CPU commands. They cover success/exact output
limit, nonzero exit, overflow including buffered output after leader exit, child/
sequence deadlines, SIGINT/SIGTERM, direct reaping, group signaling before reaping,
a descendant holding the pipe, survival of a separate test-owned child, launch/
write/cleanup failures, lost reaping ownership, late receipt interruption and
prelaunch rejection/overwrite guards. They do not establish GPU behavior or actual diagnostic readiness.

Retained intermediate checks: `reports/phase2-bounded-child-tests-v1.log` passed
with a ResourceWarning in the simulated lost-ownership case; bookkeeping now
settles without signaling or interpreting unknown status as verified. The added
write-error fixture initially failed because its mock did not handle keyword
`mode`; `reports/phase2-bounded-child-tests-v3.log` retains that failure. The mock
was corrected; the collector's child had already been stopped/reaped. No benchmark
was run or repeated for either tooling correction.

Final commands actually run:

- `python -m unittest discover -s tools/bench -p test_bounded_child.py -v`: 12 pass.
  Local log `reports/phase2-bounded-child-tests-v5.log`, SHA-256
  `255cf77cb02cd24f470dfc1062e4a6c9ab366473b3109032e2d2bad92f4cd76e`.
- `CARGO_BUILD_JOBS=4 ./tools/check.sh`: passes, including **110 reporting tests**.
  Local log `reports/phase2-bounded-child-check-v3.log`, SHA-256
  `39e11960b49c9c26c20c80d0cd8d01fa312487d37f9cc361d8aaa34c23193466`.

## Launch blockers and outcome rules

This document is **not executable launch readiness**. Offline helpers and a tested
child primitive are only prerequisites; the fixed collection runner, whole-run
supervision, integrated launch/report provenance and remaining tests are pending.
No new capture exists to qualify. Before collection, the user must
choose the read-only investigation and confirm a fresh quiet 15-minute window;
the tested tooling, exact frozen manifest and identity checks must also be ready.
No extra profiling permissions, system installation or CUPTI replacement may be
made implicitly. Stop and ask if existing local tracing cannot be used safely.

Possible outcomes: incomplete/unattributable capture; spread not reproduced under
instrumentation; or spread associated with specific measured kernel/API intervals.
Instrumentation can perturb allocation and scheduling enough to suppress or create
the spread. Three processes per condition cannot guarantee both clusters occur.
Do not keep sampling until they do, filter a process, transplant results, or infer
causality from association alone. If inconclusive, report exactly what remains
unobservable and ask before a different experiment. Every outcome leaves Linux
**72/78** pending a separate full acceptance protocol with unchanged criteria.
