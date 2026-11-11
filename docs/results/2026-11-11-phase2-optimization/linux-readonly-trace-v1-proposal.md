# Read-only Linux trace investigation — selected, NOT RUN

Alternative to the optional [clock intervention](linux-clock-diagnostic-v3-proposal.md).
No GPU-clock, power, governor, service, credential or global-tooling changes.
The user approved option 1 first, with option 2 as a fallback if its data are
insufficient. Selection does not establish launch readiness or a quiet window.
Linux remains **72/78**; this is not acceptance and does not unblock Apple.

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
No inherited injection in controls, launcher or observers. Planned child/observer
environments are constructed explicitly: system default PATH, current HOME,
LANG=C, LC_ALL=C, PYTHONNOUSERSITE=1; traced children alone add
CUDA_INJECTION64_PATH. Build hints CUDA_PATH/COMPUTE_CAP and unrelated inherited
variables are not propagated. This is an explicit environment difference from v2,
not evidence that its timing conditions are identical. Native-library resolution
under this environment must be checked before launch. No Python benchmark is
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

## Child prelaunch-boundary audit

Two additional regression tests exposed a setup gap: cancellation observed while
installing handlers, or a deadline consumed while opening output files, still
reached `Popen` before the loop checked either condition. The tests mocked `Popen`
and demonstrated both failures without creating a child. The primitive now
rechecks observed cancellation and the effective deadline immediately before the
spawn attempt and retains a failed, no-PID receipt when setup has invalidated it.
This is not atomic cancellation or a hard-real-time guarantee: events arriving
during `Popen` still take the existing owned-child cleanup path.

Fake child environments are now minimal (`PATH`, `PYTHONNOUSERSITE`), so failed
mock assertions do not dump the collector's inherited environment. The original
local `reports/phase2-bounded-child-prelaunch-red-v1.log` captured inherited metadata and
is retained with mode 0600, **not for publication**. The minimal-environment
reproducer is `reports/phase2-bounded-child-prelaunch-red-v2.log`: both tests fail
before the fix, SHA-256
`baf2d57e952b5c1e2d2ed7bbd90532017543ec57a652b9d7241ea547c3a33963`.

Final commands actually run:

- `python -m unittest discover -s tools/bench -p test_bounded_child.py -v`: 14 pass.
  Local log `reports/phase2-bounded-child-prelaunch-green-v1.log`, SHA-256
  `78c9d8d10498f4ffa87497063e5788d1b6f158f542f8067e8aebcb125c045940`.
- `CARGO_BUILD_JOBS=4 ./tools/check.sh`: passes, including **112 reporting tests**.
  Local log `reports/phase2-bounded-child-prelaunch-check-v1.log`, SHA-256
  `0d1253f6e5a31dfffa16744e5ff94e0cd352118e1d9f6d8770a77fddf24fac2c`.

No benchmark, GPU-policy change, acceptance update or Apple work occurred.

## Fixed sequence and quota prerequisite

`tools/bench/readonly_trace_sequence.py` adds the fixed 12-job order, complete
structural job validation before output creation, an exclusive output directory
and pre-sequence plan, 600-second shared deadline, 120-second child limits,
1 GiB per merged capture and 6 GiB aggregate streaming cap. The per-capture cap
also applies conservatively to plain controls. At least 12 GiB free is required
initially; free space is rechecked before each child with the original headroom
adjusted only for this sequence's recorded capture bytes. External writers can
still cause a write failure between checks. Any failure stops subsequent jobs;
no retry, reordered job or replacement process is allowed. Environment values
are not serialized into the plan. Exceptions retain partial evidence and propagate.

This is an import-only prerequisite, **not a launch-ready collector**. It trusts
caller-supplied commands and the child primitive's receipts; it does not validate
executable/injection identity or enforce observer affinity. Telemetry, same-child
trace/report extraction, frozen identities and whole-run supervision remain
pending. Collector death and escaped descendants are still outside its containment.
The sequence's completion means only all child captures completed, never codec
qualification, provenance, acceptance or complete telemetry.

Nine offline tests cover all 12 fake CPU commands in order, validation of the last
job before any launch, exclusive paths, free-space preflight/rechecks, reduced
remaining aggregate allowance/exhaustion, real streaming overflow/nonzero exit,
setup-expired deadline and partial evidence on interruption. No GPU was used.
Focused command: `python -m unittest discover -s tools/bench -p test_readonly_trace_sequence.py -v`;
9 pass, local log `reports/phase2-readonly-sequence-tests-v1.log`, SHA-256
`3d51794611f573884506d3ca0a66933f60f075ca0c4f97b32dce2cc0d8d5f880`.
`CARGO_BUILD_JOBS=4 ./tools/check.sh` passes with 121 reporting tests; local log
`reports/phase2-readonly-sequence-check-v1.log`, SHA-256
`f610318acd2c7afc03c1758ad7d14acab70cb7f1d6a309c5e912d012435c49ff`.

## Collector-side report receipts

The child primitive now optionally records a report file's path, inode/device,
size and SHA-256 in the same receipt as the child's PID, exact argv and merged
capture hash. The report must be an absent absolute argv argument beside the
capture, distinct from collector outputs. After a successful exit and capture,
only a nonempty regular non-symlink file of at most 16 MiB is read, using a bounded
streaming hash; descriptor/path metadata changes while hashing fail. FIFOs are
opened nonblocking and rejected, not allowed to hang report collection. A missing,
oversized, empty, symlink or visibly replaced report makes the receipt fail even
if the child exited zero. Failed children retain partial files without promoting
them to verified report captures. Restored signal handlers remain effective during
post-exit hashing; interrupts are not swallowed to manufacture a complete receipt.

The sequence supports these optional report outputs at unique predetermined
`NN-CONDITION.report.json` paths, validated for all jobs before any launch. Tests
exercise all 12 fake CPU producers and reject duplicate report destinations.
Eight additional report-receipt tests cover the file/hash/path/failure cases;
two sequence tests cover integration. No actual codec report or GPU trace was
collected. Both fresh-report absence and post-exit hashes are integrity evidence,
**not independent producer identity, executable provenance or output quality**.
Trusted directories with no concurrent external writers remain required; metadata
checks are not an adversarial snapshot. The report cap bounds post-exit reads,
not live child disk writes, and is separate from the streaming trace cap. Complete
frozen-command/identity preflight must therefore precede using this interface.

Validation actually run:

- `python -m unittest discover -s tools/bench -p 'test_bounded*.py' -v` (22 pass),
  then `python -m unittest discover -s tools/bench -p test_readonly_trace_sequence.py -v`
  (11 pass). Local combined log `reports/phase2-readonly-report-receipt-tests-v1.log`,
  SHA-256 `8b5758b103a684854d7873e0e9c2c48e76a61129bc929e253c04ca948d923470`.
- `CARGO_BUILD_JOBS=4 ./tools/check.sh`: passes with **131 reporting tests** after
  recording the report limit explicitly in receipts. Local final log
  `reports/phase2-readonly-report-receipt-check-v2.log`, SHA-256
  `ab67bc663f90076cf9b63ad6ea2401bda775b85f8d9d159bac5a128573a65bda`.

Next: connect collected trace PID/report hashes to the existing ordinal binder
and output qualifier, bind frozen executable/input/library identities, and finish
whole-run supervision and telemetry/affinity integration. These receipts alone do
not resolve those launch blockers or authorize treating diagnostic timings as
acceptance evidence.

## Integrated offline sequence analysis

`tools/bench/analyze_readonly_trace.py` now connects the sequence plan/result and
individual child receipts to the existing ordinal binder and codec qualifier.
It requires all 12 jobs in the fixed order, unchanged limits, successful reaped
child receipts, matching commands/report destinations, distinct receipt PIDs,
exact capture/report hashes and byte totals, and distinct canonical reports.
Files are read only from fixed local names; original absolute paths are metadata,
so a reviewed collection can be relocated without following paths into unrelated
directories. Symlink/FIFO/oversized inputs fail. Expected accepted/candidate source
inventories are supplied separately, not inferred from reports being checked.

All four three-report groups (AP/CP/AT/CT) pass through the unchanged output
qualifier and pinned archived Python references. All six traced captures pass
through the full B1-layout binder with the **receipt PID** and receipt hashes.
Every first/warmup/trial window is retained; unavailable warmup timings remain
null. Plain captures containing supported trace record markers are rejected;
absence of those markers does not establish injection isolation. Descriptive
instrumentation comparisons retain all three occurrence pairs per implementation,
all 12 execution scopes, load trials and first-load/first-execution/scope timings.
There are 78 descriptive pair/scope entries, **not 78 acceptance comparisons**.
No pair is selected by nearest duration or discarded for its instrumentation cost.

The analyzer verifies consistency of supplied evidence, not its independent
provenance: `launch_provenance_verified`, `performance_acceptance_evaluated` and
`telemetry_coverage_verified` remain false. Neither distinct hashes/PIDs nor these
joins establish freshness, actual binary/injection identity, same-child production
of a file, or supervised collection. Per-process pairs are occurrence-order pairs,
not simultaneous controls or complete carryover balance. Qualification remains
limited to the existing reported-output surface, not all-trial/642-trace parity.
No new collection exists to analyze.

Three offline tests exercise the full synthetic 12-job collection, 13 negative
binding/sequence cases, bounded input handling, all four groups of real archived
v2 native reports through the actual qualifier, a numerically corrupted report
with updated valid hashes (rejected), and exclusive CLI output. Archived reports
are paired with **synthetic traces/receipts** solely for integration testing, never
represented as measured B1 evidence. Large parsing runs in subprocesses to preserve
the existing inherited-RSS test boundary.

Checks actually run:

- `python -m unittest discover -s tools/bench -p test_analyze_readonly_trace.py -v`:
  3 pass. Local final log `reports/phase2-readonly-analysis-tests-v2.log`, SHA-256
  `4790e62f5edded7082d65a13f342fc6ea33d9a360885becb9055045f084d83ee`.
- `CARGO_BUILD_JOBS=4 ./tools/check.sh`: passes, including **134 reporting tests**.
  Local log `reports/phase2-readonly-analysis-check-v1.log`, SHA-256
  `8abf971a2e5444a35748da62f29c05760879b245d81d6e61a69a330b12d01965`.

Next: frozen executable/input/library identity and injection-isolation preflight,
whole-run supervision, observer affinity/telemetry integration and readiness review,
then a fresh quiet window. No GPU measurement, clock setting, production change,
acceptance update or Apple run occurred.

## Frozen identities and environment construction

`tools/bench/readonly_trace_preflight.py` extracts the accepted/candidate codec
binary, source, fixture, manifest and complete six-file codec payload identities
from hash-bound published v2 archive members. It does not choose expected hashes
from current files. Both 12-file source inventories, both fixtures, both frozen
executables, shared manifest/payloads, and the retained original CUPTI library and
four sample/source files must match. Regular-file streaming hashes reject leaf
symlinks, FIFOs, size/digest mismatch and visible read-time mutations. Parent
directories remain trusted. This is byte identity, not renewed tensor preflight,
finite/layout validation or independent source-to-binary compilation proof. The
original CUPTI compile invocation was not retained; its earlier provenance limits
are unchanged. The installed taskset utility is observed/hashed, not compared to
an older frozen pin.

The helper validates CPU2/CPU0 eligibility/SMT placement, then constructs exactly
the fixed 12 commands with absolute executable/input/report paths and CPU2 taskset.
All jobs and observers receive independent copies of the explicit environment
above; only traced children receive CUPTI injection. Nonempty inherited LD_*,
CUPTI_*, NVTX_*, NCM_*, CUBLAS_* and runtime CUDA_* controls fail preparation rather
than being silently honored; CUDA_PATH/COMPUTE_CAP are permitted build hints but
omitted from child environments. No inherited secrets are serialized. This is
planned environment isolation, not verification of a live process's mappings.

Six offline tests cover real published archive pins, fixed argv/order, independent
environments, inherited control rejection, changed binary/source/fixture/payload/
CUPTI inputs, output/executable/topology guards and file-type/read-time mutation
checks. A Popen trap verifies job preparation does not execute even fake binaries.
The first focused test run failed because its temporary test filename `target`
collided with the fixture's build directory; renamed only that test file. Retained
failed log `reports/phase2-readonly-preflight-tests-v1.log`, SHA-256
`c516d9ae517e1f1183b0ee9f6edca227ec81c06ab7ad4d300d8228714302573b`.
The corrected focused command
`python -m unittest discover -s tools/bench -p test_readonly_trace_preflight.py -v`
passes six tests; log `reports/phase2-readonly-preflight-tests-v2.log`, SHA-256
`a2b01587cb51fccf7a0c8ec0d21504509ee4afb41690abf7609258a362eb01f0`.
Final `CARGO_BUILD_JOBS=4 ./tools/check.sh` (including the explicit user-site disable)
passes **140 reporting tests**; log `reports/phase2-readonly-preflight-check-v2.log`,
SHA-256 `1a27720fef6cf436d775c039a3962511f150e210e525d618c81989ef17d54cdc`.

An actual **file-only preparation** also succeeded locally: 40 pinned file entries
match, plus one observed taskset identity, and all 12 commands were constructed but
**not executed**. The temporary checks wrote neither a collection directory nor
its launch manifest. Local preparation snapshot
`reports/phase2-readonly-preflight-local-v2.json` has SHA-256
`d24297f3970461cc2c4de3fcbccc3a976cb4057c7ea095a303c585ec11e5bcbc`;
prior preparation v1 remains retained. Reproduce the preparation call (no launch):

```python
from pathlib import Path
from readonly_trace_preflight import prepare  # tools/bench on sys.path
root = Path.cwd()
prepared = prepare(root, root / 'reports/phase2-linux-readonly-trace-v1', Path.home())
assert prepared['launch_ready'] is False
```

This is not a reusable launch certificate: recheck identities at actual launch.
Native-library resolution, worktree state, observer placement/telemetry, whole-run
supervision and fresh availability remain explicitly unverified. No benchmark,
GPU policy mutation or Apple work occurred; Linux acceptance remains 72/78.

## Loader metadata and worktree-state guards

`tools/bench/readonly_trace_runtime.py` adds bounded metadata probes (10 seconds,
1 MiB merged output each), retained command/exit receipts and exclusive private
0700 evidence directories. The installed glibc loader's **`--list`** mode lists
DT_NEEDED dependencies without running benchmark main or requesting CUDA
initialization. Each prepared target is rehashed against its recorded identity
before any probe, then again after listing. The accepted/candidate codec, original
injection library and taskset are listed under the prepared clean environment. Unsupported/unresolved/duplicate
listing records, missing required dependencies and differing accepted/candidate
resolutions fail. Canonical library files are hashed, symlink destinations checked,
and utility/target/library identities rechecked after listing.

An actual local metadata collection succeeded for all four targets, identifying
**14 canonical libraries**. Accepted/candidate dependency resolutions match.
`static_loader_resolution_verified=True` is limited to these loader listings;
`runtime_loaded_libraries_verified=False` and `launch_ready=False` remain explicit.
This does not cover lazy dlopen, driver/JIT dependencies, actual benchmark process
mappings or equivalence to libraries loaded during historical measurements. Current
library identities are observed, not invented historical pins; freeze/recheck them
at collection. No benchmark main or GPU initialization was requested.

The Git guard requires the accepted `3307dff` checkout and a caller-declared current
candidate HEAD, both tracked-clean; only candidate untracked `.pi/` files are
permitted. It disables optional locks/fsmonitor/global configuration and does not
fetch, checkout, reset or invoke credentials. Raw status captures may name private
session files: they remain in a private local directory and **must not be published**.
Summaries contain only their count. A real invocation correctly rejected this
session's uncommitted tooling; clean current-checkout verification is still required
after publication and again before launch. Temporary real Git repositories also
exercise the successful path in tests without changing global configuration.

Production source and Cargo.lock must match `d44beaa`. Initially comparing all
Cargo manifests to that commit also exposed the already-approved `codec_memory`
example declaration added at `211b51d`. Direct diff inspection confirms that sole
four-line example addition. Manifest guards therefore compare to the **already
measured v2 tree `fa040ed`**, while source/Cargo.lock remain anchored to `d44beaa`;
all reviewed diffs under those respective anchors are empty. No dependency pin or
production change was made. The inspection is retained at
`reports/phase2-readonly-runtime-manifest-review-v1.log`, SHA-256
`14bfd8cf68c13941a8f2a79071f185143d2c88078ca315d3e4b8dba08f89a6c0`.

Eight tests cover parser variants/rejections, all target/library bindings,
identity/environment failures before probes, missing/different dependencies,
real tiny CPU metadata children, private summary handling, real temporary Git
repositories and seven negative worktree/head/source/manifest cases. Commands:

- `python -m unittest discover -s tools/bench -p test_readonly_trace_runtime.py -v`:
  8 pass. Local final log `reports/phase2-readonly-runtime-tests-v3.log`, SHA-256
  `45482127e9cbf34828f653b74ee95d2afa9a06649dd67007bc28c186486d87f1`.
- `CARGO_BUILD_JOBS=4 ./tools/check.sh`: passes with **148 reporting tests**;
  `reports/phase2-readonly-runtime-check-v2.log`, SHA-256
  `2e222efce2bb3bc05f342c10848d22deb42fd2fc7b251bb49c210ebbd24052df`.
- Actual metadata result `reports/phase2-readonly-runtime-libraries-v1/libraries.json`,
  SHA-256 `4f03368072888c3948aa40f0481c54a422ea5e93f42e45e651883bc740cb5dc4`;
  run log `reports/phase2-readonly-runtime-local-v1.log`, SHA-256
  `fb6872f59c9c8627ca1c3c4e6eee085b5f6e91c01188413f38799fbeefe2e79e`.

Next: clean worktree recheck, whole-run supervision, runtime library coverage and
observer/telemetry integration, then readiness review and fresh quiet window.
All 12 proposed read-only benchmark processes remain unrun; Linux stays 72/78.

## Independent worker-supervision prerequisite

Before edits in this checkpoint, the actual clean-worktree guard passed at
candidate `156d976cd06601dce6cb38807fff6f87481a4999` and accepted `3307dff`.
Production source/lock and measured-v2 manifest guards passed. Private local
`reports/phase2-readonly-clean-worktrees-v1/worktrees.json` has SHA-256
`f673ce95eff41bd906933f1d9157c22b0b75e3093fceca2db3b05b96f04de16b`.
This is checkpoint evidence, not a substitute for another launch-time recheck.
Raw Git status captures remain private and unpublished.

`tools/bench/supervised_worker.py` now supervises one trusted worker in a dedicated
Linux interpreter. It requires the caller-supplied parent PID, one Python/OS
thread, default SIGCHLD, no pre-existing children and no existing subreaper role.
A narrow ctypes call to Linux PR_GET/SET_CHILD_SUBREAPER (37/36) changes only this
supervisor process's child-adoption role; no system service/policy is changed.
The original role and signal handlers are restored on verified cleanup.

The existing bounded-child primitive gains a trusted nonblocking cancellation
predicate, checked before spawn and during capture. The supervisor uses it to
notice parent disappearance or orphan adoption while the worker lives. Worker
crash/timeout/interrupt/parent loss stops the worker; a surviving supervisor then
adopts and cleans descendants, including descendants in new sessions and successive
generations. Each numeric PID is verified waitable before signalling, and reaped
only afterward; unconfirmed ownership prohibits signalling. Cleanup is bounded to
two seconds beyond the worker primitive's bounded cleanup/drain. Any orphan,
missing ownership, incomplete cleanup or role-restoration failure makes the
supervisor result fail, even if the worker exited zero. Unrelated processes are
never enumerated or signalled by process name or GPU occupancy.

This is a tested **worker supervisor primitive**, not the integrated collection
coordinator. It must remain alive independently of the worker. Supervisor SIGKILL,
host failure and uninterruptible kernel tasks cannot be guaranteed cleaned up;
failures retain partial evidence and are not successful containment. No real CUDA
abort/teardown behavior is established by fake CPU children. Worker captures are
bounded to 1 MiB; the full diagnostic coordinator must still enforce the existing
600-second sequence and 900-second telemetry protocol, provide its total deadline,
retain all collection evidence and apply observer placement. No benchmark or
GPU-setting command is embedded in the supervisor.

Eleven tests exercise success/nonzero exit, worker SIGKILL/normal-exit orphans,
timeout/interrupt, parent loss after worker readiness, cancellation before spawn,
multiple generations escaping sessions, orphan adoption while the worker lives,
setup/capture/cleanup/restoration failure, lost ownership and unrelated-child
survival. Real supervision runs in dedicated child interpreters; parent-loss tests
use a separate adopting driver and reap the orphan supervisor too. Successful
cleanup tests independently check the role reset, absence of children and reaped
PIDs, not only the returned status flag.

The first full check failed the existing resource-test comparison:
`76382208 not greater than 76767232`. The larger suite's launcher RSS floor affected
the test's small child. Its two measurements now run in the **same fresh interpreter**
instead of the accumulated suite process, still large then small, with the same
64 MiB allocation, same wait4 field and unchanged 32 MiB assertion. This is test
isolation, not a resource-tool, acceptance threshold or benchmark change. Failed
log `reports/phase2-supervised-worker-check-v1.log` is retained, SHA-256
`e4ea9406a7d31a897dec1d281f8ea7c19496fd252d8a47011fc1f491ca29e4cd`.

Commands actually run:

- `python -m unittest discover -s tools/bench -p test_supervised_worker.py -v`:
  11 pass. Local final focused log `reports/phase2-supervised-worker-tests-v3.log`,
  SHA-256 `c563d6ed233eef5fdc93cfc956dddc1b8638d10a20b7b4cb54a5770a11640578`.
- `CARGO_BUILD_JOBS=4 ./tools/check.sh`: passes with **159 reporting tests** after
  isolation and the OS-thread guard; final log
  `reports/phase2-supervised-worker-check-v3.log`, SHA-256
  `153a1633f82621cc2a7019bccdf62258d728dee5cdb6e53bf46e7a62081a39ea`.

Next: compose this supervisor with the fixed sequence and full telemetry lifecycle,
complete runtime-library/provenance coverage and integrated failure-path tests,
then readiness review and a fresh quiet window. No new GPU run, setting change,
acceptance update or Apple work occurred.

## Telemetry worker and sequence cancellation

`tools/bench/readonly_trace_telemetry.py` constructs read-only commands for the
existing CPU sampler (900 seconds, interval 1) and a UUID-selected nvidia-smi CSV
loop (existing clock/utilization/memory/temperature/power columns plus UUID).
It launches **no codec benchmark**. A dedicated worker pins itself/GPU observer
to CPU0. The CPU sampler receives startup affinity `{0,2}` because its existing
preflight checks target CPU2 eligibility before pinning itself to CPU0; its first
metadata must confirm final affinity `{0}`. The sampler source and its validation
are unchanged. No workload may begin until both observers' first records pass.

Readiness requires the expected CPU source hash, target/observer CPUs, duration,
interval and first sample index; the GPU CSV header, timestamp syntax and UUID
must match. A completed marker is published with an exclusive atomic hard link,
not exposed as partially written JSON. Startup has a 15-second bound. GPU exit,
invalid first records or observer output size/type violations cancel CPU capture.
Observers have 16 MiB polling/post-read bounds (not hard live file-size quotas),
and CPU command stdout/stderr has a streaming 1 MiB cap. CPU completion requires
all 901 sample indices plus metadata/completion records and the full duration;
missing completion is not success. GPU shutdown is intentional SIGTERM before
reaping, with bounded SIGKILL escalation; lost ownership never permits signalling.
Escalation/unexpected exit is retained as failed telemetry, not silently promoted.
The worker must itself run beneath the independent supervisor for orphan cleanup.

The fixed sequence now accepts the same trusted nonblocking cancellation hook,
checked before each job and inside the bounded child. A real fake-observer test
waits for atomic readiness, starts job 1, then causes observation failure; the
in-flight child is stopped/reaped and jobs 2–12 are not launched or replaced.
The final coordinator must still compose this path with the supervisor and wait
for the full sampler after successful/ordinary failed sequence work, unless
interrupted or unsafe. No real CPU/GPU sampling window was started here.

Eight tests use **short synthetic CPU/GPU producers**, never nvidia-smi or model
execution. They cover fixed command/affinity construction, prefix identity and
protocol failures, complete/incomplete capture, GPU failure, startup/output limits,
interrupt cleanup, owned-PID escalation/loss and in-flight sequence cancellation.
Dedicated test workers independently verify restored affinity and no unreaped
children. Short test durations/records are fixtures, not 900-second evidence.
`telemetry_contents_qualified`, `sequence_coverage_verified` and
`launch_provenance_verified` remain false: full CSV/counter/timestamp recomputation,
sequence coverage and runtime provenance are still mandatory downstream checks.
No per-request clock inference or performance acceptance is provided.

Staged-code review added a delayed-prefix regression: validation that consumed the
startup allowance could still publish readiness afterward. Before the fix the
synthetic case returned `complete` instead of `failed`; retained red log
`reports/phase2-readonly-telemetry-tests-v3-red.log`, SHA-256
`29f3f8b314101619ae2d029b0583466bb61cb2922ecbad9b355f4edce75eab8b`.
Readiness now checks the deadline before validation and again before atomic
publication. A late pending marker is retained but never linked as ready. This is
an observed-deadline guard, not a hard-real-time or atomic cancellation guarantee.
No measurement was repeated for this fake-process regression.

Commands actually run:

- `python -m unittest discover -s tools/bench -p test_readonly_trace_telemetry.py -v`:
  8 pass including the delayed-prefix case. Final focused log
  `reports/phase2-readonly-telemetry-tests-v4-green.log`, SHA-256
  `ccad4cdf6d832a847766f16b46690e63cb809943b1fd0a4fae05976adf08a3a7`.
- `CARGO_BUILD_JOBS=4 ./tools/check.sh`: passes with **167 reporting tests**;
  final local log `reports/phase2-readonly-telemetry-check-v2.log`, SHA-256
  `f31ce844e6a1ef71506e8e2b8ed15fd46096ab6cad062021fb64343855d2ee5a`.

Next: integrated coordinator/supervisor entry and frozen launch manifest, complete
telemetry coverage/content verification and runtime-library/provenance coverage,
then readiness review and a fresh quiet window. Linux remains 72/78; no GPU run,
clock/power change, production modification or Apple work occurred.

## Supervised coordinator composition — synthetic workers only

`tools/bench/readonly_trace_coordinator.py` now composes the existing observer and
fixed sequence in one dedicated worker. It is import-only: no real collection CLI,
frozen launch manifest or availability claim is supplied. The final entry must
run it under the independent subreaper, not as an unsupervised library call.

Before spawning observation it validates all 12 jobs and their sequence-owned
report paths. An exclusive private directory contains `telemetry/`, `sequence/`,
`observer-worker.txt` and `collection.json`; the final launcher must bind this nested
layout and any publication/sibling-file exports explicitly before collection.
The observer must publish atomic readiness with its actual waitable worker PID,
CPU0 affinity, expected sampler source/protocol, GPU UUID and explicit local-time
offset. Recheck exit/cancellation/deadlines after parsing readiness before job 1.
No readiness inferred from merely creating the directory or starting the process.

The coordinator stays on CPU0; prepared workload commands retain CPU2. It monitors
the observer while bounded jobs execute, cancelling an in-flight job and stopping
submissions if observation exits (even exit 0 before the sequence finishes), fails
or exceeds its lifetime/output allowance. Ordinary job nonzero/timeout/capture
failures stop later jobs **without truncating the sampler**. A signal caught by a
bounded job is propagated from its receipt, so SIGTERM does not accidentally turn
into an ordinary failure followed by an unwanted 900-second wait. Exceptions,
interruptions and unsafe observer conditions instead enter bounded cleanup.

Startup allowance is 20 seconds including interpreter setup; the observer's own
15-second first-sample deadline is unchanged. The coordinator has a 950-second
observation wait allowance including setup headroom, followed by bounded cleanup;
this does not extend the 600-second sequence, 120-second jobs or 900-second sampling
protocol. On abort it sends TERM
only while ownership is held, allows up to six seconds for observer cleanup, then
performs final group cleanup **before** reaping (two-second reap bound). A surviving
independent subreaper must clean adopted descendants and mark orphan cases failed.
Observer text and JSON reads have 1 MiB bounds; text is checked by polling and can
overshoot, not an enforced filesystem quota. Existing trace streaming caps remain
unchanged. Trusted directories/no external writers remain required.

Eight `test_readonly_trace_coordinator.py` tests use short synthetic CPU observers,
not nvidia-smi, CUPTI or model binaries. They cover all 12 jobs, ordinary first-job
failure with completed observation, observer nonzero/early successful exit, missing
readiness, mismatched PID/affinity/source/UUID/timezone, expiration during readiness
validation, output/deadline failure, missing result, sequence setup exception,
invalid prelaunch jobs/paths, and lost ownership without signalling. Real dedicated
subreaper tests verify coordinator SIGTERM and SIGKILL cleanup, including observer
and GPU-surrogate PID disappearance and no remaining adopted children.

The first test run failed because the deadline fixture used fast jobs yet asserted
that job 2 never launched before the deadline. The fixture now leaves job 1 running
through that deadline; no production timeout, gate or expected failure semantics
were relaxed. Both logs are retained. Commands and SHA-256:

| Command / retained local log | Result | SHA-256 |
| --- | --- | --- |
| `python -m unittest discover -s tools/bench -p test_readonly_trace_coordinator.py -v` → `reports/phase2-readonly-coordinator-tests-v1.log` | Initial fixture assertion failure | `a9483adee4bdcf73afaa19398fa15b812bbbf3aa13ea9b2a4cc4f643dac38809` |
| Same command → `reports/phase2-readonly-coordinator-tests-v2.log` | 8 tests pass | `641c990fa3864f5a7c9b1dcf092d65bfc18ed3c2e7c7d6009786d3c660b5ac12` |
| `CARGO_BUILD_JOBS=4 ./tools/check.sh` → `reports/phase2-readonly-coordinator-check-v1.log` | Pass, 175 reporting tests | `bb3df5dd9ada379869325b1b17cddb191e4a795d46e46316474867c8cc92297b` |

Success here means lifecycle composition, not complete telemetry content/coverage,
output qualification, runtime library/executable provenance or acceptance; those
flags remain false. Remaining work is the frozen launch manifest/entry, independent
runtime-process/library/injection bindings, complete telemetry verification and
remaining integration failure paths. Supervisor SIGKILL/host failure/uninterruptible
tasks are not guaranteed contained, and fake children do not prove CUDA abort
behavior. No real telemetry window or GPU query/model benchmark was run. Linux
remains 72/78. Review readiness before requesting fresh quiet availability.

## Offline telemetry contents and receipt-window bracketing

`tools/bench/validate_readonly_telemetry.py` adds an offline validator and exclusive-
output CLI; it never launches observers or benchmarks. `validate(cpu_raw, gpu_raw,
jobs, expected_cpu_source=..., gpu_uuid=..., timezone_name=...)` requires the expected
CPU source, GPU UUID and IANA timezone from independently reviewed launch evidence,
not values inferred from the supplied samples. The final manifest must still bind
these declarations and inputs to the actual collection.

CPU checks cover metadata, all 901 ordered samples, the unchanged 900-second/one-
second schedule, complete duration/count, stable CPU inventory including 0/2/3,
first-eight tick fields, recomputed activity, context-switch counters, frequency/
load/pressure syntax and explicit unavailable labels. Decreased tick counters and
zero tick deltas retain the sampler's unavailable status; they are not silently
excluded or turned into CPU work. Snapshot read intervals must be ordered. The
UTC-to-monotonic offset must fit all read intervals; an unrecorded scheduling
origin must also fit lateness and completion bounds. The timestamp checks allow
one microsecond for serialized datetime quantization/clock conversion, **not a
model numerical tolerance, sampling-gap allowance or new acceptance threshold**.

GPU CSV checks require the complete header, UUID on every row, supported units/
numeric ranges or explicit N/A labels, increasing timestamps and a caller-declared
IANA timezone. Ambiguous/nonexistent DST times fail rather than guessing a fold.
Every complete declared receipt window must be ordered, use its unique PID and be
bracketed by both CPU and GPU samples. Output retains every GPU sample, overlapping
interval indices, CPU lateness/read durations, all intersample gaps, optional-data
absence and GPU-vs-CPU endpoint offsets. An interval spanning missing GPU samples
is still only a **coarse bracket**, never continuous observation or per-trial clock
attribution. No gap is filled or excluded; no new maximum-gap acceptance rule is
invented. Gap/absence sufficiency must be dispositioned in the investigation.

The CLI takes positional `cpu`, `gpu`, `sequence_result`, `output` paths and required
`--cpu-source-sha256`, `--gpu-uuid`, `--timezone` declarations. It uses bounded,
no-follow reads and exclusive output, records raw/sequence/canonical-job/tool hashes
and leaves live provenance, continuous GPU observation and acceptance flags false.
It does not replace the existing codec/trace analyzer or prove supervisor/worker
completion, collector hashes, same-child production, timezone provenance or fresh
independent processes. These cross-bindings remain launch/audit prerequisites.

Ten tests exercise the actual CPU serializer over a virtual clock/fake procfs (no
real 900-second wait), correct counters/windows, late samples, unavailable deltas,
truncation/duplicate JSON keys, bad source/topology/affinity, malformed optional
fields, nonfinite/negative values, schedule/time inconsistencies, GPU UUID/units/
DST errors, retained GPU gaps, receipt failures/overlaps/out-of-coverage, and CLI
hashes/overwrite rejection. A separate fresh-interpreter regression validates all
901 retained v2 CPU samples from the SHA-bound archive: original raw CPU digest
`397f9aa3fdc1f6ca86f1d7f3e72d8e27241f8a79f9c5e1ca124851f237516c47`, duration
900.0006672290037 seconds, no optional-data gaps/unavailable activity. This is
serializer compatibility, **not a v2 rerun, new GPU capture or new 12-job evidence**.

| Command / retained local log | Result | SHA-256 |
| --- | --- | --- |
| `python -m unittest discover -s tools/bench -p test_validate_readonly_telemetry.py -v` → `reports/phase2-readonly-telemetry-validation-tests-v1.log` | Initial 7 tests pass | `142c031d1569f909ddd4d1e9765f38db4a6c8d3faddc9d53aeed660eac4ca164` |
| Same command → `reports/phase2-readonly-telemetry-validation-tests-v2.log` | 10 tests pass, including archive-only compatibility and CLI | `93c66ad524f5ce2759cdeebf5f8d5af11022ce27238ebb7bc7fb5cad6c4aba8c` |
| `CARGO_BUILD_JOBS=4 ./tools/check.sh` → `reports/phase2-readonly-telemetry-validation-check-v1.log` | Pass, 185 reporting tests | `6d9aca66c03eda283e9934b23fdd988dfa94bdc533f839cbdbe65c9259420fbd` |
| Focused command above → `reports/phase2-readonly-telemetry-validation-tests-v3.log` | Final 10 tests pass; reject nonfinite JSON constants even in metadata | `30d0ad9d7d4fbc128111b2d82529435ee1df1e7e8bb192ee23bf7336fb96d28e` |
| Default check above → `reports/phase2-readonly-telemetry-validation-check-v2.log` | Final pass, 185 reporting tests | `6f71cae10865454574de270f30e5e47c13cd2985b83d01b215db70f9322881ee` |

The separate retained CPU-only inspection result is
`reports/phase2-readonly-telemetry-validation-archived-cpu-v1.log`, SHA-256
`402936e6a184aeeaf81f9df5ae77792c10ee4ac79556595f96e0dc27a92b631c`;
its archive extraction/digest/check logic is now in
`test_archived_v2_cpu_in_fresh_interpreter`. No production, GPU policy, oracle,
artifact/workload schema, pin or numerical-tolerance change occurred. Linux remains 72/78. Next: frozen
manifest/entry, runtime library/injection/process evidence and cross-bind the
collector/supervisor/telemetry/analyzer results before readiness review.

## Owned-child runtime observation — CPU-only validation

`tools/bench/observe_readonly_process.py` supplies an optional bounded-capture
callback. It inspects only a direct child still held waitable by this collector;
no ptrace, signalling, injection changes, GPU queries or unrelated-PID inspection.
The unchanged native benchmark binary is not recompiled. The callback is propagated
through `run_bounded(observe_child=...)`, `run_sequence(process_observer_factory=...)`
and `coordinate(process_observer_factory=...)`; a future frozen launch entry must
construct one observer per job and enable it for **all 12 conditions**.

At most one observation is attempted per 50 ms, including when a trace emits rapid
stdout chunks. `/proc` reads are bounded (4 MiB maps, 64 KiB environment/command/stat),
with at most 3,000 samples/observed races, 8 MiB distinct mapping inventory and a
16 MiB serialized result cap. These are logical/read caps, not RSS or filesystem
quotas. Each valid sample checks the kernel executable stat against prelaunch
identity, exact argv/environment equality, process start/parent identity and CPU2
workload affinity. A declared taskset launcher may exec the declared workload;
other stable images fail. Changed exec images/argument data, not-yet-installed
startup fields and proc entries/VMAs disappearing before exit becomes waitable are
counted as incomplete samples, never combined into a guessed snapshot. At least
one later complete workload sample remains mandatory. Missing fields after an
earlier valid sample likewise provide no new evidence; whole-lifetime coverage
remains false. Access denial before the first complete workload sample is also
recorded as not-ready, not permission verification; persistent denial cannot
qualify. After workload observation, access denial fails capture. Stable mismatches
and limits fail capture and trigger owned-child cleanup; a subsequent child exit
cannot erase a detected mismatch. Fixed guard failure reasons are retained without
serializing environment values or unexpected command lines.

Executable file mappings retain kernel-map device/inode, map-file symlink checks
and current pathname stat device/inode. **Initial real CPU-child tests failed**:
this host reports device 28 in `/proc/PID/maps` but device 29 via pathname/executable
stat for the same named inode. A read-only self-process check established that
map_files symlink readlink works, while stat of the underlying map_files target
returns `PermissionError`. No privilege escalation, permission or system-policy
change was attempted. The implementation now explicitly preserves both devices,
requires the kernel map symlink to match the recorded path and the pathname inode
to match the map inode, and compares prelaunch identities with pathname metadata.
It does **not** equate the two device namespaces or claim direct mapped-byte access.
Unsupported inode translations, deleted/escaped paths and changed identities fail.

Plain conditions fail if the known injection image is observed. Traced conditions
must observe its matching prelaunch identity during the workload, plus declared
required library identities. No observation is whole-lifetime absence proof; short
loads/unloads and anonymous/JIT executable regions are not byte-qualified. Mapping
inventories preserve non-file executable labels separately. File hashing is not
done during the workload/full telemetry; the final collector must bind observations
to prelaunch and **post-collection** file checks and receipt PIDs. Sample overhead
is part of this diagnostic's conditions, not acceptance timing.

With the hook enabled, captures write `NN-CONDITION.process.json` exclusively. A failed
observation stops the sequence even if the child exited successfully; there is no
replacement/retry. Hash/report qualification and whole-run supervision remain
separate. Results leave `launch_provenance_verified`, `mapped_file_bytes_verified`
and `whole_lifetime_mapping_coverage` false. The per-job files and enabled-hook
flags must be cross-bound by the final analyzer/manifest before collection readiness.

Eleven tests use Python children and a read/execute-mapped copy of `/usr/bin/true` as
an injection **surrogate**, never actual CUPTI. They cover plain/traced mapping
presence, rate limiting, exact environment/command/image/affinity mismatches,
unknown PID ownership, read/sample/inventory bounds, malformed/deleted/ambiguous
maps, distinct device identities, permission failure, changed injection inode,
missing required libraries, all 12 fresh sequence children, rejecting a factory
that silently returns None, incomplete proc/exec samples without promoting missing
evidence, and stopping after an otherwise-successful child's failed observation.
All children are reaped. This does
not establish real CUDA process observability or abort behavior.

| Command / retained local log | Result | SHA-256 |
| --- | --- | --- |
| `python -m unittest discover -s tools/bench -p test_observe_readonly_process.py -v` → `reports/phase2-readonly-process-observation-tests-v1.log` | Initial mapping-device assumption fails, 5 subcase failures | `06bebb5fed7d5efd025eccfb97a02db9fcb511cf9282042e5515710f995b2383` |
| Same command → `reports/phase2-readonly-process-observation-tests-v2.log` | 7 tests pass after explicit device separation | `11f1ced10e97a1f9ea80618821ea4160c6e8204902e12d11b24971a3bc0dec95` |
| Same command → `reports/phase2-readonly-process-observation-tests-v3.log` | 8 tests pass, including device/permission regression | `22bbe4850b0e8e05e341643172a824381e1a71fce75760f7c74647a726e0c3ae` |
| Same command → `reports/phase2-readonly-process-observation-tests-v4.log` | 9 tests pass, including full fixed sequence | `9184ab9b0bf383f334fd07be54a7d11f168c137db7ab7a5bd509efa923a1cad9` |
| `CARGO_BUILD_JOBS=4 ./tools/check.sh` → `reports/phase2-readonly-process-observation-check-v1.log` | 192 reporting tests pass | `33fc60544e7f3976cd9a3b905c9297d97181b3325be7bbfda05f2c17798ff2f2` |
| Same command → `reports/phase2-readonly-process-observation-check-v2.log` | 193 reporting tests pass | `99b00f17a78a7ff4f9a8100a5029b231c45145d9de363cdc4f2e4457bb09cda1` |
| Same command → `reports/phase2-readonly-process-observation-check-v3.log` | 194 reporting tests pass | `46fbfaa133c070659e3fdb96c045ae9e90b5347cb740b890550afa81eced94d5` |

Follow-up review found that an enabled factory could return None and silently
skip observations. A mocked no-child regression failed before the fix; every enabled
job now requires callable observe/finish methods before launch. Log
`reports/phase2-readonly-process-observation-factory-red-v1.log` (command above with
`-k enabled_factory`) has SHA-256
`92672d7ce7d5d5f8c5158ab481bce003567dc2d224e479319a6bf670b54495ad`.

A subsequent focused run failed intermittently despite a passing default suite;
that was **not** accepted as sufficient evidence. Fixed batches of five CPU-only
12-child tests (`... -p test_observe_readonly_process.py -k complete_sequence -v`)
exposed incomplete mappings/environment during exec, then disappearing proc entries
before exit became waitable. Stable image/environment mismatches still fail; only
unverifiable partial samples are now explicitly counted without being promoted.
Deterministic tests keep proc data unavailable for an entire child and verify that
it cannot pass observation qualification. A final fixed five-case batch passes.
All 15 batch logs (four failures across the first two batches) are indexed by
`reports/phase2-readonly-process-observation-race-index-v1.json`, SHA-256
`a9d8730d290ec79850a4fcb8dc826710570814f8b365133fd0a91104392a042c`.
These are fixture/supervision tests, not measurement retries or new GPU samples.

| Retained follow-up log (same focused/default commands above) | Result | SHA-256 |
| --- | --- | --- |
| `reports/phase2-readonly-process-observation-tests-v5.log` | 10 tests, intermittent full-sequence failure | `50158736c87c46ef0a8f7fc6ca835b3fbb78899a70f2039c230f9f700d054db6` |
| `reports/phase2-readonly-process-observation-check-v4.log` | 195 tests pass; not enough to resolve intermittent failure | `24f00c91d1c5dabf9c39263c344a4ef0550551044774ef4d323580ad13c9884c` |
| `reports/phase2-readonly-process-observation-tests-v6.log` | 11 tests pass; separate fixed batch still exposed exit race | `e1c3193ea81418057fa49fbe9d2c807c68b08665efe41c0ddcbac091e6b574dc` |
| `reports/phase2-readonly-process-observation-check-v5.log` | 196 tests pass; exit handling still under review | `4514395b96bdfbda616e74343e7a8f4395d4e88f080df019f280d5015757750b` |
| `reports/phase2-readonly-process-observation-tests-v7.log` | 11 tests pass before final guard review | `431c35a8bb5c95660f0ed1f637ec5f3bec361d8919f07a296fa42284a8cb18d5` |
| `reports/phase2-readonly-process-observation-check-v6.log` | Default check passes, 196 reporting tests | `de5b59bac81a1bdfd6ab0866622929db360040d5b0f2fa0f395b19803d223159` |

Final guard review additionally reproduced that catching **any** error after a
child exit could erase an already-detected environment/injection mismatch. That
catch-all was removed; the no-masking regression is retained in the environment
mismatch test. A later focused run exposed startup `PermissionError`, although its
full suite passed. Startup denial is now explicitly not-ready and counted; a later
complete workload observation is mandatory, an entirely denied child fails
qualification, and denial after a valid workload sample still fails capture. No
permissions changed and no unavailable data was promoted. The last fixed five-case
batch passes; all 20 batch logs (including the four earlier failures) are indexed
in `reports/phase2-readonly-process-observation-race-index-v2.json`, SHA-256
`b169cd913a01c64dde2e8e69074342cecefa487c705c1aee3a2bfb1840ade160`.

| Final review log | Result | SHA-256 |
| --- | --- | --- |
| `reports/phase2-readonly-process-observation-late-exit-red-v1.log` (focused command with `-k mismatch_cancel`) | No-masking regression fails before catch-all removal | `111c6e91991fe298463a5417f141193fd2856c601b3ab724e3ced019ca7e2b13` |
| `reports/phase2-readonly-process-observation-tests-v8.log` | Startup permission failure retained | `124b55c56fbfc93b107227bd281a12c5e53f3ddf1f2d61bda7353c8dfb5cd14e` |
| `reports/phase2-readonly-process-observation-check-v7.log` | 196 pass, insufficient alone to resolve startup failure | `c4f17d02a2fd12297a931843ea78c122731806782226acb09394d6876db3a94f` |
| `reports/phase2-readonly-process-observation-tests-v9.log` | 11 pass, including no-masking and denial regressions | `c005360597be04c0802486b57998e6c23e4d2ccef4e3c3719624d478c9c05dc4` |
| `reports/phase2-readonly-process-observation-check-v8.log` | Default check: 196 reporting tests pass | `ca3dd541c508f37770c506c26c9baaf727f6f510c3091aba7db16aa2a7e3ca51` |
| `reports/phase2-readonly-process-observation-tests-v10.log` | Final 11 pass; required-library iterables are snapshotted once, never silently consumed before enforcement | `69c1fd980d8f74dc7ac2372f4ee048c49054df6c923a4c088477445d5dd2bc50` |
| `reports/phase2-readonly-process-observation-check-v9.log` | Final default check: 196 reporting tests pass | `b2a09e0c6b1a28bd88b78703f250e02cb8b889f6075e3defae6883b46d9dd297` |

Retained local cause inspections: `phase2-readonly-process-observation-debug-v1.log`
(`0bd1505a64ffae544495ed42070381e97177d970e09d6194391f3cc525762dbe`),
`-debug-v2.log` (`e7669866a911315e507485ef6b003d0dea600a44f888d640179d55a31af1ad76`),
and `-mapfiles-v1.log` (`0aca0a2cbfbc0950f009831fa20bcb28d388cdf7f1245366e4281b21dd34035b`),
all under `reports/`. Only fake child/self-process metadata was inspected. Linux
remains 72/78; no real GPU/model collection or production/policy change occurred.

## Post-collection process and file binding

`tools/bench/verify_readonly_processes.py` adds two import-only stages. It does not
launch programs or infer collection readiness:

- `validate_processes(jobs, observations, prepared, libraries)` checks all twelve
  ordered records/PIDs, successful observed receipts, declared CPU2/taskset commands
  and isolated environments, loader/prepared target agreement, complete process
  status, sample ordering/rate/inventory, recomputed mapping digests and consistent
  file identities across processes. Every workload must map its declared image and
  required native libraries. Traced records must observe the declared injection
  image; plain records must not. Boolean flags alone cannot substitute for mapping
  contents. Unused maps, stronger unsupported proof claims, changed identities,
  incomplete counters, missing observations and incompatible metadata fail.
- `verify_files(binding, prepared, libraries, library_roots=...)` rechecks all
  prepared input/source identities, separate declaration identities, loader/native
  library identities and observed executable paths. Known paths must retain their
  prelaunch inode, size and SHA-256. Additional observed libraries must be canonical
  `.so` paths under explicitly reviewed library roots (defaults `/usr/lib` and
  `/opt/cuda/targets/x86_64-linux/lib`), regular files with ELF magic and the observed
  pathname inode/device. They remain **post-only**; no historical/prelaunch pin is
  invented. File magic is not full ELF validation or executable qualification.

Postflight has a 256-file/8 GiB aggregate hash-read budget. The whole path/type/
identity/size inventory is checked before hashing any file; mapped-file ELF magic
is checked before each file's hash. Symlinks, replaced files, unreviewed extra paths,
invalid pins and budget violations fail. These are new **post-collection read
bounds**, not changes to capture quotas, the 600-second workload deadline, model
numerical tolerances or acceptance limits. No library bytes are copied into output.
Kernel map devices remain separate from pathname devices. Current file hashes do
not establish immutable mapped bytes, unsampled load/unload behavior or historical
library parity; `mapped_file_bytes_verified` and `launch_provenance_verified` stay
false. Process/receipt consistency does not independently authenticate the producer.

**Call-order enforcement is still pending in the frozen launch entry:** invoke file
hashing only after the independent supervisor and full telemetry have completed,
never between jobs or during the remaining sampler window. This helper deliberately
has no collection CLI or automatic invocation. Receipt/observation clock bracketing,
supervisor/collector/telemetry raw-hash bindings and final analyzer composition also
remain separate mandatory checks. Expected preflight declarations must come from
the reviewed launch path, not from the supplied observation files.

Nine tests use synthetic twelve-job records and temporary ELF-magic fixtures. They
cover explicit post-only classification, all predeclared source/declaration files,
changed same-inode/same-size bytes, replaced files/symlinks, forbidden extra paths,
read budgets, invalid pins/non-ELF data, PID/order/receipt/environment disagreements,
map hash/inode/cross-process changes, missing native libraries, timing/affinity/
counter errors and injection flags that disagree with maps. A Linux-only fresh-
interpreter test also collects **12 actual CPU-only child observations** and passes
them through the reader and postflight hashes. Its loader declaration is synthetic;
Python children and read/execute-mapped `/usr/bin/true` copies are not native codec,
CUPTI, loader-probe or tensor qualification. No model/GPU benchmark was run.

| Command / retained local log | Result | SHA-256 |
| --- | --- | --- |
| `python -m unittest discover -s tools/bench -p test_verify_readonly_processes.py -v` → `reports/phase2-readonly-postflight-tests-v1.log` | Initial 8 tests pass | `3dea4494e65d0e8287a2de967797e0265d88eb7404814e4525e062d8b6bbcb82` |
| Same command → `reports/phase2-readonly-postflight-tests-v2.log` | 9 pass, including real writer/reader compatibility | `9bd2f2fb33f3113e24b34f7031b6da464cc312e16177da44af4434497eee2aae` |
| Same command → `reports/phase2-readonly-postflight-tests-v3.log` | 9 pass, including typed observer CPU guard | `3e610e9508536fc5653a26f2eefc9582448913102c977d0fa423b4fef9a83532` |
| Same command → `reports/phase2-readonly-postflight-tests-v4.log` | Final 9 pass, including separate declaration rechecks | `ac91404b07533dc5b355e81a273f0e305d28206e3c127770a2f84337be45eef7` |
| `CARGO_BUILD_JOBS=4 ./tools/check.sh` → `reports/phase2-readonly-postflight-check-v1.log` | 205 reporting tests pass | `f7df8bb9fb8ba596de22707474b6343dce49934443c1f3f7eccb833d3c7df4b9` |
| Same command → `reports/phase2-readonly-postflight-check-v2.log` | 205 reporting tests pass | `5c56f9f6ccac948f4361e7865acff95a82b007f211f77dd312e39a8c3b4d2817` |
| Same command → `reports/phase2-readonly-postflight-check-v3.log` | Final default check: 205 reporting tests pass | `782f85e4edd94b5e4649d58a3c84206377c358ff1f079903aeae2b139538e635` |

Linux remains 72/78; no production, policy, oracle or pin change. Next is the frozen
manifest/entry and lifecycle/time/raw-file cross-bindings, then readiness review
and fresh availability—not a new benchmark or automatic clock fallback yet.

## Launch blockers and outcome rules

This document is **not executable launch readiness**. The coordinator, telemetry
lifecycle/validator, sampled process observer and postflight verifier are tested
prerequisites. Frozen launch manifest/entry, postflight call-order enforcement,
process/receipt timing, supervisor/collector/telemetry/analyzer cross-bindings,
mapping-limitation review and remaining integrated failure tests are pending.
No new capture exists to qualify. The user has chosen the read-only investigation;
before collection, obtain a fresh quiet 15-minute window after the tested tooling,
exact frozen manifest and identity checks are ready.
No extra profiling permissions, system installation or CUPTI replacement may be
made implicitly. Stop and ask if existing local tracing cannot be used safely.

Possible outcomes: incomplete/unattributable capture; spread not reproduced under
instrumentation; or spread associated with specific measured kernel/API intervals.
Instrumentation can perturb allocation and scheduling enough to suppress or create
the spread. Three processes per condition cannot guarantee both clusters occur.
Do not keep sampling until they do, filter a process, transplant results, or infer
causality from association alone. If inconclusive, report exactly what remains
unobservable before invoking the conditionally approved clock fallback; its
policy/controller, supervisor, permission and fresh-window prerequisites still
apply. Changes beyond the two approved proposals require renewed approval.
Every outcome leaves Linux
**72/78** pending a separate full acceptance protocol with unchanged criteria.
