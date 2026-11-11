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

## Launch blockers and outcome rules

This document is **not executable launch readiness**. Offline helpers and a tested
child primitive are only prerequisites; the fixed collection runner, whole-run
supervision, integrated launch/report provenance and remaining tests are pending.
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
