# Decoder quiet-window Linux gate — 74/78

After the user quieted the Linux desktop and a separate diagnostic observed
fewer timing spikes, a fresh gate was declared before collection: three
counterbalanced processes per runtime for all five workloads, followed
unconditionally by nine independent processes per runtime for **all five**
workloads and every scope/load. The plan SHA-256 is
`b5125011b5a55c0141a557cf4a5fbc3c0e52b7e0a5a2a15288d3193ce6d113c9`.
Candidate `9f8773e` has the code of `211df63`; both previous and candidate
native benchmark binaries match the [prior failed gate](decode-final-linux-gate-v1.md)
byte-for-byte. The pinned fixtures, comparators, tolerances and limits are
unchanged. A fresh nine-process lifecycle capture was collected.

The initial matrix passes **77/78**, with batch-1 codec decode versus Rust
missing the process envelope. The predeclared all-workload control passes
**74/78**. Four codec cells versus Rust miss the 95% p95 interval upper limit
of 1.10, despite passing controlled median and p95 point estimates:

| Cell | Controlled median ratio | Controlled p95 ratio | p95 interval upper |
| --- | ---: | ---: | ---: |
| Batch-1 decode | 0.95823 | 0.95074 | 1.17282 |
| Batch-2 encode | 0.95890 | 0.96759 | 1.18122 |
| Batch-2 decode | 0.96746 | 0.85327 | 1.12851 |
| Batch-4 encode | 0.96912 | 0.96348 | 1.12107 |

Individual trial distributions contain occasional large values in both Rust
implementations. An offline inspection of the four failed cells compared
each wall trial with its matching CUDA stream span, normalized by that
process's respective median. Of 65 wall trials above 1.2 times the process
median across both Rust implementations, **all 65** also had a stream span
above 1.2 times its process median. This is descriptive and does not alter the
fixed p95 audit: stream spans include host submission and possible GPU idle
time, so they cannot distinguish CPU scheduling, GPU scheduling or other
causes. The cause is not established. The initial/control telemetry
contains 155/463 GPU samples; reported utilization had median **2%** in each,
with benchmark work included. The process monitor recorded nonzero compositor
and terminal GPU activity; the samples do not prove exclusive GPU access or
identify a per-request cause. No clock or power policy changed. The earlier
71/78 initial and 70/78 controlled results remain published; this quieter
collection does not erase them. The Linux gate and Phase 2 remain open.

[Raw matrix, fixed control, plans, telemetry, lifecycle and audits](raw/decode-quiet-gate-v1.tar.xz)
have SHA-256 `e7419752231b4f77e5340b40c0b2d04ed9d0d2bbd9c5c44969627e8ae2d6cf2e`.
The controlled audit SHA-256
`e4a01609a10af13ecad2bf995f5490069244fd8b29bf5da88173101b50f8544f`
reproduces byte-for-byte, including the four failures, from a clean `9f8773e`
export and the archive. All 120 native inventories and 180 shared Python report
copies match clean candidate or baseline source exports.

## Instrumented scheduler diagnostic

Three candidate-only processes ran a separate diagnostic that reads
`/proc/thread-self/schedstat` around each timed call. This instrumentation
perturbs execution and includes pre-timing synchronization; it is not an
acceptance measurement. Across all codec scopes, 24 measured calls exceeded
their own process/scope median wall time by 20%. All 24 also exceeded their
own median CUDA stream span by 20%. Their excess wall times were at least
**0.281 ms**; recorded run-queue wait during each bracketing interval was at
most **0.031 ms**. A separate short CPU-only self-check with two owned
processes on CPU 2 showed the counter increasing by approximately 299 ms under
known contention.

This narrows the symptom in the instrumented sample: the benchmark thread's
recorded run-queue wait alone does not account for those excesses. It does not
exclude sleeping/blocking, other threads, CPU execution effects, GPU
scheduling or other causes. CUDA stream spans include host submission and
possible idle time. The sample does not repair or replace either failed gate.
[Source, reports, self-check and telemetry](raw/codec-sched-diagnostic-v1.tar.xz)
have SHA-256 `1218447afccb6ce0250b885a6a37a39c47d5f3d52c6566b0c4ab20c74eb44b2c`.


## Targeted batch-1 decode profile

One predeclared, instrumented CUDA process profiled the batch-1 decode scope
(1 first, 5 warmup and 30 measured calls); it is a diagnostic, not an
acceptance run. Among measured trials, the median wall time was 1.771808 ms
and one trial exceeded 1.2 times that median at 3.051365 ms. That trial's
summed kernel durations were 0.851432 ms, versus a 0.851667 ms median. Its
681 `cuEventCreate` calls totaled 1.352633 ms, versus a 0.088584 ms median;
the longest was 1.263400 ms, versus a 0.001367 ms median of trial maxima.
The uncovered part of the kernel hull was 2.152654 ms, versus 0.873529 ms
median; this is not proof of device idle or CPU overhead. CUPTI perturbs
timing and leaves some activity unattributed, so this observation does not
establish the cause of the uninstrumented gate failures.

The local pinned `cudarc` source creates two events for each tracked slice
allocation. Its event tracking protects cross-stream allocation, use and
freeing; disabling it is unsafe without establishing those guarantees for
the full application. No tracking or synchronization behavior was changed.

The profiled report's input/output identity fields matched the uninstrumented
candidate in all 12 scopes, and recorded first logits matched exactly. This
is a limited diagnostic comparison, not full trained parity. There were 36
profile windows and no capture lines reporting dropped records; that does
not prove capture completeness. An initial parser retained an unrelated
block-scope filter and produced no windows; the corrected copy selected the
decode scope. The [diagnostic archive](raw/codec-b1-cupti-diagnostic-v1.tar.xz)
contains the plan, diagnostic source, corrected parser, telemetry and per-call
aggregates. The 41,585,794-byte raw capture remains local at
`reports/phase2-b1-decode-cupti-v1/capture.txt`, SHA-256
`080e99a8fd447955616546aa44a2de291dee16de544b0226dee8aaa5ed984e7f`;
the raw result and full parsed windows also remain local. No GPU policy was
changed. The Linux gate remains 74/78, with Apple validation pending.
