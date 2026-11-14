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
implementations. The cause is not established. The initial/control telemetry
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
