# TinyStories loader candidate Linux gate

Candidate `5200924` passes the initial **78/78** Linux audit and the
predeclared independent all-workload control **78/78**, using the unchanged
pinned Python reference, process-level uncertainty and acceptance limits. Nine
lifecycle processes agree on output IDs. The committed candidate passes the
full CUDA check, all four real tokenizer probes, and the CUDA codec benchmark,
trace and lifecycle binaries are byte-identical to the previously qualified
candidate; its existing 642-trace/2,500-decision qualification is retained.

Against accepted Rust `3307dff`, controlled TinyStories checked-load median
and p95 ratios are **0.86563** and **0.86399**, with p95 interval upper
**0.87246**. Its peak host RSS/HWM ratios are both **0.93978**. The initial
and all-workload control GPU monitors recorded 153 and 452 samples respectively,
with median utilization 3% including benchmark activity; process telemetry is
retained. These samples do not establish exclusivity. No clock or power policy
changed.

[Raw matrices, telemetry, plan, lifecycle, probes, qualification and audits](raw/tiny-borrow-linux-gate-v1.tar.xz)
have SHA-256 `b16be8834fa12e30b5c70d9ac4073461d9caf1f7ab5f2fb0673b3ce2fe1aa87e`.
The controlled audit SHA-256
`50960a48e6e026b377b547fcc3df39096a5408ed27c576114d3d79a9b52500e3`
reproduces byte-for-byte from a clean candidate export. All 120 native source
inventories and 180 shared Python report copies match clean baseline/candidate
exports. The benchmark inventory omits the CUDA fused module and core trait;
a supplemental manifest verifies both source hashes and the compiler depfile
includes them. Earlier failed gates remain retained.

Apple is next: recheck checkout and activity, pull the reviewed pushed commit,
run the full MLX check and four tokenizer probes, and execute the fixed full
Apple gate with unchanged limits. Linux success does not imply Apple success.
