# TinyStories candidate: Linux gates remain open

Candidate `fb6c75e` passes the full CUDA check, tokenizer suite and four real
asset probes. Its unchanged codec benchmark is byte-identical to the previously
trained-qualified candidate. In the first predeclared full gate, the initial
matrix was 75/78 and the unconditional all-workload control was **71/78**.
Five controlled decoder-block p95 intervals and two BERT decode median
intervals exceeded unchanged limits. TinyStories decode and skip-special
controlled median ratios versus Rust were 0.85892 and 0.86113, and both passed.

The GPU monitor saw nonzero compositor and terminal activity; the user then
confirmed desktop activity paused. A separately predeclared full quiet-window
gate yielded **74/78 initial and 75/78 controlled**. TinyStories decode and
skip-special again pass, with controlled median ratios 0.86433 and 0.86085.
Three codec p95 upper bounds exceed 1.10:

| Cell / comparator | Controlled median | Controlled p95 | p95 interval upper |
| --- | ---: | ---: | ---: |
| Batch-1 decode / Rust | 0.90757 | 0.74738 | 1.11749 |
| Batch-2 resident block / Rust | 0.89426 | 0.94781 | 1.53059 |
| Batch-2 resident block / Python | 0.71882 | 0.72463 | 1.17724 |

The first matrix/control recorded 160/474 GPU samples, median utilization
10%/9%; quiet-window matrix/control recorded 156/457 samples, median 9%/6%.
Compositor and terminal processes had nonzero GPU samples in both windows.
These observations include benchmark work, are not proof of GPU exclusivity,
and do not establish the cause of outliers. No clock or power policy changed.
All initial failures, controls and the prior 78/78 candidate remain retained;
the current Linux gate does not pass, so Apple testing must wait. Do not rerun
this unchanged candidate based only on these results.

[Raw matrices, controls, plans, telemetry, lifecycle, quality and audits](raw/tinystories-linux-gates-v1.tar.xz)
have SHA-256 `685fbf0af4607e557ce896d3a4ab9bbd77befafddd596c15dc7c9ea85fda2acf`.
The quiet controlled audit SHA-256
`8e6aeec92101f3af609174de27b303b376e118bc6e7d1f3a575acfb6b510c714`
reproduces byte-for-byte from a clean `fb6c75e` export. In each gate, all
120 native inventories and 180 shared Python report copies match clean source
exports. Each lifecycle capture has nine matching-output processes.

## Targeted decoder-block profile

One separate, bounded CUPTI process profiled only the batch-4 resident block
(1 first call, 5 warmups, 30 trials). All 36 source-assumed windows parsed;
activity attribution remains incomplete and no dropped records were reported.
The final measured trial contained 22 kernels: two copy kernels, six adds, two
layer norms, two RMS norms, one GELU, one softmax, one scalar affine and the
matrix-multiply kernels. Its 55 event-creation calls and 27 async allocations
include safety and library bookkeeping. This is substantially fewer copy
kernels than the seven seen in an older retained block profile, but profiles
are instrumented and are not a controlled performance comparison. Median wall
was 0.30425 ms and none of the 30 wall trials exceeded 1.2 times their median,
so this sample does not explain the failed p95 intervals. Output identity
fields in all 12 scopes and recorded first logits match an uninstrumented
candidate report; this is not a replacement for trained qualification.

[Diagnostic plan, runner, receipt and aggregate](raw/tiny-cuda-block-profile-v1.tar.xz)
have SHA-256 `f8b834949366f076ecb7a82b189b09f59d70e995110d745c634c3893fb77631e`.
The 3,946,377-byte raw capture remains local at
`reports/phase2-tiny-cuda-block-profile-v1/capture.txt`, SHA-256
`55568200e39d2897c73c74fd1f0c82210a7319613227569b599c0a8ad0b27327`.
The trace identifies potential fusion work but does not justify disabling
cross-stream event tracking or changing CUDA precision.
