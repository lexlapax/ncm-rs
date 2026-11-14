# CUDA convolution candidate: Linux gate

Candidate `b0a0058` passes the fixed controlled Linux acceptance audit:
**78/78 comparisons**, with unchanged limits, pinned Python reference and
trained qualification. The predeclared initial 45-process matrix passed 75/78;
its batch-1 codec decode p95 against previous Rust and two BERT decode process
envelopes failed. An independent, unconditional 135-process control covering
every workload and scope passes all 78. Nine fresh lifecycle processes completed
with identical output IDs. This control does not erase the initial failure or
prior failed gates.

For the controlled comparison against accepted Rust `3307dff`, the previously
problematic codec cells now pass:

| Cell | Median ratio | p95 ratio | p95 95% interval upper |
| --- | ---: | ---: | ---: |
| Batch-1 decode | 0.90813 | 0.91133 | 0.93255 |
| Batch-2 encode | 0.92189 | 0.87282 | 1.02774 |
| Batch-2 decode | 0.91814 | 0.91945 | 0.92672 |
| Batch-4 encode | 0.93217 | 0.93909 | 1.03075 |

The 151 initial and 450 control GPU samples each have median utilization 3%,
including benchmark work; the process monitor also sampled nonzero terminal
activity. These observations do not establish exclusive GPU access or a cause
for any individual outlier. No clock or power policy changed.

[Raw matrices, control, lifecycle, telemetry, audit and source verifier](raw/conv-bias-linux-gate-v1.tar.xz)
SHA-256 `8000540d220149b58f951679e1a05dd22d752f4b5ef36e19a21d373c73b90927`.
The controlled audit SHA-256
`d1fcc136cf9e95fb8b0dff86941351d6ab5daa0d0f49ac5f2ef7a9ae5c677589`
reproduces byte-for-byte from a clean `b0a0058` export and this archive. All
120 native report inventories and 180 shared Python report copies match clean
candidate/baseline exports. The [candidate report](conv-bias-cuda-v1.md)
records the full CUDA check, 642-trace trained qualification and retained
failed check attempts. Apple validation is the next gate.
