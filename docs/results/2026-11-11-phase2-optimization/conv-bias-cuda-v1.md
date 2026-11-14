# CUDA convolution bias layout candidate

The CUDA/Candle backend previously materialized a contiguous copy after
transposing convolution output from BCL to BLC, then added bias. The candidate
lets Candle's broadcast addition read that transposed view directly. This
removes an intermediate copy without changing CUDA event tracking, stream
synchronization, numerical tolerances, or the exposed backend API. A regression
covers both regular and transposed convolutions with two batches, multiple
channels and nonzero bias on CPU and real CUDA.

A predeclared, counterbalanced diagnostic compared three fresh processes of
each binary on CPU 2, with five warmups and 30 trials per scope. The table
shows the ratio of the median of candidate process medians to the median of
baseline process medians. It is a diagnostic, not the acceptance gate.

| Batch | Encode | Decode | Reconstruct | Resident decoder block |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.9680 | 0.9471 | 0.9725 | 0.9947 |
| 2 | 0.9683 | 0.9520 | 0.9661 | 0.9987 |
| 4 | 0.9619 | 0.9501 | 0.9655 | 1.0009 |

The six reports agree in input/output identity fields across all 12 scopes;
recorded first logits match exactly. This does not replace trained qualification
or a whole-matrix process-envelope audit. A fresh CUDA capture passed the
existing pinned qualification: 642 traces and 2,500 decisions, zero ID
disagreements and two near ties. Its native trace hash matches the previously
qualified native trace. The CPU and CUDA component tests also passed.

The first project-check attempt stopped at Rust formatting; formatting was
fixed. The next attempt ran 483 Python tests and failed three clock-evidence
fixture tests because shell umask `022` allowed non-private parent directories.
Those tests passed when rerun with process-local umask `077`. The full `umask 077; RUSTC_WRAPPER= ./tools/check.sh --cuda` rerun passed,
including all 483 offline tests and the real-GPU checks.

Linux performance acceptance and the later Apple gate remain pending. Previous
failed audits remain retained. No GPU clocks or system policy were changed.

[Diagnostic reports, qualification and check logs](raw/conv-bias-cuda-v1.tar.xz)
retain the plan and both failed check attempts. Benchmark input/output source
inventories for the three candidate processes match the current source. The
raw trained capture remains local; its hash is in the qualification report.

## Predeclared Linux gate

Before collection, run the unchanged five-workload, three-runtime matrix
with three fresh processes per runtime and workload, then an independent,
unconditional nine-process-per-runtime control for every workload and scope.
Run nine lifecycle processes and apply the existing acceptance limits and
qualification to the full control; retain the initial audit and all failures.
Use CPU 2 for workloads and CPU 0 for the observer, continuous GPU activity
telemetry and a 15-second CPU sample. Do not select, retry or extend based on
results. The established accepted Rust baseline is `3307dff`; Python and all
workload definitions remain pinned. Apple execution follows only after Linux
passes.
