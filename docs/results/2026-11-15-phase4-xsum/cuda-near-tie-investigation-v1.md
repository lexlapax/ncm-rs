# XSum CUDA near-tie investigation

The full quality run stopped at a Rust-only FP32 tie; the affected Python
batch has a unique winner and reproduces the saved reference. See
[the progress record](rust-full-progress-v1.md). This investigation has not
found a complete fix and does not change the exact-ID quality rule.

A singleton retry selects code 3,690 instead of Python's 3,182 and changes
three codes and five raw output IDs. Replacing Rust's cuBLAS libraries with
Python's in the process library path still produces the tie. Recomputing only
the final head in FP64 also favors 3,690 on both captured Rust and Python
inputs; this is not a full FP64 model or a basis for changing the reference.
None of these approaches is an acceptable tie resolution.

We replayed the saved 32-row batch and compared individual stages. Its first
transformer-block input is bit-identical, but layer-normalization values
first differ there. On the same captured input, a diagnostic CUDA adaptation
of the pinned PyTorch vectorized Welford layer normalization produced equal
values for all 12,681,216 elements. On identical normalized input the existing
Q/K/V projection produced bit-identical output for all 38,043,648 elements.
For captured Q/K projection inputs, diagnostic RMS normalization with
separately rounded squares and sequential multiplication matched all
25,362,432 Python output bit patterns.

The existing rotary sine and cosine tables differ from those computed by the
pinned Python implementation: among 16,512 entries, 2,210 cosine and 3,704
sine bit patterns differ, with maximum absolute differences approximately
3.05e-5 and 3.03e-5. A native diagnostic that used the captured Python tables
and separately rounded multiply/add operations matched all 25,362,432 captured
Q/K rotary output bit patterns. The tables are diagnostic inputs, not a new
production artifact or an inference-time Python dependency.

On identical captured attention inputs, matching PyTorch's separate Q/K
scales and final softmax division produced bit-identical output for all
12,730,368 values. The ordering is grounded in the pinned PyTorch
[attention implementation](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/transformers/attention.cpp),
[softmax implementation](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cuda/SoftMax.cu)
and [layer-normalization kernel](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cuda/layer_norm_kernel.cu).
These operator observations establish behavior only on the captured inputs,
not equivalence across shapes or hardware.

Prescaling Q/K alone, and prescaling plus softmax division, eliminated the
explicit failure in the affected batch but selected 3,690, changing three
codes and five raw IDs against the saved Python batch. Combining the
layer-normalization and RMS diagnostics with attention had the same result.
A further exploratory build combining captured Python rotary tables and
separate rotary operations failed to link because cuDNN was not on its linker
path. The following run accidentally reused the previous executable, so it
does not establish a result for that rotary candidate. The earlier completed
candidates are rejected as complete fixes. No production source or artifact
was changed by the temporary diagnostic builds.

CUDA diagnostics, input captures, scripts, build logs, candidate sources and
results are retained locally in `reports/phase4-xsum-tie-probe-v1/`; this
ignored directory is not published in Git. For example, the final rotary
operator diagnostic was built and run as follows (the run needs an accessible
CUDA device):

```sh
/opt/cuda/bin/nvcc -O3 -arch=sm_86 reports/phase4-xsum-tie-probe-v1/rope-pytorch-order.cu -o reports/phase4-xsum-tie-probe-v1/rope-pytorch-order
reports/phase4-xsum-tie-probe-v1/rope-pytorch-order
```

Next compare subsequent transformer operations and levels on the same
captured batch to find where the remaining decision drift enters. Any
production fix must preserve the upstream algorithm, be tested on the actual
failed batch and broader quality set, and retain the required speed and memory
advantages. Full Rust generation, quality scoring and both XSum runtime gates
remain pending.
