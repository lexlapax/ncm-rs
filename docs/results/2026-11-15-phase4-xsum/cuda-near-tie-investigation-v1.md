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
candidates are rejected as complete fixes. After fixing the linker path, the
rotary combination was built successfully and also changed three codes and
five raw IDs. Its first block matched Python bit-for-bit through the attention
output, including the block input, first normalization, Q/K/V projection,
rotary outputs and attention. The subsequent attention projection output
had 11,556,758 differing bit patterns out of 12,681,216 (maximum absolute difference
1.20e-6). The same difference persisted with Python's cuBLAS libraries and
with the fused-bias projection disabled. The diagnostic does not yet identify
the reason for that projection difference. A subsequent isolated comparison
showed identical projection inputs and weights: PyTorch's fused linear matched
its saved output, whereas separate matrix multiply plus bias reproduced the
Rust differences. cuBLASLt logs show that Python selected a split-K algorithm
with a 1 MiB workspace preference; the existing Rust wrapper selected a
non-split algorithm with a 4 MiB preference. Using the 1 MiB preference made
this particular projection bit-identical, but applying it globally also
changed earlier projection outputs and did not fix the batch.

On identical captured input, a diagnostic CUDA GELU using the same order as
[PyTorch's CUDA implementation](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ActivationGeluKernel.cu)
matched all 50,724,864 output bit patterns. Combining this with the other
reference-order diagnostics and a shape-specific projection preference made
all 12 first-level transformer block outputs bit-identical. It still changed
three codes and five raw IDs on the complete batch. At the next level, the
first input differed only in the generated target portion; the first sampled
embedding was bit-identical but the first reconstructed latent differed in
551,893 of 589,824 bit patterns, maximum absolute difference 2.39e-6. The
subsequent pooled latent differed in 70,168 of 73,728 bit patterns, maximum
absolute difference 5.48e-7. These are diagnostic results on one batch, not a
production fix. No production source or artifact was changed by the temporary
diagnostic builds.

CUDA diagnostics, input captures, scripts, build logs, candidate sources and
results are retained locally in `reports/phase4-xsum-tie-probe-v1/`; this
ignored directory is not published in Git. For example, the final rotary
operator diagnostic was built and run as follows (the run needs an accessible
CUDA device):

```sh
/opt/cuda/bin/nvcc -O3 -arch=sm_86 reports/phase4-xsum-tie-probe-v1/rope-pytorch-order.cu -o reports/phase4-xsum-tie-probe-v1/rope-pytorch-order
reports/phase4-xsum-tie-probe-v1/rope-pytorch-order
```

Next isolate the first reconstruction difference on the same captured batch
and check whether its correction preserves quality and runtime performance. Any
production fix must preserve the upstream algorithm, be tested on the actual
failed batch and broader quality set, and retain the required speed and memory
advantages. Full Rust generation, quality scoring and both XSum runtime gates
remain pending.
