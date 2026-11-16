# Experimental XSum reference-order CUDA path

The explicit `xsum-inference-v2` artifact profile uses a strict CUDA/cuDNN
backend with reference-order FP32 normalization, GELU and rotary operations.
It also uses a 1 MiB cuBLASLt preference for 768-by-768 projections. The
existing v1 profile keeps its previous backend behavior. Backends that do not
implement the v2 reference mode reject it explicitly; CPU and MLX are not v2
fallbacks. The CUDA normalization reduction is adapted from PyTorch; its
upstream license is retained in `crates/ncm-backend-cuda/third-party/`.

On 259 saved Python rows (eight full batches and the three-row final batch),
the native CUDA path matched all 51,800 sampled codes and 12,432 raw IDs,
including the batch that stopped the earlier run. That screen is not the full
quality gate. A source- and linked-library-bound two-shard driver smoke matched
all 25,600 codes and 6,144 IDs across 128 rows. Its shard manifests are
`8250d6b1b41d991057327ab8b205a84a7f2eabd70485c8770d21f0b474c49539` and
`375cec5e2d1b4c840be1fb2ee6ef9b78af188971653d242238c9a8f0a9bbf5dd`.
The previously failing batch also matched using the system cuBLAS libraries.
A small independent PyTorch CUDA fixture covers reference layer normalization,
RMS normalization, group normalization, GELU and rotary embedding; all
values matched on the RTX 3060 Ti. The fixture SHA-256 is
`3d5d9b3ff34efded200ef3ac8d1d180cec5b402de57f784c79d1306b87e2ea9a`.
It was reproduced byte-for-byte with `tools/reference/xsum_reference_ops_fixture.py`.

A paired exploratory benchmark ran one fresh process per runtime, five warmups
and 30 measured trials per batch, using the same input and checking all codes
and IDs. The measured median generation latency and maximum whole-process RSS
were:

| Batch | Rust median | Python median | Rust peak RSS | Python peak RSS |
| --- | ---: | ---: | ---: | ---: |
| 1 | 154.99 ms | 188.93 ms | 1,057.95 MiB | 1,861.50 MiB |
| 4 | 562.50 ms | 681.63 ms | 1,063.50 MiB | 1,860.38 MiB |
| 32 | 4,384.90 ms | 5,331.07 ms | 1,060.55 MiB | 1,864.34 MiB |

Python explicitly collects unreachable batch objects outside the generation
measurement; the harness also records cleanup separately. These are short
single-process screens with sampled activity, not the final multi-process
performance gate. Raw results and source, binary, artifact and library hashes
are retained in `reports/phase4-xsum-native-performance-b1-b4-v1/` and
`reports/phase4-xsum-native-performance-b32-v1/`. The CUDA group-normalization
source was subsequently reformatted; the new PTX matches the measured PTX
after normalizing the two renamed shared-memory symbols. The activity collector
recorded 66 samples for batches 1/4 and 336 samples for batch 32, with no GPU
query failures; those device-wide counters cannot rule out other GPU activity.

The source-bound quality driver and benchmark collector now accept `--profile
v2`; their default remains v1. Profile, source, binary, artifact and library
hashes and actual linked CUDA libraries are bound into each new run. Existing v1 partial receipts remain
historical evidence and cannot be resumed under the modified source manifest.
The full v2 generation, held-out/full quality comparison, repeated performance
measurement and both XSum runtime gates remain pending.

The full local `tools/check.sh --cuda` passed, including all 18 real-device
operator cases; its log is `reports/phase4-xsum-native-cuda-check-v1.log`.
The additional CUDA/cuDNN reference-operator fixture was run explicitly with
`cargo test -p ncm-backend-cuda --features cuda-cudnn --test xsum_reference -- --ignored`
and passed. Strict CUDA/cuDNN Clippy, the scorer and library-provenance unit
tests, and the 128-row source-bound smoke also passed. An initial strict lint
attempt caught two unnecessary `vec!` allocations in the new test; those were
replaced by arrays before the successful rerun. The checks use local hardware
and do not by themselves establish full model quality.
