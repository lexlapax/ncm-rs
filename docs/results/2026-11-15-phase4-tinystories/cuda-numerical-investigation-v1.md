# TinyStories CUDA numerical investigation

The original CUDA backend matched synthetic final IDs but exceeded the Phase 2 accumulated-error comparison band (`atol=5e-4`, `rtol=2e-4`) on some reconstruction and generator-logit elements. That band was used as a diagnostic reference, not silently changed into a TinyStories acceptance rule. For 128 development first-level codes, four learned upsampling stages were bit-identical to pinned Python, but 15,708 of 6,291,456 residual outputs lay outside that band; maximum absolute Python/Rust difference was 0.0112305. The initial two-code case also differed in the upsampling stages.

The checkpoint's active first-level embeddings can reach billions in magnitude and are scaled down through the cascade. Against an independent FP64 residual calculation on the same 128 inputs, the pinned Python FP32 output itself lay outside that old accumulated band at 7,697 elements. Rust had lower mean and p99 error versus FP64 but a higher worst error in this sample; it exceeded the Python FP64 error plus the old band at 2,275 elements. This explains why merely calling the old band a pass would be misleading, and does not justify silently widening it. Raw FP64 report: `reports/phase4-tinystories-residual-fp64-v2.json`, SHA-256 `c42ae707d80021b99bdc8923cc741cc01bb927f24dd77db63d249280b1f73652`.

The difference was traced to convolution layout: an equivalent PyTorch `conv2d` using Candle's `[B,C,L,1]` descriptor matched Rust exactly but differed from upstream `conv1d`; using `[B,C,1,L]`, upstream padding and explicit `CUDNN_FMA_MATH` matched all 6,291,456 sampled residual-convolution values exactly. The same upstream-layout cuDNN transposed convolutions matched all four stages in the original two-code case. The direct residual prototype used a 46,792,704-byte workspace and measured 3.048 ms median versus 5.717 ms for the previous production path in separate isolated probes; these are diagnostic measurements, not the full performance gate.

The opt-in `cuda-cudnn` path uses explicit FP32 FMA math, bounds each workspace at 128 MiB and returns errors rather than silently substituting another algorithm. Algorithm metadata is model-owned and capped at 16 shapes; one serialized cuDNN handle belongs to each backend instance and is destroyed with it. Its cross-thread safety relies on exclusive access and context rebinding, consistent with [NVIDIA's cuDNN thread-safety guidance](https://docs.nvidia.com/cudnn/latest/developer/misc/). The tested native cuDNN version was 9.26.0; production requires native cuDNN 9 libraries, not Python inference. Cross-thread model use, handle destruction on another thread, bounded-cache behavior and retry after cache failure were tested locally.

On two synthetic inputs and a separate 16-item fixed-code development trace, the strict path matched all codec codes, final IDs and generator argmaxes. For the 16-item trace, this covered 10,256 codec codes and 2,048 final IDs; every observed tensor fell within the existing Phase 2 accumulated comparison band. Some late generator logits exceeded the more stringent primitive starting band, which is not a blanket end-to-end budget. The two-item and 16-item summaries are `reports/phase4-tinystories-teacher-strict-v1.json` and `...-v2.json`, with SHA-256 `3e3c32e4456bd11c3c95eead60d6a60d7677ff66cd01bc513d15ca5ce74b1440` and `c4b30d425b722acc552d12f0864507c9832b3e785f5cc11dac31d38431a3357a`. The corresponding [stochastic quality retest](heldout-quality-v1.md) meets the frozen criteria. These observations do not by themselves qualify all TinyStories inputs, all cuDNN versions or Apple MLX; performance, hardware activity and full checks remain separate.

For the tested local setup, `.reference/phase4-cudnn-probe-lib/libcudnn.so` points to the native cuDNN library directory installed under the isolated reference environment. The explicit opt-in build and execution environment was:

```sh
CUDA_COMPUTE_CAP=86 LIBRARY_PATH="$PWD/.reference/phase4-cudnn-probe-lib" \
  cargo build -p ncm-backend-cuda --release --features cuda-cudnn \
  --example tinystories_sample_bench --locked
LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  target/release/examples/tinystories_sample_bench artifacts/canonical/tinystories-phase4-v1 1
```

The library search paths above are local to these commands. Other installations should supply a compatible native cuDNN 9 development linker name at build time and its runtime libraries at execution time; the tested numbers do not qualify a different version. The `cuda-cudnn` feature is opt-in, and `CandleBackend::cuda_cudnn` must be selected explicitly by an application. The ordinary `cuda` feature and `CandleBackend::cuda` keep their existing behavior.

## Expanded natural-input CUDA trace

Five independent Rust development sample files, seeds 100–104, supplied 16 rows apiece. Their sampled code sequences were forced into both generators; the corresponding generated raw token IDs were also used as codec encoder inputs. Thus the 80 codec inputs are generated model output, not an independent natural-text corpus, and the generator comparison is teacher-forced rather than a stochastic-seed identity test. All 51,280 encoded codec codes, 10,240 decoded output IDs, and 51,200 per-level generator argmaxes matched pinned Python. All captured floating-point values across 28 named tensors per run were within the unchanged Phase 2 accumulated diagnostic band (`atol=5e-4`, `rtol=2e-4`). This expands the earlier 16-row trace and does not imply exhaustive input or cuDNN-version coverage.

Across 61,440 generator-head and final-decoder argmax rows, 14 generator rows had a Python top-two gap no greater than twice the maximum absolute Python/Rust logit difference in that entire tensor; none changed argmax. This deliberately conservative bound identifies possible near ties, not a prediction that these rows should differ. The minimum observed top-two gap was `3.2424927e-5`. The analysis is reproducible with `tools/reference/tinystories_near_ties.py`; its output `reports/phase4-tinystories-natural-trace-v1/near-ties.json` has SHA-256 `93f341e81cce91727548446b7aa704e2f6bcced92afd2694547bd8d35cca1005`, and an independent invocation reproduced it byte-for-byte.

| Development seed | Python trace SHA-256 | Rust comparison SHA-256 |
| ---: | --- | --- |
| 100 | `bf56325a8cd2b2e25056e86730c6617c9e48115126dbf9885b020d08ee27f6f6` | `097efce794040457804cc95a0a5c8fc4e362792abdc3d2a9af3752576274e748` |
| 101 | `6538f0dca83cea11be1e205426d3edccf32acd2225f857d612a9511ae7fcc8b9` | `d7d78dc08f88eefe7266157f341a2e83d3265fd2de27482e5b506e1db66b89bc` |
| 102 | `4b7e5857ed5d6e77c9dd9295db4e1480ed45fd29ae6125237f831c72cb40a0b1` | `a7cf645c108392a6007273dd6076273b9a9874e434cd8735fcf17a8e0958404f` |
| 103 | `bcd9e06d42df1bd3ad375b06c7b1c0c2ced416fb9d3faae42c93d42450effab6` | `97da206a45b6bec58e6a3f217543b4e8513e283bf9b327838bc092a958d227ec` |
| 104 | `b519ef14a41d6c98031586b03084d2b7ab3055a3fdb6b316a03f82cdaad75ec1` | `d4fb53903e91029cc4afc65a74b2903b54ada2a8c97439545c14411cd12af956` |

Each run used `tools/reference/tinystories_trace.py --sample reports/phase4-tinystories-development-v2/rust-<seed>.json --count 16 --natural-codec-input` with the pinned `--upstream` and `--source` arguments and a new output directory, followed by the `cuda-cudnn` build of `tinystories_teacher_probe` against that trace. The trace script SHA-256 was `b80fe52ea6f6d3ebc9e567681d3993e3491b6d3d3a0dfb46e5552720c6946aba`. Full trace, input-file identities and comparison summaries are retained under ignored `reports/phase4-tinystories-natural-trace-v1/`.
