# LM1B-medium and LM1B-large CUDA teacher-forced parity

The LM1B model code now reads generator depth, head count and width from a pinned size table
(`Lm1bSize` in `crates/ncm-model/src/lm1b_small.rs`). The artifact profile selects the size, and the
pinned `config.json` must identify the same size. The shared codec and tokenizer, the 12 levels and
the 64-wide heads are unchanged. Two CUDA operator whitelists now admit the larger head counts:
16 and 20 heads for the efficient attention kernel, and 12, 16 and 20 for the fused rotary kernel.
Both kernels index heads only through the flat grid, so their arithmetic is unchanged. The CLI accepts all
three sizes through `ncm sample-lm1b` (`sample-lm1b-small` still works). The tokenizer checkpoint path that
medium and large configs name (`../tokenizer.pt`) is normalized as metadata after the pinned config hash check.

## Method

Each comparison forces the same two fixed development-code histories (the first two rows of the small
seed-11 development capture) through the model. It then compares 51 tensors with the existing
`compare_lm1b_small_teacher_trace.py` policy. The tensors are every level's head logits, post-upsampling
convolution input and output, codec reconstruction, final logits, and greedy and penalty-adjusted IDs.
The tolerances are unchanged. The Python references come from the pinned upstream checkout and checkpoints on
the same RTX 3060 Ti, using `tools/reference/lm1b_small_teacher_trace.py --size SIZE`.

There are two Rust CUDA backends, and each is compared with the Python settings it reproduces:

- **cuDNN (strict):** `CandleBackend::cuda_cudnn`, against a strict reference with math attention, FP32
  matmul, convolution TF32 off and deterministic cuDNN.
- **Efficient (the `sample-lm1b` CUDA path):** `cuda_lm1b_efficient_experimental` with the pinned rotary
  tables. It is compared against a reference with PyTorch's default attention dispatch and the pinned
  evaluator's convolution settings (`--sdpa default --upstream-convolution-precision`: convolution TF32 on,
  nondeterministic, no benchmarking). This matches how the earlier small efficient-path parity was
  established ([controlled evidence](small-controlled-cuda-v1.md#controlled-generation-at-evaluator-convolution-precision)).

## Results

| Size | Backend vs reference | Tensors within policy | Head argmax mismatches | Greedy / penalized ID mismatches | Max head-logit error |
| --- | --- | ---: | ---: | --- | ---: |
| small | cuDNN vs strict (regression) | 51 / 51 | 0 | 0 / 0 | — |
| small | efficient vs evaluator | 51 / 51 | 0 | 0 / 0 | — |
| medium | cuDNN vs strict | 51 / 51 | 0 | 0 / 0 | 1.91e-4 |
| medium | efficient vs evaluator | 51 / 51 | 0 | 0 / 0 | 1.07e-4 |
| large | cuDNN vs strict | 51 / 51 | 0 | 0 / 0 | 2.14e-4 |
| large | efficient vs evaluator | 51 / 51 | 0 | 0 / 0 | 1.22e-4 |

The small regression run used the original pinned small reference
(`fd815da4…`) and passes as before. Sampling all three sizes through `ncm sample-lm1b ... cuda-cudnn`
succeeds, and small's sampled output is byte-identical to the pre-change build.

### Mismatched settings, retained as a diagnostic

The efficient backend compared against the **strict** reference fails 45 of 51 tensors at every size,
including small, with 6–8 head argmax mismatches. Greedy and penalized IDs still match. The first difference
is level-zero reconstruction (max 3.03e-2). A direct probe recomputed that reconstruction for the same codes in
PyTorch. With convolution TF32 enabled it reproduces the efficient backend's values exactly (difference 0.0);
with TF32 off it reproduces the strict reference exactly. The efficient path therefore follows the
evaluator's TF32 convolution precision by design. This is not a defect, and it does not change any tolerance.
Default attention dispatch with strict convolutions also fails, the same way. An experiment uploading forced
codes as contiguous slices produced a byte-identical trace and was reverted.

| Item | SHA-256 |
| --- | --- |
| medium strict reference trace | `5e5a2e916069271ff7c669566370b7e6d96f5991d511b86e0ec894381096a2a8` |
| large strict reference trace | `4c49bcd72a4a6f5d6f3cc906adfac3d866ee7412aca0aa1366bbb96176521184` |
| small evaluator reference trace | `acf6c97b661a3e85d261dd710c8504736ade859a09ba7358326b546f8741136a` |
| medium evaluator reference trace | `3ee9dd897f8e1c7f0cbdf4946d5bba77d518f31f8c7bda069a20fd191140d6e6` |
| large evaluator reference trace | `e6ce2ff7fcd21d06ed5d14b5d741208b1f2e0cfdb5dd47cbdaad758f9038d031` |
| medium cuDNN comparison | `b964016372896cbb9c94ab813e3f338e698bad5c42e9b753dddd0df6c90ac3c8` |
| medium efficient comparison | `4c1c14ed888887ef7687da182ce78b074429f72f00fde389cb1a97b374303093` |
| large cuDNN comparison | `fdb92f69b0972b1aa2e37055037f65627b02cab9c33e839bd85c68279323eafb` |
| large efficient comparison | `266bd9526ad94a38e19d1c5a545b68e6d9c461e53681e23c907f62d41488ca81` |

Raw traces and comparisons are in ignored `reports/phase4-lm1b-{small,medium,large}-teacher-*`.

## Reproduce (from the repository root; `LD_LIBRARY_PATH` set to the venv cuDNN as in other LM1B commands)

```sh
cargo build --locked --release -p ncm-backend-cuda --features lm1b-experimental,cuda-cudnn \
  --example lm1b_small_teacher_trace
PYTHONPATH=tools/reference .reference/venv/bin/python tools/reference/lm1b_small_teacher_trace.py \
  --size medium --sdpa default --upstream-convolution-precision --upstream .reference/upstream-ncm \
  --source artifacts/upstream/lm1b-phase4 --prefixes reports/phase4-lm1b-small-cuda-development-v1.json \
  --output reports/NEW-REFERENCE
target/release/examples/lm1b_small_teacher_trace artifacts/canonical/lm1b-medium-inference-v1 \
  reports/NEW-REFERENCE reports/NEW-RUST efficient reports/phase4-lm1b-level7-replay-v1/reference-rope-tables.safetensors
PYTHONPATH=tools/reference .reference/venv/bin/python tools/reference/compare_lm1b_small_teacher_trace.py \
  reports/NEW-REFERENCE/trace.safetensors reports/NEW-RUST/traces.safetensors reports/NEW-COMPARISON.json \
  --reference-sha256 "$(sha256sum reports/NEW-REFERENCE/trace.safetensors | cut -d' ' -f1)"
```

For the strict pair, omit `--sdpa default --upstream-convolution-precision` and the `efficient ROPE`
arguments. Quality and performance for medium and large are separate gates.
