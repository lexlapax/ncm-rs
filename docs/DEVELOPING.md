# Developer Guide

This guide explains how `ncm-rs` is structured, how to embed it as a Rust library, how to extend the CUDA and Metal/MLX backends, and how to add a new local NCM model or profile.

For building the CLI, downloading/exporting checkpoints, and running inference, see [docs/USAGE.md](USAGE.md). For system design and invariants, see [docs/ARCHITECTURE.md](ARCHITECTURE.md).

## 1. Workspace Layout

```text
crates/
  ncm-core/          Validated FP32 configs (CodecConfig, OwtConfigOnly), errors,
                     WeightBackend / CodecBackend / GeneratorBackend traits.
  ncm-io/            Integrity-checked artifact loading (Artifact::load,
                     load_inference, load_inference_streaming), parallel SHA-256,
                     single-tensor streaming scratch (anonymous mmap on macOS).
  ncm-tokenizers/    Pinned tokenizers: Sudoku, TinyStories byte BPE, GPT-2 (XSum),
                     Qwen3 (WMT14), BERT (LM1B). Zero Python runtime dependency.
  ncm-model/         Shared FP32 model semantics generic over Backend: Codec,
                     Generator (Sudoku), TinyStoriesTeacher, XsumTeacher,
                     Wmt14Teacher, Lm1bSmallTeacher (small/medium/large), TraceSink.
  ncm-backend-cuda/  Candle + CUDA/cuDNN backend (CandleBackend), custom CUDA
                     kernels (sampling, resampling, penalties, argmax, RoPE/norms).
  ncm-backend-mlx/   Native Apple Silicon MLX backend (MlxBackend) via mlx-rs/mlx-c.
  ncm-serve/         Zero-new-dependency HTTP/1.1 + SSE/NDJSON OpenAI & Ollama API server
                     (`Server`, `InferenceEngine`, `ModelRegistry`).
  ncm-cli/           Unified CLI (`ncm sample`, `ncm serve`, `ncm list`, `ncm inspect`,
                     `ncm generate`, `ncm reconstruct`, and legacy `sample-*` aliases).

tools/
  check.sh           Local validation entrypoint (CPU default, opt-in --cuda / --mlx).
  reference/         Offline Python exporters, pinned upstream references, evaluators.
  bench/             Reproducible equivalence, quality, profiling, and alternating
                     before/after benchmark runners + offline unit tests.
```

### Cargo Feature Matrix

Build each platform separately; do **not** run `--all-features` on Linux or macOS.

| Platform | Crate | Feature(s) | What It Enables |
| --- | --- | --- | --- |
| Any (CPU checks) | workspace | default (`cpu` on `ncm-backend-cuda`) | Unit tests, synthetic fixtures, report tests; no GPU SDK required |
| Linux / NVIDIA | `ncm-cli` | `--features cuda-cudnn` | All 9 qualified profiles on CUDA/cuDNN (`wmt14-experimental` and `lm1b-experimental` are folded in) |
| Linux / NVIDIA | `ncm-cli` | `--features cuda` | Sudoku + WMT14 without cuDNN |
| Apple Silicon | `ncm-cli` | `--features mlx` | All 9 qualified profiles on Metal/MLX (`wmt14-experimental` and `lm1b-mlx-experimental` are folded in) |

For exact toolchain, `CUDA_COMPUTE_CAP`, cuDNN, and `MLX_SYS_PREBUILT` setup, see [docs/USAGE.md](USAGE.md).

## 2. Using `ncm-rs` as a Rust Library

All model logic lives in `ncm-model`, generic over `ncm_core::GeneratorBackend`. No Python interpreter is linked or invoked at runtime.

### Loading & Running a Local Artifact

```rust
use ncm_core::CodecBackend;
use ncm_io::Artifact;
use ncm_model::TinyStoriesTeacher;
use ncm_tokenizers::TextTokenizer;

// 1. Validate the artifact manifest, whole-file & per-tensor SHA-256 hashes,
//    and finite FP32 weights using bounded streaming buffers.
let artifact = Artifact::load_inference_streaming("artifacts/canonical/tinystories-phase4-v1")?;

// 2. Construct the tokenizer from verified in-memory metadata (optional).
let tokenizer = TextTokenizer::from_asset_bytes(
    artifact.file("tokenizer-manifest.json")?,
    artifact.file("tokenizer.json")?,
    artifact.file("tokenizer.LICENSE")?,
)?;

// 3. Select the native GPU backend explicitly (no silent CPU fallback).
//    Linux/CUDA: ncm_backend_cuda::CandleBackend::cuda_cudnn(0)?
//    macOS/MLX:  ncm_backend_mlx::MlxBackend::gpu()?
let backend = /* CandleBackend or MlxBackend */;

// 4. Preflight complete tensor inventory before the first GPU allocation,
//    then stream weights one tensor at a time directly to the device.
let model = TinyStoriesTeacher::from_artifact(backend.clone(), &artifact)?;
drop(artifact); // closes validated file descriptors

// 5. Run inference and read back token IDs.
let generated = model.sample_seeded(&[0], 1, 42)?;
let ids = backend.read_indices(&generated.ids)?;
let text = tokenizer.decode(&ids, true)?;
```

Key library invariants to preserve:
- **Immutable weights, local request state:** models own immutable device weights and static level geometry; activations and RNG state live per request. `Generator` / `TinyStoriesTeacher` / `XsumTeacher` serialize concurrent calls via an internal `Mutex<()>` (serving concurrency is out of scope for v0.1/v0.2).
- **Preflight before allocation:** every `from_artifact` / `from_owned_artifact` constructor checks config, tensor inventory, dtypes, shapes, finite FP32 values, `lvl_1L`, and `attn_mask_for_masking` **before** calling `backend.upload_f32` for the first time.
- **Explicit device errors:** unsupported device/profile/option combinations return typed `ncm_core::Error` rather than falling back to CPU or silently changing semantics.

## 3. Backend Contract & Extending Operations

The backend traits in [crates/ncm-core/src/lib.rs](../crates/ncm-core/src/lib.rs) are intentionally narrow:

- `WeightBackend`: `upload_f32(&self, values: &[f32], shape: &[usize]) -> Result<Self::Tensor>`.
- `CodecBackend: WeightBackend`: indexing, reshape/permute/narrow, arithmetic, `project` / `project_bias`, `embedding`, `select_last`, `layer_norm` / `rms_norm` / `group_norm`, `gelu`, masked noncausal `attention`, `conv1d` (and transposed), exact `area1d` / `linear1d`, `evaluate`.
- `GeneratorBackend: CodecBackend`: `concat_f32`, `split_qkv`, `merge_heads`, `rms_rope`, categorical sampling (`categorical_4096`, `categorical_8192`, `lm1b_categorical_8192_fp32`), top-1 / penalty helpers, and `xsum_reference_backend`.
- `TraceSink<B>` ([crates/ncm-model/src/blocks.rs](../crates/ncm-model/src/blocks.rs)): zero-overhead hook (`NoTrace` default) used for numerical parity traces (`tensor`, `indices`) and Step 2 stage profiling (`stage_tensor`, `stage_indices`).

When adding or optimizing a backend operator:
1. Keep the shared mathematical specification in `ncm-core` / `ncm-model` identical across CUDA and MLX.
2. Add synthetic CPU/GPU unit & operator tests in `crates/ncm-backend-cuda/tests/` and `crates/ncm-backend-mlx/tests/` (checking edge shapes, ties, and NaN/Inf rejection).
3. Keep `unsafe_code = deny` at the crate root; any unavoidable FFI or `mmap` must be narrowly scoped with `#[allow(unsafe_code)]` and explicit `// SAFETY:` comments.
4. Validate on **Linux/CUDA first**, then **macOS/MLX**, using real hardware (`./tools/check.sh --cuda`, then `./tools/check.sh --mlx`).

## 4. Running Your Own Local NCM Model (`local-ncm-v1`)

For custom local NCM checkpoints that follow the standard NCM multiscale RVQ codec + block-diagonal RoPE transformer architecture, you do **not** need to write custom Rust structs. Export the checkpoint to the verified `local-ncm-v1` artifact profile using the stdlib offline exporter [tools/reference/export_local_ncm.py](../tools/reference/export_local_ncm.py) (optional `torch` only when converting `.pt`/`.pth` files):

```sh
python3 tools/reference/export_local_ncm.py \
  --config path/to/config.json \
  --codec path/to/codec.safetensors \
  --generator path/to/generator.safetensors \
  --tokenizer path/to/tokenizer.json \
  --output artifacts/canonical/my-local-ncm
```

`config.json` contains `"tokenizer"` (`RawCodecConfig` validated by `CodecConfig::validate_local`) and `"model"` (`LocalGeneratorConfig`, supporting `"head_kind": "untied_affine" | "untied_linear" | "shared_head_biases"`, optional `"pad_token_id"`, `"eos_token_id"`, and `"temperature"`). Run it directly on CUDA or Metal with the unified CLI:

```sh
# Inspect manifest & SHA-256 inventory:
target/release/ncm inspect artifacts/canonical/my-local-ncm

# Generate from token IDs or text (when tokenizer.json is bundled):
target/release/ncm sample artifacts/canonical/my-local-ncm cuda-cudnn prompts.json --text --decode --seed 42
target/release/ncm sample artifacts/canonical/my-local-ncm mlx prompts.json --text --decode --seed 42
```

From Rust, load it via `LocalNcmModel::from_artifact(backend, &artifact)` (`crates/ncm-model/src/local_ncm.rs`), which runs full host preflight before GPU allocation, enables per-layer FP32 prefix K/V caching automatically when `prefix_len > 1`, and supports greedy (`seed: None`) or deterministic SplitMix64 sampling (`seed: Some(u64)`).

## 5. Adding a New Bespoke Pinned Profile

If a model introduces new architectural components beyond `local-ncm-v1` (custom decoding penalties, CFG schedules, or unigram tables):

1. **Export to the canonical non-executable artifact format** (see [docs/ARTIFACTS.md](ARTIFACTS.md)):
   - Write an offline exporter under `tools/reference/` using `torch.load(..., weights_only=True)` (never execute pickle in Rust).
   - Emit `manifest.json` (schema v1, explicit `profile` name, file & tensor SHA-256 digests), `config.json`, `codec.safetensors`, `generator.safetensors`, `training.safetensors`, optional `inference.safetensors` (unigrams, RoPE tables), `tokenizer*`, and `provenance.json`.
   - Verify bit-exact tensor roundtrip before leaving Python.
2. **Register the profile & byte budgets in `ncm-io`** ([crates/ncm-io/src/lib.rs](../crates/ncm-io/src/lib.rs)):
   - Add the new profile string, exact allowed file/bundle inventory, and per-file / total byte limits in `Artifact::load_impl`.
   - Add positive and negative-control tests in `crates/ncm-io/tests/artifact.rs` (including `load_inference_streaming`).
3. **Validate config semantics in `ncm-core` and `ncm-model`**:
   - Add a strict config validator (reject unknown inference-critical fields, check level schedule, head ratios, vocab bounds, EOS/PAD policy).
   - Implement preflight inspection (supporting both `Artifact::is_streaming_bundle` via `bundle_tensors` / `read_bundle_tensor` and in-memory `SafeTensors`) and streaming weight upload via `Artifact::visit_bundle_tensors`.
   - Add a zero-allocation-on-invalid-input test in `crates/ncm-model/tests/loading.rs`.
4. **Tokenizer parity in `ncm-tokenizers`**:
   - Add or reuse a pinned tokenizer loader from verified artifact bytes; add encode/decode boundary tests (special tokens, padding, truncation, Unicode) matching the offline Python tokenizer.
5. **CLI dispatch in `ncm-cli`**:
   - Register the profile in `crates/ncm-cli/src/sample.rs` (unified `ncm sample` manifest dispatch) and `crates/ncm-cli/src/decode.rs` (`--text` / `--decode`).
   - Fail closed on unsupported devices or experimental flags.
6. **Qualify on both GPUs before advertising support**:
   - Teacher-forced numerical trace parity against the pinned reference (`TraceSink`).
   - End-to-end quality gate on the profile's official metric (accuracy / ROUGE / BLEU+chrF / generative PPL+entropy).
   - Alternating 3-process performance & memory comparison (`docs/REPORTING.md`, `tools/bench/run_phase5_step3_bench.py`) proving Rust matches or beats Python on latency and memory.

## 6. Local Validation & Benchmarking Workflow

```sh
# 1. Default CPU checks (fmt, clippy, unit tests, docs, Python report tests):
./tools/check.sh

# 2. Linux/CUDA real-device check:
NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 ./tools/check.sh --cuda

# 3. Apple Silicon / Metal real-device check:
export MLX_SYS_PREBUILT="$PWD/.reference/mlx-native"
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$MLX_SYS_PREBUILT/lib"
./tools/check.sh --mlx
```

- Stage profiling: `tools/bench/run_phase5_profile.py --platform {cuda,mlx}` (builds on `phase5_stage_profile` examples).
- Optimization before/after comparison: `tools/bench/run_phase5_step3_bench.py --platform {cuda,mlx} --before-binary ... --after-binary ... --processes 3`.
- CLI output equivalence: `tools/bench/phase5_cli_equivalence.py --platform {cuda,mlx}`.

See [docs/TESTING.md](TESTING.md) and [docs/REPORTING.md](REPORTING.md) for the full numerical, quality, and reporting contracts.
