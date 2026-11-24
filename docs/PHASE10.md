# Phase 10 Roadmap (`v0.5.0` → `v0.6.0`)

Approved sequential milestones (each validated on **Linux/CUDA first**, then **Apple Silicon/Metal** on `macstudio` before advancing):

1. **Milestone 10A — Pure-Rust `ncm pull` (Zero-Python Single Binary)**
2. **Milestone 10B — Fused Attention (CUDA Tensor-Core FlashAttention + Metal Fast SDPA)**
3. **Milestone 10C — `ncm-train` v3 (Multi-Profile LoRA + Hot-Swappable Adapters in `ncm serve`)**

Hard invariants across all milestones:
- **Strict `FP32` defaults remain 100% byte-identical** across all 9 qualified inference profiles on both CUDA and Metal.
- **Zero Python** in production `pull`, `quantize`, `sample`, `run`, `serve`, and `train`.
- **Zero silent pin/tolerance changes**; unsupported features fail explicitly and fail closed.
- **Deliberate dependencies:** avoid adding new external crates wherever existing workspace crates or bounded safe Rust suffice.

---

## Milestone 10A — Pure-Rust `ncm pull` (Zero-Python Single Binary)

### Goals
- Eliminate the last Python dependency for end users: `ncm pull MODEL` downloads, SHA-256 verifies, parses PyTorch `.pt` checkpoints, converts `config.yaml`, builds canonical verified tokenizer assets matching `tools/reference/tokenizers-v1.lock.json` bit-for-bit, and exports verified `safetensors` + `manifest.json` artifacts **entirely in pure Rust** (zero Python required on Linux or macOS).
- Preserve all existing SHA-256 and byte-size pins from `tools/reference/fetch_checkpoints.py`, `tokenizer-sources.lock.json`, `tokenizers-v1.lock.json`, and the upstream exporters.

### Architecture
1. **Pure-Rust PyTorch `.pt` reader (`crates/ncm-io/src/pytorch.rs`):**
   - Bounded safe ZIP_STORED reader (all upstream `.pt` checkpoints use uncompressed `STORED` entries) + protocol 2/4 safe data-only pickle VM supporting only `collections.OrderedDict`, `torch._utils._rebuild_tensor_v2`, storage persistent loads (`FloatStorage`, `LongStorage`, `HalfStorage`, `BFloat16Storage`, `ByteStorage`), dicts, tuples, lists, ints, floats, bytes, and strings. Rejects arbitrary globals/callables and unexpected shared storage aliases.
   - Reuses the same safe pickle VM for TinyStories `bpe_vocab_8192.pickle` and `bpe_merges_8192.pickle`.
2. **Pure-Rust bounded YAML-to-JSON converter (`crates/ncm-io/src/yaml.rs`):**
   - Clean safe parser covering the exact YAML subset used by all pinned upstream `config.yaml` files (mappings, nested indentation, inline lists, scalars, comments, booleans, ints, floats, null, strings), verified for exact `serde_json::Value` equality against every upstream `config.yaml`.
3. **Pure-Rust canonical tokenizer builder (`crates/ncm-tokenizers/src/convert.rs`):**
   - Builds `tinystories` (`byte_bpe`), `gpt2`, `qwen3`, and `bert` canonical assets offline using vendored `tokenizers-0.22.2` + `ncm-io::pytorch` pickle reader, producing the **exact byte-for-byte `tokenizer.json` and `manifest.json` SHA-256s** pinned in `tools/reference/tokenizers-v1.lock.json` (plus verified `LICENSE` copy).
4. **Pure-Rust verified HTTPS fetcher + exporter (`crates/ncm-cli/src/pull.rs`):**
   - Streams pinned files via system `curl --fail --location --proto =https --proto-redir =https` with Rust-side exact byte-count & SHA-256 verification (enforcing budgets, rejecting symlinks/existing destinations, atomic rename).
   - Supports all quickstart models (`tinystories`, `tinystories-legacy`, `xsum`, `wmt14`, `lm1b-small`, `lm1b-medium`, `lm1b-large`, `sudoku-easy`, plus `sudoku-medium`/`sudoku-hard`) and `--dry-run`, `--list`, `--models-dir`, `--upstream-dir`, `--reference-rope`, and `--python-legacy` (opt-in legacy script delegation).

### Validation Gate (CUDA then Metal)
- Unit tests: synthetic ZIP_STORED `.pt` roundtrip, pickle safety/rejection tests (arbitrary global/callable rejected), YAML-to-JSON exact match on all upstream `config.yaml` schemas, canonical tokenizer SHA-256 bit-exact match against `tokenizers-v1.lock.json`.
- Live pure-Rust `ncm pull` offline-from-cache / full export test on `sudoku-easy`, `tinystories`, `wmt14`, `xsum`, and `lm1b-small`: verify every exported `safetensors` tensor SHA-256 matches the existing Python-exported canonical artifacts **bit-for-bit**, and `ncm inspect` + `ncm sample` produce **0 diffs** on both CUDA and Metal.

---

## Milestone 10B — Fused Attention (CUDA Tensor-Core FlashAttention + Metal Fast SDPA)

### Goals
- Eliminate $O(B \cdot H \cdot S_q \cdot S_k)$ attention score/probability materialization in opt-in low-precision modes (`bf16`/`fp16`/`q8`/`q4`) on both GPUs.
- Deliver major latency and peak-VRAM improvements at batch 1 and high batch (`b=8..16`) across all profiles, including prefix KV cache (`seq_q != seq_k`).

### Architecture
1. **CUDA (`crates/ncm-backend-cuda/src/fused_attention.cu` & `fused_attention.rs`):**
   - Custom Ampere Tensor-Core FlashAttention-2 tiled kernel (`mma.sync.aligned.m16n8k16.row.col.f32.{bf16,f16}`) with FP32 online softmax, supporting `head_dim = 64`, arbitrary `seq_q` and `seq_k` (including prefix KV cache), optional broadcastable additive mask (`[1|B, 1, S_q, S_k]`), and zero $O(S^2)$ VRAM staging.
   - Wired into `CodecBackend::attention` for `bf16`/`fp16`/`q8`/`q4` on SM80+ CUDA devices; strict `FP32` path untouched.
2. **Metal (`crates/ncm-backend-mlx/src/codec.rs`):**
   - Wire `mlx_rs::fast::scaled_dot_product_attention` for `bf16`/`fp16`/`q8`/`q4` (supporting optional additive mask & `seq_q != seq_k`), leaving strict `FP32` untouched.

### Validation Gate (CUDA then Metal)
- Operator & precision tests verifying fused attention vs reference FP32 attention (`< 2e-3` max abs in FP16, `< 1e-2` in BF16 across `seq_q == seq_k` and `seq_q != seq_k`, masked and unmasked).
- **9/9 FP32 CLI byte-equivalence** vs `v0.5.0` on both CUDA and Metal.
- Paired before/after latency, peak VRAM/MLX memory, and task quality across `sudoku-easy`, `tinystories-15k-ema` (b=1 & b=8), `xsum` (b=1 & b=8), `wmt14` (b=1 & b=8), and `lm1b-large` (b=1 & b=16) on RTX 3060 Ti and M1 Ultra.

---

## Milestone 10C — `ncm-train` v3: Multi-Profile LoRA + Hot-Swappable Adapters in `ncm serve`

### Goals
- Extend `ncm train --init-from` LoRA fine-tuning beyond TinyStories/`local-ncm-v1` to upstream **WMT14**, **XSum**, and **LM1B** (`small`/`medium`/`large`).
- Support lightweight verified unmerged LoRA adapter artifacts (`ncm-lora-adapter-v1`, ~2–8 MB) and **hot-swappable multi-adapter serving** in `ncm serve` (multiple fine-tuned adapters sharing one resident GPU base model with near-zero extra VRAM, including over `Q8`/`Q4` quantized bases!).

### Architecture
1. **Multi-profile `--init-from` (`crates/ncm-train/src/artifact.rs`):**
   - Admit `wmt14-inference-v1`, `xsum-inference-v1`/`v2`, and `lm1b-{small,medium,large}-inference-{v1,v2}` alongside `local-ncm-v1` and `tinystories-inference-v1`, converting each generator's heads & codec weights cleanly into `local-ncm-v1` (`UntiedAffine` / `SharedHeadBiases`).
2. **Verified unmerged LoRA adapter artifact (`crates/ncm-io/src/lora_adapter.rs`):**
   - Schema `ncm-lora-adapter-v1`: `adapter.safetensors` + `adapter_manifest.json` pinned to `base_manifest_sha256`, `rank`, `alpha`, and per-tensor SHA-256 hashes (fail-closed on tamper or base-model mismatch).
   - `ncm train --save-adapter PATH` exports the lightweight unmerged adapter (with or without full merged `--output`).
3. **Runtime LoRA delta & multi-adapter serving (`ncm-model`, `ncm-cli`, `ncm-serve`):**
   - Runtime LoRA adapter support in `LocalNcmModel` (and `ncm sample --adapter PATH`, `ncm run --adapter PATH`), working seamlessly over `fp32`, `bf16`, `fp16`, `q8`, and `q4` base weights (QLoRA-style inference!).
   - `ncm serve --adapter ALIAS=BASE:ADAPTER_PATH` registers adapter aliases sharing the single resident base model in GPU VRAM, switching adapters cleanly per request and reporting them in `/v1/models` and `/api/tags`.

### Validation Gate (CUDA then Metal)
- Unit & integration tests: unmerged adapter save/load/tamper-fail-closed, adapter-vs-merged inference equivalence (`< 1e-5` in FP32, exact greedy token match), WMT14 & XSum & LM1B `--init-from` LoRA convergence tests on CPU/CUDA/Metal.
- Live dual-platform E2E benchmark: fine-tune 2 distinct LoRA adapters on `tinystories-15k-ema-v1` (plus live LoRA fine-tune on `wmt14` and `xsum`), verify `ncm serve` serving the base model + 2 hot-swapped adapters from one shared GPU base (including over `Q8` quantized base) on both RTX 3060 Ti and M1 Ultra.
