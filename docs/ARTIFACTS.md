# Canonical Artifact Contract (v1)

Canonical runtime artifacts are non-executable, integrity-checked directories portable across CUDA and Metal/MLX. Python is used strictly offline to convert pinned upstream checkpoints (`torch.load(..., weights_only=True)` wherever supported) into safetensors + versioned JSON; production Rust never executes pickle or Python code.

## Implemented Profiles (Schema Version 1)

All nine released profiles share schema version `1`, pinned upstream commit `c70274273f489abc73832b2b98584a8483b6b203`, and checkpoint repository `hazyresearch/ncm-checkpoints` at revision `d342ce8aaea58830eef3b8ad473664673e41f692`.

| Profile | Exporter (`tools/reference/`) | Bundles (`*.safetensors`) | Per-file / Total Budget | Notes |
| --- | --- | --- | --- | --- |
| `sudoku-easy-inspection-v1` | `sudoku_artifact.py` | `codec`, `generator`, `training` | 256 / 512 MiB | 238 tensors (36 EMA/training separated) |
| `sudoku-medium-inspection-v1` | `sudoku_artifact.py` | `codec`, `generator`, `training` | 256 / 512 MiB | Same schema; provenance-only re-export documented in Phase 5 |
| `sudoku-hard-inspection-v1` | `sudoku_artifact.py` | `codec`, `generator`, `training` | 256 / 512 MiB | Same schema; provenance-only re-export documented in Phase 5 |
| `tinystories-inference-v1` | `tinystories_artifact.py` | `codec`, `generator`, `training`, `inference` | 384 / 640 MiB | Byte BPE assets + `level0_unigram` in `inference.safetensors` |
| `xsum-inference-v1`, `xsum-inference-v2` | `xsum_artifact.py` | `codec`, `generator`, `training`, `inference` | 384 / 640 MiB | GPT-2 tokenizer + `token_unigram`; `v2` adds hash-pinned `rope_cos`/`rope_sin` |
| `wmt14-inference-v1` | `wmt14_artifact.py` | `codec`, `generator`, `training` | 640 / 1,280 MiB | Qwen3 tokenizer; 63 codec + 186 generator + 33 training tensors |
| `lm1b-small-inference-v1`, `lm1b-small-inference-v2` | `lm1b_small_runtime_artifact.py` | `codec`, `generator`, `training`, `inference` | 448 / 1,024 MiB | BERT tokenizer + `level0_unigram`/`token_unigram`; `v1` uses sibling rotary sidecar `lm1b-reference-rope-v1.safetensors`, `v2` (`--reference-rope`) bundles hash-pinned `rope_cos`/`rope_sin` in `inference.safetensors` |
| `lm1b-medium-inference-v1`, `lm1b-medium-inference-v2` | `lm1b_small_runtime_artifact.py` | `codec`, `generator`, `training`, `inference` | 1,280 / 1,920 MiB | 24-layer / 1024-wide generator + shared codec & BERT tokenizer; `v2` self-contained |
| `lm1b-large-inference-v1`, `lm1b-large-inference-v2` | `lm1b_small_runtime_artifact.py` | `codec`, `generator`, `training`, `inference` | 2,944 / 3,584 MiB | 36-layer / 1280-wide generator + shared codec & BERT tokenizer; `v2` self-contained |
| `local-ncm-v1` | `export_local_ncm.py` | `codec`, `generator` (optional `training`, `inference`) | 4,096 / 8,192 MiB | User-exported local NCM checkpoints: required `{config.json, codec.safetensors, generator.safetensors, provenance.json}`, optional HF `tokenizer.json`; strict `LocalNcmConfig` validation (`head_kind`: `untied_affine` / `untied_linear` / `shared_head_biases`, optional `pad_token_id` / `eos_token_id` / `temperature`) |
| `owt-inference-v1` / `openwebtext-inference-v1` | — | **Explicitly rejected** | — | Weights unreleased by upstream; loader and CLI refuse execution |

Exact manifest SHA-256 digests and step-by-step export commands for every profile are in [docs/USAGE.md](USAGE.md).

## Directory Layout

```text
<artifact-root>/
  manifest.json             # schema_version, profile, pins, files{bytes,sha256}, tensors[{name,shape,dtype,bytes,sha256}]
  config.json               # pinned upstream config envelope
  codec.safetensors         # FP32 codec weights (encoder, VQ codebooks, cascade, decoder)
  generator.safetensors     # FP32 generator weights + lvl_1L (I64) + attn_mask_for_masking (F32)
  training.safetensors      # separated EMA/count/hit buffers (validated, dropped on inference load)
  inference.safetensors     # optional profile statistics (unigrams, XSum v2 RoPE tables)
  tokenizer.json            # pinned tokenizer asset
  tokenizer-manifest.json   # tokenizer provenance/pin metadata (text profiles)
  tokenizer.LICENSE         # upstream tokenizer license (text profiles)
  provenance.json           # export metadata and input checksums
```

## Loading Modes & Verification (`ncm-io`)

`crates/ncm-io` provides three checked loaders, all enforcing exact flat filename inventories, strict manifest deserialization (no unknown fields, no duplicate keys), rank ≤ 8, checked shape-product arithmetic, per-profile byte budgets, non-symlink roots/files, whole-file SHA-256, and per-tensor SHA-256:

1. **`Artifact::load_inference_streaming(root)` (default for `ncm-cli` and benchmarks):**
   - Loads small JSON/tokenizer/`inference.safetensors` payloads into memory.
   - Opens `codec.safetensors`, `generator.safetensors`, and `training.safetensors` with `O_NOFOLLOW` and verifies them concurrently: for bundles ≥ 4 MiB, one thread hashes the whole file while up to 6 worker threads verify per-tensor SHA-256 digests and finite FP32 values (exempting `attn_mask_for_masking`, whose exact `0.0`/`-inf` values are validated in `ncm-model`) in file-offset order using 1 MiB scratch buffers (anonymous `mmap`/`munmap` on macOS aarch64). Small bundles (< 4 MiB) run in a single sequential pass.
   - Drops `training.safetensors` after validation and retains open validated file descriptors for `codec.safetensors` and `generator.safetensors`.
   - `ncm-model` performs complete host preflight (`bundle_tensors`, `bundle_f32_weights_verified_finite`, `read_bundle_tensor` for `lvl_1L`/`attn_mask_for_masking`/`log_inv_T`) before the first GPU allocation, then streams tensors one at a time via `visit_bundle_tensors` into a single reusable `Scratch` buffer (immediately unmapped on macOS on drop).
2. **`Artifact::load_inference(root)`:** copies `codec.safetensors` and `generator.safetensors` into owned `Vec<u8>` buffers, validates `training.safetensors` via streaming, and parallelizes whole-file + per-tensor SHA-256 verification.
3. **`Artifact::load(root)`:** retains all declared files (including `training.safetensors`) in owned `Vec<u8>` buffers for inspection and tests.

## Security & Trust Boundaries

- **Integrity vs authenticity:** manifest SHA-256 hashes prove internal consistency relative to `manifest.json`, not cryptographic authenticity of the manifest itself. Callers must check the expected `manifest.json` SHA-256 (or trust the local artifact directory).
- **Non-mutating directory assumption:** as with `Artifact::load`, the artifact directory must be trusted and not concurrently mutated by an untrusted process during loading. `load_inference_streaming` opens weight bundles with `O_NOFOLLOW` and reads from the validated open file descriptors.
- **No silent weakening:** missing/extra tensors, dtype/shape mismatches, nonfinite FP32 weights, corrupted masks/level buffers, path traversal, symlinks, or unknown profiles fail closed before any GPU allocation.
- **Weights & data stay out of Git:** checkpoints, exported artifacts, and evaluation datasets remain under ignored local paths (`artifacts/`, `reports/`, `.reference/`).
