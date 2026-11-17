# LM1B-small inference artifact preparation

This is a CPU-only proposed layout, not a supported runtime artifact. Phase 4
profile ordering and the earlier XSum/WMT gates remain in force. The current
loader rejects `lm1b-small-inference-preparation-v1` explicitly; no model or
GPU support is implied.

The pinned small checkpoint and qualified BERT asset produce nine files in the
existing manifest style. `generator.safetensors` holds all 213 generator
tensors; `codec.safetensors` holds 100 codec tensors;
`training.safetensors` retains 36 named EMA/hit tensors; and
`inference.safetensors` holds both required unigram distributions. Config,
tokenizer payload/manifest/license and provenance are retained. The training
split is based on the pinned source and the bounded fixed-code mutation probe;
it is not a claim about training or every upstream API. No tensor is discarded.

Two independent CPU exports round-tripped names, shapes, dtypes and bytes and
matched byte-for-byte. Both manifests hash to
`82ebe150ef1fc848b3dc4520c38342b9e44021e2eb219aba2b6eab5c4ce48ed1`.
All 351 tensor inventory records match the earlier lossless small preparation
export when codec and training bundles are recombined. The exporter hashes to
`e3b70fa3e2134243583f6b5b20dd0a1d08601373dd4e94cebf548ec0d0fd6ed8`.
The output directory `artifacts/canonical/lm1b-small-inference-preparation-v1`
and the second export remain ignored local data.

The combined manifest file sizes total 988,636,075 bytes; the largest file is
445,628,784 bytes. A later small-only loader could consider independent
448 MiB per-file and 1024 MiB total limits, subject to explicit tests and
review; these are not implemented and do not relax other profiles. Medium and
large need separate budgets and are not covered by this layout.

The command was `setpriv --inh-caps=-all env OMP_NUM_THREADS=1
TOKENIZERS_PARALLELISM=false .reference/venv/bin/python
tools/reference/lm1b_small_inference_preparation.py --source
artifacts/upstream/lm1b-phase4 --tokenizer-asset
artifacts/canonical/tokenizers-qualified/bert --output NEW_OUTPUT`. Two
focused tests cover split preservation and invalid inventory/output rejection.
The freshly rebuilt CLI rejected the preparation profile as unsupported;
its binary SHA-256 is
`a2fcec4aac64e32939510849d0095709d47377f0634ab1e826eb3ff0002ce838`.
An earlier probe hit a missing cuDNN library path, and an initial shell status
assignment failed because `status` is read-only in zsh; both are retained as
failed diagnostics, not loader evidence.

Before runtime support, implement and test profile-specific file/bundle bounds,
full and inference-only loading, exact semantic schema validation before GPU
upload, tokenizer behavior, and explicit unsupported-feature rejection. Then
complete the ordered CUDA and Metal correctness, quality and performance gates.

The repository standard `tools/check.sh` passed; log SHA-256
`28cdd49394ad4d8e3be36839a6c41c102691dcd8c34293a523ef1eddf655040e`.

## Host-only generator preflight

The explicit `lm1b-experimental` feature adds a host-only small-generator
inspection over preparation bytes. It requires the exact canonical config
SHA-256, checks the complete 213-tensor generator inventory and expected
shapes/dtypes, scans ordinary weights for nonfinite values, and verifies the
level buffer and block-diagonal attention mask. It separately checks both
inference distributions for shape, finite nonnegative values and sum within
the existing `1e-6` distribution check. This accepts no artifact profile and
does not construct a model or upload weights. Integrity of supplied bytes is
the caller's responsibility; the local probe bounds and hashes its inputs.

The actual preparation config, generator and inference files passed the CPU
probe with 213 generator and two inference tensors. Unit tests cover the exact
shape map, wrong config and malformed distributions; strict feature-enabled
Clippy passes. The probe executable SHA-256 is
`7c6d9e028427556345de6d4324df145d42004e7713d3b4071f20c4a74e0abc33`;
its source and preflight module hashes are
`03b33ce26ffd43d43d7e7504c2554498badaf036177dd9e43fc6019b265a4dc6` and
`2bd96ad01a9fdbb025ec5e734cee770c0af9dd5518f475960a6d152b333af09c`.
Full `tools/check.sh --cuda` passed 18 real-device operator cases (log SHA-256
`5f19e381e9ba7189537341bb2c37655be6130c2d1297f91583e0189ad0615477`);
all-feature model tests and strict all-target Clippy also pass. Codec semantic schema, artifact loader
support, generator execution and all LM1B GPU/quality gates remain open.
