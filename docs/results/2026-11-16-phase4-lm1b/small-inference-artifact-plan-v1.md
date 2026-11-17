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
