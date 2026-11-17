# WMT14 inference artifact preparation plan

The existing checkpoint revision, both checkpoint hashes, config and Qwen3
assets are independently verified in the [inventory](inventory-preparation-v1.md).
The upstream generation path uses the 63 non-EMA/hit tokenizer tensors; its 33
EMA/hit tensors are retained separately rather than discarded. This
classification is for the generation path, not a claim about training or every
upstream API. The generator has no inference metadata bundle.

Prepare a new explicit `wmt14-inference-v1` artifact using the existing manifest
schema and these exact files: `codec.safetensors` (63 tensors),
`generator.safetensors` (186), `training.safetensors` (33), `config.json`,
`tokenizer.json`, `tokenizer-manifest.json`, `tokenizer.LICENSE`, and
`provenance.json`. Preserve names, shapes, dtypes and bytes, round-trip every
tensor, verify inputs before and after and never overwrite an output. The config
identity is the existing revision's independently verified
`6401af2a14d41c6582c3181a0b683e8970c1eeefe8cf7afaf74f1597da3fcd72`.
The proposed loader bounds are a profile-specific 640 MiB per file and 1280 MiB
in total; they do not relax other profiles. The observed tensor bytes are
576,011,264 codec, 480,523,968 generator and 69,746,688 retained training bytes.
Validate the exact file/tensor schema and limits with negative tests and real
CPU loading before any GPU upload. An exported or loaded artifact does not
establish generator, decoder, quality, performance or GPU support.

The CPU export command was `setpriv --inh-caps=-all env OMP_NUM_THREADS=1
TOKENIZERS_PARALLELISM=false .reference/venv/bin/python
tools/reference/wmt14_runtime_artifact.py --source artifacts/upstream/wmt14-phase4
--tokenizer-asset artifacts/canonical/tokenizers-qualified/qwen3 --output
artifacts/canonical/wmt14-inference-v1`. All 282 tensors round-tripped
byte-for-byte. The manifest SHA-256 is
`efc13bc5c6c6c854af31c80bb2449436cfb3352db16aa9a7861549e5ef194c78`, and the
exporter SHA-256 is `896142e7c91f376ef78fd24a78aad0053e1a0d0013ab884128c62e0ac52baf82`.
The output and its log remain ignored local data, not weights in Git.

`setpriv --inh-caps=-all env RUSTC_WRAPPER= CARGO_BUILD_JOBS=4 cargo test --locked
-p ncm-io` and strict Clippy passed, including negative
tests for WMT14 inventory and per-file/total bounds. The release probe
`setpriv --inh-caps=-all env OMP_NUM_THREADS=1 target/release/examples/wmt14_artifact_probe artifacts/canonical/wmt14-inference-v1
MODE` passed in separate `full` and `inference` CPU processes; the latter
validated the retained training bundle as a stream without keeping its bytes.
The probe SHA-256 is
`390089669c3cf00b3d0e36828af346c77776f31cc843c0edb213e7ad20e1d8fa`.
Logs are under `reports/phase4-wmt14-artifact-{full,inference}-probe-v1.log`.
The loader recognizes this explicit artifact profile, but model construction
and all WMT14 GPU/quality gates remain unsupported and unqualified.

The separate CPU check `setpriv --inh-caps=-all env RUSTC_WRAPPER=
CARGO_BUILD_JOBS=4 cargo run --release --locked -p ncm-backend-cuda --features
cpu --example wmt14_unsupported_probe -- artifacts/canonical/wmt14-inference-v1`
loaded the checked inference artifact and confirmed `Codec::from_artifact`
rejects the unimplemented WMT14 model explicitly before device upload. Its
source SHA-256 is `a0469be0acc04a6f1aec6b58145b3def74079dba9c95fc080c8754e125fd2dc1`;
focused strict Clippy passed. This negative check does not implement inference.

The first export's provenance included absolute helper paths, so its manifest
was location-dependent. That initial artifact and the v1 exporter/probe sources
remain retained under ignored `artifacts/` and `reports/`. The corrected exporter
records stable helper and tokenizer asset names with their hashes instead. Two
exports from distinct source and tokenizer directories were byte-identical in
every file; a third final export also matched. The final candidate is
`artifacts/canonical/wmt14-inference-v1-final`, with manifest SHA-256
`01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81` and
corrected exporter SHA-256
`4903be2f595b2d9835cc816c996bd581e679158478c7ed73e541dad68c8544b9`.
The revised probe, SHA-256
`27e659443e07c843449d605d716593a6553b2c140b0da51fc5c1181ad50d187e`,
passed full and inference-only loads of the final artifact. The explicit
unsupported-model probe also passed on the byte-identical candidate. These are
CPU artifact checks, not model execution.

Full `setpriv --inh-caps=-all env RUSTC_WRAPPER= CARGO_BUILD_JOBS=4
CUDA_COMPUTE_CAP=86 ./tools/check.sh --cuda` passed after the loader change,
including 18 real-CUDA operator cases; log
`reports/phase4-wmt14-artifact-cuda-check-v1.log` hashes to
`7d14f17df9bea978dc67b188e31a2b3933f5b5d34dfdd6f9c0876480b7f0fd1c`.
Only exporter provenance and the probe's expected manifest hash changed after
that check; focused exporter tests, real CPU loads and probe Clippy passed again.

The next CPU-only semantic preflight must validate the exact WMT14 codec
configuration and compare its expected 63 tensor names, shapes and dtypes with
the checked artifact before any upload. WMT14's 151,680-token vocabulary and
576,011,264 codec tensor bytes exceed the existing generic 65,536-token and
256 MiB codec limits. Any allowance must be profile-specific and retain the
prior Sudoku, TinyStories and XSum limits; it must not make the existing codec
execution path accept WMT14 until its larger allocations and operations have
been implemented and validated.

The host-only `CodecConfig::validate_wmt14` now uses explicit WMT14 vocabulary
and weight limits while the existing profiles retain their previous bounds.
The pinned codec fixture SHA-256 is
`a7f85e20ad9c1aac8aa74cceb9c5f9fa8bfa6b4d9a875179e20cbc370bfd69be`.
`cargo test --locked -p ncm-core` passed profile, geometry, token/code bounds and
mutation checks. The real CPU `wmt14_codec_preflight` verified all 63 tensor
names, shapes, FP32 dtypes and finite values (576,011,264 bytes; 313 codes),
without allocating a backend or running inference. Its source SHA-256 is
`7e0bb78e1764777f74da8444ae219e1c15f9014a70548152699ba785245dd560`;
`reports/phase4-wmt14-codec-preflight-v3.log` records the successful run.
The first build failed on an iterator type mismatch and is retained as v1.
The existing codec loader still rejects WMT14 execution explicitly.

The host-only generator inspection checks the exact model and tokenizer
configuration, 184 FP32 weight names/shapes/finiteness, and the two level and
attention-mask buffers, for 186 tensors total. It uses the pinned model fixture
(SHA-256 `7f07e8237dabdd38c527c57a97ea5249b46b2b23fbde0985850aaceaaa8c8b68`)
and does not construct a backend. On the final artifact, `setpriv --inh-caps=-all
env RUSTC_WRAPPER= CARGO_BUILD_JOBS=4 cargo run --release --locked -p
ncm-backend-cuda --example wmt14_generator_preflight --
artifacts/canonical/wmt14-inference-v1-final` passed; log
`reports/phase4-wmt14-generator-preflight-v3.log`. The inspection module and
real probe source SHA-256 values are
`4e6769aa71561361ab318feaa029e4c6122fdb537784fc5d1e15e835fa3fbd23` and
`946b2fadd4de7a9b26c9d80bcefa2ad5e74de59992c2959b64597b305b2c103e`.
Focused core/model/IO tests, configuration-mutation checks and strict Clippy
passed. The initial model test build failed on an error-type conversion and was
corrected; its v1 log is retained. Full `setpriv --inh-caps=-all env
RUSTC_WRAPPER= CARGO_BUILD_JOBS=4 CUDA_COMPUTE_CAP=86 ./tools/check.sh --cuda`
passed, including 18 real-CUDA operator cases; log
`reports/phase4-wmt14-semantic-preflight-cuda-check-v1.log` SHA-256
`a904d89f30dbace52d00ab25f5f932d8e1ca282ce7c411c71d7a4d0c865811e7`.
The explicitly rebuilt XSum strict-CUDA CLI still matched the 32-row Python
reference (`reports/phase4-wmt14-semantic-xsum-cli-v1/result.json`, SHA-256
`5abdf14f2e56a262b41474a197c36ee20f4d131fab45c789adec79d52f0742fc`), and
the rebuilt CPU probe continued to reject WMT14 model execution. No WMT14 model
or GPU inference is claimed.
