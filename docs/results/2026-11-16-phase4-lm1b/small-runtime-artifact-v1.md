# LM1B-small bounded artifact and host checks

The explicitly named `lm1b-small-inference-v1` artifact preserves the pinned
checkpoint and BERT tokenizer identities and all 351 tensors: 213 generator,
100 codec, 36 retained training-state and two inference-distribution tensors.
The existing preparation profile remains separate and unsupported. Two CPU
exports round-tripped all tensor bytes and matched byte-for-byte; their manifest
SHA-256 is
`58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76`.
The exporter SHA-256 is
`a8dd547b02b06a557c581c885e4eea289a5a8ffa60a6682556737409c06caf52`.
Artifacts remain in ignored `artifacts/canonical/lm1b-small-inference-v{1,2}`.

`ncm-io` recognizes only this explicit small profile with a 448 MiB per-file
and 1024 MiB total budget, without changing other profiles. Its expected nine
files and four tensor bundles are exact. Full loading retains training state;
inference-only loading validates it as a stream and does not retain it.
Negative tests cover missing bundles, per-file and total size limits, and
unsupported medium profiles. Full and inference-only CPU processes loaded the
actual artifact and checked the 213/100/36/2 inventories. The probe SHA-256 is
`d8d38074688f83bc1671d1ba63d9b2767bc7e0dff0250c5e170804c1b8e1c953`.

An opt-in host preflight applies the existing pinned LM1B-small generator,
codec and inference-distribution schema checks to the checked artifact. The
normal `Codec::from_artifact` explicitly rejects this profile before backend
upload: model execution remains unsupported. The first negative probe instead
reached an unsupported `code_corruption` config field before the profile-level
error; that failed diagnostic is retained in
`reports/phase4-lm1b-small-model-unsupported-v1.log`. The explicit guard was
added and the corrected CPU probe confirms the intended rejection. The host
preflight and negative-probe binary SHA-256 values are
`813d5bb593c8b7babfa7c8debb4733173fad9748edb8b284bf8e6136b4fbb56c` and
`756601d2298239df4dfccbbf9661a7699c71340bd8abea371070dd2ce4718d2c`.

Focused exporter, `ncm-io` and all-feature model tests and strict lint pass.
Full `tools/check.sh --cuda` passed 18 real-device operator cases; log SHA-256
`cb50af1c333aca1a55bdb11dc593fab4efcb80d86a1f0a7ed5bff6e96c3ff5c5`.
Artifact loading is integrity validation,
not manifest authentication, model support, parity, quality or performance.
The ordered XSum and WMT gates, LM1B runtime and both hardware gates remain
open; medium and large profiles are not enabled.
