# WMT14 first Metal CLI comparison

Commit `2c5b748d6316debe1f79be0c5a09c4cf6285fff4` adds the opt-in
`sample-wmt14` command with explicit CUDA and MLX backends. CUDA selects
the [one-pass final selector](cuda-fast-argmax-plan-v1.md). It uses the
uncached generator. MLX uses the native backend with the same shared
generator.

The three four-row batches start at corpus offsets 0, 2180 and 2656,
with 96-token prefixes from the frozen WMT14 tokenizer record. The CUDA
CLI reproduced all 3,756 codes and 768 raw IDs per mode at CFG scales 0
and 1. The focused release build and Clippy passed after changing the
initial fixed-width output slicing to the recommended `as_chunks`.
`tools/check.sh --cuda` then passed 524 report tests, three data tests
and 18 real CUDA operator cases.

On Apple M1 Ultra, the release CLI build and focused Clippy passed. All
three scale-zero batches and the scale-one batches starting at 0 and
2180 matched pinned Python CUDA in every code and raw ID. The
scale-one batch starting at 2656 returned the existing
`tied or nonfinite WMT14 head maximum is unsupported` error. Each of its
four rows matched the retained Python codes and IDs when invoked alone.
This isolates a batch-shape sensitivity on these inputs. The error does
not distinguish a tied maximum from a nonfinite value.

On both devices, omitting the scale produced the same result as
explicit zero. Negative and NaN scales were rejected. The initial Metal
wrapper stopped at the failed four-row case. The unchanged Metal suite
was run separately and passed 414 report tests, three data tests and
18 real Metal operator cases. Mac service snapshots show the same PID.
The four-row CFG failure remains open. Full Metal task quality,
performance and memory are still pending.

The ignored bundle `reports/phase4-wmt14-cli-v1.a9xGWa/` retains the
fixture builder, shell wrappers, pinned inputs, outputs and full logs.
The CUDA, Metal batch and Metal singleton comparison SHA-256 values are
`3a6dde1112329f564bc2b98254213116b5112d6415777851f1284999f410f3e7`,
`c1b555c44b01027893983f5e02e95b26c41a6247c11072a2d27afa7119a9e3bc`,
and `0f7741f9d5feff9b5361c910e63e7b8ffdf70b74c84e398044eb773ae21e2166`.
The Linux and Mac full suite logs hash to
`65c8ebf06b8b54f51cdbc3c424bd0682f30f730dfd8b259cd9a285be8aacbc64`
and `1a28b2d02174ec3bf897be49d4d17f1f15a674c854369467e7a518232ab8260f`.
