# XSum CLI preparation

The prepared `sample-xsum` command accepts an XSum inference artifact, the
explicit `cuda-cudnn` device and a JSON array of 1–128 tokenized, padded
512-token prefixes. It invokes the existing native temperature-1, top-k-1,
top-p-0, CFG-0 generation path and returns codes and final token IDs. Empty,
oversized, malformed, all-padding or out-of-range prefixes fail before GPU
allocation. Other device selections return an error without fallback. This is
a restricted interface; CFG, stochastic selection and XSum MLX are not exposed
by this command.

The default-feature check, CPU and strict-cuDNN Clippy checks, release build,
and the two CLI unit tests passed. The first strict-feature test build failed
because the concrete backend's `CodecBackend` trait was not imported; the
import was added and the test passed. Logs are
`reports/phase4-xsum-cli-default-check-v1.log`,
`reports/phase4-xsum-cli-default-clippy-v1.log`,
`reports/phase4-xsum-cli-cpu-clippy-v1.log`,
`reports/phase4-xsum-cli-clippy-v1.log`,
`reports/phase4-xsum-cli-build-v1.log`, and
`reports/phase4-xsum-cli-cpu-test-v{1,2}.log`.

Real-CUDA CLI output comparison and end-to-end CLI performance/memory remain
pending until the current source-bound full quality run finishes. These
preparation checks do not establish the XSum Linux or MLX gate.
