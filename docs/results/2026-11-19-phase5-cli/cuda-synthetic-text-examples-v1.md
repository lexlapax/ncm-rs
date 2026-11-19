# CUDA synthetic text examples screen

2026-11-19. The four requests in the draft [setup guide](../../USAGE.md#synthetic-text-examples--cuda-development-screen)
were screened as individual CUDA processes using the previously built `cuda-cudnn` CLI SHA-256
`cbd3d8d1fa39cbd878c6ff5f41a4f32aafe726d7706ee7c0dc28e0fd4e3a9b67`. The command environment
was `setpriv --inh-caps=-all`, `OMP_NUM_THREADS=1`, and the reference cuDNN `LD_LIBRARY_PATH`.
The tests used the guide's synthetic inputs/options but substituted the retained artifacts and
output directory below; they were not run from a fresh clone or on Metal. A desktop shared the GPU.

| Profile | Retained qualified artifact | Device | Output SHA-256 |
| --- | --- | --- | --- |
| TinyStories | `artifacts/canonical/phase5-guide-tiny-replay` | `cuda-cudnn` | `0f61c8acd53cfe12db841c0d88c293f37920ba120b876dd58b1bcd41b0403e81` |
| XSum | `artifacts/canonical/phase5-guide-xsum-replay` | `cuda-cudnn` | `82eb25bd32b2e14bb52b977abfe0deeee8cafc1e70477b726265a2d23786460e` |
| WMT14 | `artifacts/canonical/phase5-guide-wmt14-replay` | `cuda` | `a3296512f15198892c9b05c78548ad592bc4f0b68e96bd0eadae2b4ab26594ef` |
| LM1B small | `artifacts/canonical/phase5-guide-lm1b-small-replay` | `cuda-cudnn` | `cd87171d7f866ad664cedf252c8eee86e857d5c66a00855468da73e009a423c6` |

The four artifact manifests matched the qualified hashes in the
[export investigation](linux-export-repro-v1.md). LM1B used the default sibling sidecar with
SHA-256 `9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc`. Each result
contained one row in `codes`, `ids` and `texts`, with a decoded string, as checked by the guide's
validation snippet. Inputs and outputs remain under ignored `reports/phase5-text-examples-v1/`.
These are smoke examples, not claims of exact output on all devices, evaluator decoding parity,
model quality or performance. The fresh-clone guide, other profiles and Metal gates remain pending.
