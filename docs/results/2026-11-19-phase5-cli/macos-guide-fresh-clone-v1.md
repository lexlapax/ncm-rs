# macOS / Apple Silicon guide replay from a fresh clone

2026-11-20. A fresh remote clone of `https://github.com/lexlapax/ncm-rs.git` at
`88d29562331f6fa3a492f712b6fad781fb6aef4f` on `macstudio` (Apple M1 Ultra, macOS 26.7.3,
Rust 1.99.1) executed the two documented macOS shell blocks extracted unchanged from
[`docs/USAGE.md`](../../USAGE.md) (block 3 and block 17, with `set -euo pipefail`).

- **Setup, build, and checks (block 3):** `uv venv --python 3.14.10 .reference/mlx-venv` and
  `tools/reference/requirements-macos.lock` installed `mlx==0.32.2`, `mlx-metal==0.32.2`, and
  `numpy==2.5.4`; `.reference/mlx-native` was populated from the wheel; `cargo build --release --locked -p ncm-cli --features mlx` produced `target/release/ncm` (SHA-256 `bd18bce000d843102b828e4f6ca7873f0cbfb78b5ffaec82cfbebd7ff3956723`, distinct from the main checkout binary solely due to the embedded local rpath `/private/tmp/ncm-rs-mac-guide-fresh-v1/.reference/mlx-native/lib`); `./tools/check.sh --mlx` passed (workspace fmt/check/clippy/test/doc, 493 macOS report tests, reference tests, MLX release clippy/test/doc, and `mlx_probe` 18 cases).
- **Verified artifacts and synthetic examples (block 17):** copied verified Linux-exported canonical artifacts (`sudoku-easy-qualified`, `tinystories-phase4-v1`, `xsum-inference-rope-v2-final`, `wmt14-inference-v1-final`, `lm1b-small-inference-v1`) and `lm1b-reference-rope-v1.safetensors` into `artifacts/canonical/`. All six `shasum -a 256 -c` checks passed. Synthetic Sudoku-easy `ncm sample` and `ncm generate` matched byte-for-byte (`ac078dcba270fe6a069b2e0e42d27336897bf88ebb93d36a2f883e6dd0eb29a2`), and TinyStories (`c150908c02da9b00f998c55c00155dcf165c7cfcbf69ed3336789d00268fe078`), XSum (`82eb25bd32b2e14bb52b977abfe0deeee8cafc1e70477b726265a2d23786460e`), WMT14 (`a3296512f15198892c9b05c78548ad592bc4f0b68e96bd0eadae2b4ab26594ef`), and LM1B-small (`3f28b8157b25387ce9c191d3dbd40ce7f8fc2bb0fe9d7789bd2d88b30624c7e0`) each produced one valid decoded row.
- **Cleanup:** `/tmp/ncm-rs-mac-guide-fresh-v1` and temporary extracted scripts were deleted after verification.

This verifies the documented Apple Silicon / MLX fresh-clone setup and synthetic examples. Offline checkpoint export remains a Linux procedure; do not install the Linux Torch lock on macOS.
