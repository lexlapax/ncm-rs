# Phase 5 Linux fresh-clone Sudoku-easy slice

2026-11-19. This is a **single-profile development replay**, not the Phase 5 fresh-clone gate.
The remaining profiles, complete usage document, independent quality/performance suite and macOS
fresh-clone run remain pending. The [setup guide](../../USAGE.md#linux-sudoku-easy-development-replay)
was drafted from this run afterwards; the entire document was not followed end-to-end.

## Procedure and results

- Cloned `https://github.com/lexlapax/ncm-rs.git` into a new repository-local temporary directory;
  checkout was clean at `17a1ee6a2179db91198e618b245e3554ccce7bb4`. Created a new Python 3.14.10
  venv and installed `tools/reference/requirements-linux.lock`. Pip and temporary caches stayed
  under the clone. Torch reported `2.15.0+cu132`.
- Built `ncm-cli --release --locked --features cuda-cudnn` using a project-local cuDNN link,
  `NVCC_CCBIN=/usr/bin/g++-15`, `CUDA_COMPUTE_CAP=86`, `CARGO_BUILD_JOBS=4`, and a new clone-local
  `CARGO_HOME`. Rust was 1.100.0; hardware was RTX 3060 Ti, driver 620.70.06. Binary SHA-256:
  `27e43472fffe21a10c4b1f15db3d5bfc08616c7a0805ef60ea4859e9bcd835f4`.
- Planned, freshly downloaded and verified Sudoku-easy checkpoints using the tracked downloader
  and its exact 162,378,392-byte budget. No retained local checkpoints were copied into the clone.
- Downloaded the official CPython 3.14.9 source tarball and checked SHA-256
  `e02d9ac21cebda6e46435369744a92c65cc7cd8669697694122cf97ffacd1e53`; checked every archive path
  and rejected links/devices before extracting. Built project-locally and used the build-tree
  interpreter only for offline export, borrowing the fresh venv's Python 3.14 packages through
  `PYTHONPATH`. As in the earlier investigation, `_ssl` failed to import due to missing
  `TLSv1_method`; no successful general Python installation is claimed.
- In an isolated worktree at historical base `f7aad49c3b904f76c86a3d5e36c3b552d2995000`, restored
  the known historical exporter from `bbdf4222b734c4ebcb6cfa3ed1f23dbdf0c5f58c`. Its verified
  SHA-256 was `cfcae7d716fa7a9692c57019454bd28da5f8098c6d3bec32698795076dc0034e`. The first
  checksum invocation used a mistyped expected hash and failed before export; the source was
  unchanged, the expected hash was corrected from the retained provenance and retry passed.
  This also corrected a transcription error in the earlier [investigation](linux-export-repro-v1.md).
- The exporter reported bit-exact tensor roundtrips and reproduced qualified manifest SHA-256
  `ad6a99ad7d5ccc069e0fedf26a85f755e06cd76590463e1e04e17363e35d9832` exactly.
- A locally created all-blank synthetic 91-ID Sudoku prefix produced the same stdout through
  `sample` and `generate`; output shapes were 1×322 codes and 1×89 IDs. Input SHA-256:
  `d7770de15c156ac6dd8983236354f0785e8739a144ea796dc8a9c5a1a3508adb`; stdout SHA-256:
  `ac078dcba270fe6a069b2e0e42d27336897bf88ebb93d36a2f883e6dd0eb29a2`. This is not a model
  quality result.
- In the clone, `setpriv --inh-caps=-all ... ./tools/check.sh --cuda` passed, including 18
  real-CUDA operator cases. This is not the full model-quality or performance gate.

The temporary clone and its intentionally modified historical-exporter worktree were inspected and
removed after capture. No weights or binary were retained from it. Small local evidence files are
in ignored `reports/phase5-fresh-clone-sudoku-evidence-v1/`: `fresh-cuda-check.log` SHA-256
`8f128bed3abe89b0b305bfb8d03def8bdf54228da18f55389ca51ef0950ccdc8`, `cuda-build.log`
`fd66dcfd1d5042d7823ab0aa180d4503ecaf591f63f007fadea0b7ac0a2eee9b`, `python-build.log`
`a8c6b91b784d1b93711568a7ed7dbffa974288f4f874800c7638382a6ff06c06` and
`sudoku-easy-export.json` `b2cf2f9f553428c1f6a1935c1ce759a41ccac5852441c99d7f9a7ffae3ae682e`.

The documented steps now reproduce this slice in principle but need to be followed as written
from a second fresh clone before claiming the usage document itself is verified. No conclusion is
made for the other profiles or for Metal.
