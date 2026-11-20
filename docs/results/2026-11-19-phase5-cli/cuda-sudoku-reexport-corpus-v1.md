# Sudoku re-export full-corpus CUDA comparison

2026-11-20. The approved provenance-only medium/hard candidates were compared to the retained
originally qualified artifacts on the existing Phase 3, 2,000-board-per-difficulty datasets.
The CUDA-feature release CLI `target/release/ncm` SHA-256 was
`6194111fe3860584492676c6d14f50124a77f8dcf18f316a959b8abd16a20278` (built from the current
checkout with `cargo build --release --locked -p ncm-cli --features cuda-cudnn`). The desktop GPU
was shared; this is not timing evidence.

Both artifacts were invoked via `ncm sample ARTIFACT cuda INPUT` in separate processes using
`setpriv --inh-caps=-all`, `OMP_NUM_THREADS=1` and the reference cuDNN library path. The command
uses the existing default batch size 128. Full original and alternate stdout matched byte-for-byte
for both difficulties, including every code and ID. Before and after the runs, binary, dataset,
input and manifest hashes matched the identities below.

| Difficulty | Dataset SHA-256 | Input SHA-256 | Output SHA-256 (each artifact) |
| --- | --- | --- | --- |
| Medium | `7d5ff3b523fff2e8dc55a94407ae8aa505913b3ada64e6c0fc537921bdcc15ca` | `82c6c1ff6619dad7ca4ab3295762a2a190186cf6791d4093400a13e7512e87da` | `5163a99d685cfd35095dfbd728e4dda7abd836e50296fc87488e6548fe9799f3` |
| Hard | `828998509024a6d1fb04b956da055f40595da83253ccaae7ce7f9b665d79430c` | `47c51b66fcbf9d3335778c3870893ba714b9ba462c5199b92a79f6d190df3e48` | `c1ba7ba8bfd2470bd479c4eb4f583630370e1e8d53206d8137b8643fd26c9b5e` |

Original manifest hashes were medium `facff42955d42294dc94f33d54688dc53046c1f9cc029e903f1ffe787ce45219`,
hard `9ed3e86fdc3d193c88e1f4aae8f58a54019f985bdcf2acb9c851772c51dad86b`; alternate manifest
hashes were medium `9bc8e30673e426d6b10ef14204e2b0a0162fdc6a38c92e453fe5c156fab375c4`,
hard `25ec837745601d69435a41f7ef5131b70d5fe17e1ca908e7b660d15531031158`. Original artifact
paths were `artifacts/canonical/sudoku-{medium,hard}-phase3`; alternate paths were
`artifacts/canonical/sudoku-{medium,hard}-reexport`. The input files are
`artifacts/canonical/sudoku-data-phase3/{medium,hard}-2000-inputs.json`; the dataset files have
the same names without `-2000-inputs`.

A separate local script verified that each input equals its dataset's input rows, output arrays
contain 2,000 rows and 322 codes per row, and the two complete JSON objects match. It invoked the
existing `score` function in `tools/bench/phase3_sudoku_score.py` on the alternate output and
retained dataset references. Results match the published [Phase 3 table](../2026-11-14-phase3-sudoku/README.md):

| Difficulty | Solved | Cell accuracy | Valid | Clues preserved | Malformed | Code / ID mismatched boards vs original |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Medium | 1,928 | 99.793827% | 1,928 | 1,993 | 1 | 0 / 0 |
| Hard | 1,726 | 98.352469% | 1,726 | 1,997 | 1 | 0 / 0 |

The raw output and score summary remain ignored locally under
`reports/phase5-sudoku-reexport-corpus-cuda-v1/`; summary SHA-256 is
`9d2558d8bec08f7487f53f7f23fd022a90f27b7592fb3e4a413f004768ab2c10`. This compares the two Rust
artifacts and scores against dataset references; it is **not** a new direct Python output
comparison. It does not qualify Metal, performance or a full Phase 5 release gate.
