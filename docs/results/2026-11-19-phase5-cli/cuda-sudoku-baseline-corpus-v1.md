# CUDA Sudoku CLI full-corpus regression

2026-11-20. The retained pre-change CLI (`reports/phase5-cli-equivalence/baseline-ncm`, SHA-256
`f10abaaec672dd7c0c62034972fb4b15fc204f2b1f5693e4a09f5e9640767098`) ran `generate` on each
of the three retained Phase 3 2,000-board inputs using the original qualified artifacts and CUDA.
Its stdout was compared byte-for-byte with the current CUDA-feature release CLI
(`target/release/ncm`, SHA-256 `6194111fe3860584492676c6d14f50124a77f8dcf18f316a959b8abd16a20278`)
running `sample` on the same artifacts and inputs. The current medium/hard outputs were copied
from the preceding [re-export corpus comparison](cuda-sudoku-reexport-corpus-v1.md), rather than
rerun; easy was run in this screen. All processes used the default batch 128,
`setpriv --inh-caps=-all`, `OMP_NUM_THREADS=1` and the reference cuDNN library path. The desktop
GPU was shared; these are not performance measurements.

| Difficulty | Dataset SHA-256 | Input SHA-256 | Both output SHA-256 |
| --- | --- | --- | --- |
| Easy | `8bead9fdb11f7fce3fefa6efec6c495297d8ee5f91cd0363ce1468eda735aa59` | `3b5176cdd5136ffddcc6975aefe1e86ef1282d48ed7178af7fa5a2d78ef6a45e` | `c0be5340941b4160447becb47fe761ede17ecb8ccb37bb522108168e8f43ced6` |
| Medium | `7d5ff3b523fff2e8dc55a94407ae8aa505913b3ada64e6c0fc537921bdcc15ca` | `82c6c1ff6619dad7ca4ab3295762a2a190186cf6791d4093400a13e7512e87da` | `5163a99d685cfd35095dfbd728e4dda7abd836e50296fc87488e6548fe9799f3` |
| Hard | `828998509024a6d1fb04b956da055f40595da83253ccaae7ce7f9b665d79430c` | `47c51b66fcbf9d3335778c3870893ba714b9ba462c5199b92a79f6d190df3e48` | `c1ba7ba8bfd2470bd479c4eb4f583630370e1e8d53206d8137b8643fd26c9b5e` |

The qualified manifest hashes were easy `ad6a99ad7d5ccc069e0fedf26a85f755e06cd76590463e1e04e17363e35d9832`,
medium `facff42955d42294dc94f33d54688dc53046c1f9cc029e903f1ffe787ce45219`, and hard
`9ed3e86fdc3d193c88e1f4aae8f58a54019f985bdcf2acb9c851772c51dad86b`. Binary, dataset, input and
manifest hashes were checked before and after. JSON equality covers all 6,000 rows' codes and IDs;
a separate local script checked 2,000 rows per difficulty, 322 codes per row and exact equality
of the input files with dataset inputs. The script used `score` from
`tools/bench/phase3_sudoku_score.py` against dataset references. The resulting solved counts were
1,988 / 1,928 / 1,726 (easy / medium / hard); all other metrics matched the published
[Phase 3 table](../2026-11-14-phase3-sudoku/README.md). There were zero code or ID mismatched
boards between the two Rust CLIs.

Raw outputs and score summary remain ignored under `reports/phase5-sudoku-baseline-corpus-cuda-v1/`;
summary SHA-256 is `ac2669c4682595fde32216ab61b49c1767cbbb3404dd5404860f837505376f9c`.
This is not a new direct Python output comparison, performance evidence or Metal validation. The
separate re-export comparison covers the alternate medium/hard artifacts on the same inputs.
