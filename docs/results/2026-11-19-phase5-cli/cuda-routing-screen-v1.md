# Phase 5 unified CLI: CUDA routing screen

2026-11-19. This is a development screen, **not the Phase 5 release gate**. One fixed request per profile was run in three fresh processes: the pre-change alias, the candidate alias, and the candidate `sample` command. Their stdout bytes, including the newline, matched exactly in every row. LM1B used seed 42 and the pinned reference rotary export. TinyStories used seed 42. XSum and WMT14 used top-1, CFG 0. Sudoku used the two retained easy inputs. No text encoding, decoding, quality or performance comparison is implied.

| Profile / device | Input SHA-256 | Common output SHA-256 |
| --- | --- | --- |
| Sudoku / cuda | `7dbb5c739c25e6f8a1ede490278261d4d800ec8529a14c038b08e11a4278c41c` | `bd2a86aa8d3c626178bfd1d07bd2bbbdfee2a98643298c02c189e59bb2a06212` |
| TinyStories / cuda-cudnn | `3aa8ac761c5a23906d30fb86ca2db09c9b391142a956cdb4263570adb94bfbe6` | `7c828e7bcf538bd65a60762429e4df318c6d71a52e3d8cb20210bfaf14eb6837` |
| XSum / cuda-cudnn | `74703e40c8fd26fff5a6d919c6d40550a39a33b94183455739093947b7ef2489` | `87be072a79210d62215efaa82f801c64612f6868cf545a3fb888ca14aee6cdca` |
| WMT14 / cuda | `8b0fd6d29cacfd8acf1e033a3664aec21c2a28976af9f9fc1ad77d62b1c8a825` | `17257d936f55b159c8cdc8ce9a465bf9d0498a446d544215c171eae82f2a604a` |
| LM1B small / cuda-cudnn | `a283936d3db1e156d26a0ace722f476f1895782f23729587a45c4f838168e7dc` | `520ae6b5372b2013bda8235dfec40c85265716014ac4a9ba58fec03419be5d71` |
| LM1B medium / cuda-cudnn | same | `d9c1c40663e66cf472846e3781ca4995bab998b5bc86ec0c675fe0b29c506488` |
| LM1B large / cuda-cudnn | same | `bbdc24f6d0bd2150c41af25fb4a37835e9a191fcce5fe181461863421b9bfbf9` |

The baseline was built at `702168c` with features `cuda-cudnn,wmt14-experimental,lm1b-experimental` (binary SHA-256 `f10abaaec672dd7c0c62034972fb4b15fc204f2b1f5693e4a09f5e9640767098`). The candidate used `cuda-cudnn` alone (binary SHA-256 `55951fddb0ca0bbef95fc388eaac931cd36a55075bdb6083ed2121a917d8fda1`). Separate Cargo target directories were necessary: a first attempt sharing one target directory reused the baseline binary, detected by identical binary hashes; those runs were discarded. Binaries, inputs, and stdout are in ignored `reports/phase5-cli-equivalence/` locally. Both binaries were built `--release --locked` with `NVCC_CCBIN=/usr/bin/g++-15`, `CUDA_COMPUTE_CAP=86`, `CARGO_BUILD_JOBS=4`, and the project-local cuDNN link directory. Runs used `setpriv --inh-caps=-all`, `OMP_NUM_THREADS=1`, and the reference environment's cuDNN `LD_LIBRARY_PATH`.

The TinyStories input was `[[0]]`, LM1B input `[[101]]`, and Sudoku input `artifacts/canonical/sudoku-data-phase3/easy-two-inputs.json`. XSum used the first converted corpus row, truncated to 512 and padded with 50257; WMT14 used the first source string, encoded with the pinned Qwen tokenizer from `artifacts/upstream/tokenizers-v2/qwen3`, truncated to 96 and padded with 151643. The rotary export is `reports/phase4-lm1b-level7-replay-v1/reference-rope-tables.safetensors`, SHA-256 `9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc`. Artifact identities are in the Phase 4 evidence and can be checked against their manifests.

Pending: the full command surface and text gates, reproducible setup, before/after performance, and Metal equivalents. `--reference-rope` is currently required for unified LM1B; the intended no-extra-argument release surface is not yet implemented. The unified command currently rejects `--decode` for profiles other than LM1B and does not support `--text`.
