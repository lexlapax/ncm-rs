# Linux fresh-clone guide replay: Sudoku-easy slice

2026-11-19. This replay checks the draft [setup guide](../../USAGE.md) for **one** CUDA profile,
not the full setup, CLI equivalence, quality, performance or release gate. A fresh remote clone at
`c35a41d05208d00d128a6af40b15ff418f0f7312` was created under ignored
`reports/phase5-guide-sudoku-fresh-v2/`. The shell blocks were extracted unchanged from the guide's
Linux CUDA build, Linux CUDA check, checkpoint download, Linux Sudoku-easy export and synthetic
Sudoku-easy request sections, in that order. The wrapper added `set -e` and progress headings;
its SHA-256 was `44f30f6b8ca78fd2216e0dfccfc35fdb14e844d36d6d03d4e5da7ea60ee4c7d2`.
The wrapper ran under `setpriv --inh-caps=-all` with `PIP_NO_CACHE_DIR=1`. It did not execute the
guide's tokenizer, other profile, Apple or full-guide sections. Dependencies, the pinned checkpoint
and CPython source were downloaded; no pre-existing artifact or reference environment was reused.

- The `cuda-cudnn` release build completed; CLI SHA-256 was
  `cbd3d8d1fa39cbd878c6ff5f41a4f32aafe726d7706ee7c0dc28e0fd4e3a9b67`.
- The guide's `tools/check.sh --cuda` passed. Its CUDA operator report recorded 18 results,
  `passed: true`, with `model_quality: null`; retained ignored report
  `reports/phase5-guide-sudoku-fresh-v2-cuda.json` SHA-256 is
  `737c59f8c755b5134f46e6ab1bf38fcd95993f25a23aedea3fe3fcce56ee5cf0`.
- The fresh Sudoku-easy checkpoint passed the pinned size/hash verification. Its historical export
  reported bit-exact tensor roundtrips and produced the qualified manifest SHA-256
  `ad6a99ad7d5ccc069e0fedf26a85f755e06cd76590463e1e04e17363e35d9832`.
  The ignored export report SHA-256 before cleanup was
  `b2cf2f9f553428c1f6a1935c1ce759a41ccac5852441c99d7f9a7ffae3ae682e`.
- The guide's synthetic blank-board request ran `sample` and `generate` on CUDA; both outputs
  were byte-identical with SHA-256 `ac078dcba270fe6a069b2e0e42d27336897bf88ebb93d36a2f883e6dd0eb29a2`,
  and the guide's output-shape assertions passed. Input SHA-256 was
  `d7770de15c156ac6dd8983236354f0785e8739a144ea796dc8a9c5a1a3508adb`.

The CPython build logged the known `_ssl` import failure (`TLSv1_method`); the guide uses that
build-tree interpreter only for offline export, and its Python/Torch version check and export
passed. The complete captured log is ignored at `reports/phase5-guide-sudoku-fresh-v2.log`,
SHA-256 `b23dde36574110e6e08c256fcf3521551e563a4edd656cb30c61a25963709dd1`. The nested
historical exporter worktree contained only the expected restored exporter, whose hash was checked;
it was removed after inspection. The remaining fresh clone was inspected and removed. An active
desktop shared the GPU; no benchmark or absence of competing work is claimed. Other profiles,
complete fresh-clone guide replay on both machines, and all remaining Phase 5 gates are pending.
