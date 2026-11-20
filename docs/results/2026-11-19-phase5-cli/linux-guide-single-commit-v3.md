# Linux guide replay from a single fresh clone

2026-11-20. A fresh remote clone at `c63b205eb5c5adeca4ef2a787e71b9460ba7a274` followed the
current draft's 15 Linux shell blocks in one uninterrupted run. The wrapper extracted blocks
1, 2 and 4–16 unchanged from `docs/USAGE.md` (block 3 is macOS), adding `set -e` and progress
headings. It ran under `setpriv --inh-caps=-all` and `PIP_NO_CACHE_DIR=1`. The ignored wrapper
`reports/phase5-linux-full-v3-wrapper.sh` SHA-256 is
`727f4bfdfee94eb9efcf7c5b6938573f1cea54e33c3f0820c42d546c663a492f`; the captured log
`reports/phase5-linux-full-v3.log` SHA-256 is
`3f030cd7b12e9660701713d3c27aa50b141cf53ac9cf28719b24555d93953c49`.

- The release CUDA CLI built (SHA-256 `6194111fe3860584492676c6d14f50124a77f8dcf18f316a959b8abd16a20278`)
  and `tools/check.sh --cuda` passed, including 18 operator cases. The retained ignored operator
  report `reports/phase5-linux-full-v3-cuda.json` SHA-256 is
  `f03188b17edb2977042052f88196b7ff8a31566b97dfd42c45156ffef070e055`.
- Four tokenizer identities matched their lock. All nine checkpoints were downloaded afresh,
  all guide manifest checks passed, and XSum and LM1B rotary references matched their pins.
  Medium/hard Sudoku matched the user-approved provenance-only candidate hashes; their full
  qualification remains pending.
- The synthetic Sudoku alias and unified outputs matched. TinyStories, XSum, WMT14 and
  LM1B-small examples each yielded one decoded row; output hashes matched the earlier
  [resumed replay](linux-guide-resumed-v2.md). Medium/large LM1B were exported but not sampled.
- The build-tree Python 3.14.9 again logged the documented `_ssl` `TLSv1_method` import failure;
  the guide uses it only for offline export. System Python handled downloads and pip.

The five nested exporter worktrees were inspected: easy and TinyStories contained only the expected
untracked exporter; medium/hard and XSum contained only their expected tracked exporter change
(XSum also had ignored bytecode); LM1B-small contained only ignored bytecode. Their exporter hashes
matched the guide. The worktrees and inspected temporary clone were removed; the wrapper, log and
operator report remain ignored locally. The GPU was shared with a desktop.

This verifies the **documented Linux development blocks at this commit**, not an independent
quality or performance gate, the remaining Sudoku qualification, an entire Phase 5 release gate,
or any macOS/Metal preparation or inference. The Mac guide is still incomplete.
