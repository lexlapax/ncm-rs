# GPT-2 byte-level preprocessing — dependency change for review

**User review pending.** The remaining Apple gate cell is GPT-2 encode's
process-level uncertainty; [the current candidate remains 77/78](blocknorm-apple-gate-v1.md).
This isolated proposal modifies the vendored, pinned `tokenizers` 0.22.2
byte-level pre-tokenizer. It skips a string/alignment reconstruction only
when all bytes map to themselves (`!` through `~`) and the original-coordinate
range covers the entire normalized string. All other inputs use the original
path. The dependency version, assets, schemas and oracle do not change.

The guard matters: a first, unguarded shortcut failed a regression when an
earlier normalizer appended text to an empty original string. That failure is
retained. The corrected test compares complete normalized strings and
alignments, including pre-existing transformations; a separate differential
test compares byte-level splits and original/normalized offsets for ASCII,
Unicode, regex and prefix configurations, and pre-normalized inputs. The
tokenizer suite and all four real-asset profile probes pass on Linux and Mac.
The exact candidate passes full `tools/check.sh --cuda` (483 report tests,
18 real-CUDA operator cases) and `tools/check.sh --mlx` (373 selected report
tests, 166 platform-skipped, 18 real-Metal operator cases). The Mac report
suite excludes 110 Linux-only modules. Tested candidate source hashes match
between platforms. The checks ran from isolated candidate worktrees with
`RUSTC_WRAPPER= NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86
CARGO_BUILD_JOBS=4 ./tools/check.sh --cuda` under umask 077, and with the
existing pinned `MLX_SYS_PREBUILT` and process-local library path via
`./tools/check.sh --mlx` on the Mac.

Three matched fresh processes per binary, using the pinned GPT-2 fixture,
five warmups and 30 trials, showed median-of-process encode time **0.95138 to
0.90885 ms on Linux** (ratio 0.95529) and **1.31312 to 1.22775 ms on Mac**
(ratio 0.93498). Output hashes matched. Mac decode ratios were 1.00120 and
1.00533; these small diagnostics are not acceptance measurements and do not
establish a passing final gate. The live Mac showed other activity before the
sample; exclusive CPU/GPU access is not established. The measured prototype
uses the same logic as the formatted review patch, which adds comments and
regression tests.

An independent clean-export check applies the archived measured prototype
patch to `49a8f16`, verifies every source hash in all 12 native diagnostic
reports against the corresponding candidate or unchanged export, and
recomputes the encode medians and ratios above. All three scope output hashes
match across all six runs on each platform, and each run records five warmups
and 30 trials. The separately published review patch is byte-identical to the
archived review patch. This confirms provenance and does not change the
pending dependency review or either acceptance gate.

[Exact proposed patch](gpt2-bytelevel-review-v1.patch) has SHA-256
`e4cc90d5d6b657c6dbe1129c0809fafa5e40d24a071f5ae57f2e47f4454e03f5`.
[Patch, failed and passing regressions, checks, probes and raw diagnostics](raw/gpt2-bytelevel-review-v1.tar.xz)
have SHA-256 `f7e9254ae166747aa3484b80034d996499b16c6481ad847e4f29a9ee09017e20`.
If approved, integrate the patch and repeat the required Linux-first quality,
performance and Apple gates; neither historical gate automatically transfers.
