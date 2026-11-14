# Byte-level decode allocation — dependency change for review

**User approved; final gates pending.** The prior [Apple gate](bytelevel-final-apple-gate-v1.md)
remains 76/78 because Qwen decode and skip-special decode miss the
process-level median interval limit. This isolated proposal changes the pinned,
vendored `tokenizers` 0.22.2 byte-level decoder to use one byte buffer instead
of allocating a temporary buffer for each token. If a token contains an
unmapped character, the decoder rolls back that token's mapped prefix and
copies the entire original token, preserving the existing fallback. It reuses
the byte buffer for valid UTF-8 and retains the existing lossy conversion for
invalid bytes. No dependency version, assets, schema, or oracle changes.

The differential regression compares against the original decoder over byte
alphabet characters, unknown Unicode, mixed known/unknown token contents,
special-token text, and malformed UTF-8 across token boundaries. Tokenizer
tests and all four pinned asset probes pass on Linux and Mac. The exact
candidate passes `tools/check.sh --cuda` with umask 077 and
`RUSTC_WRAPPER= NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86` (483 report
tests, 18 CUDA operator cases), and `tools/check.sh --mlx` using the pinned
prebuilt MLX libraries (373 selected report tests, 166 platform-skipped,
18 Metal operator cases; 110 Linux-only report modules excluded). The tested
decoder and regression source hashes match across platforms.

Three matched fresh processes per binary, five warmups and 30 trials, used the
pinned Qwen and GPT-2 fixtures. Qwen decode and skip-special median ratios
versus the current implementation were **0.71344/0.74195 on Linux** and
**0.69421/0.71671 on Mac**; output hashes match. GPT-2 decode ratios were
**0.70808/0.71681 on Linux** and **0.68187/0.69498 on Mac**. The Mac GPT-2
encode ratio was **1.09738** even though the change only affects decoding;
this small diagnostic cannot establish its cause or rule out a regression.
These are diagnostic samples, not acceptance results. The Mac sample retained
process activity and thermal observations; WindowServer was visible in ten
samples, median reported CPU 9.35% and peak 12.4%, with no thermal warning.

Each of the three matched process pairs showed a decode improvement: Qwen
decode candidate/previous ratios were **0.732/0.659/0.739 on Linux** and
**0.711/0.741/0.655 on Mac**. Mac GPT-2 encode process medians were
**1.149/1.272/1.143 ms** previous and **1.162/1.285/1.261 ms** candidate;
the third pair accounts for most of the median-of-process difference, but
three pairs cannot establish why. For Qwen, the maximum final process HWM
was **76,619,776 / 76,464,128 bytes** previous/candidate on Linux and
**67,485,696 / 69,844,992 bytes** on Mac. These are observed process HWM
values in a small diagnostic, not a memory acceptance result.

Clean exports reproduce all 24 diagnostic source inventories and matching
output hashes. The measured prototype has the same logic as the review patch,
which adds an explanatory comment and the regression test. [Exact patch](qwen-decode-review-v1.patch)
SHA-256 is `0e9bc5ab1b9a3ec72f30048a4be8776131010162f30edc2e6fc28e73aeba7f71`.
[Raw diagnostics, activity, tests, probes and full checks](raw/qwen-decode-review-v1.tar.xz)
have SHA-256 `1f91437f41f0e717b2c711a389d35e1c302eb1be490014be9ab9871d891d89ea`.
Integrate the approved patch and repeat the Linux-first and Apple gates;
the current gate results do not transfer automatically.
