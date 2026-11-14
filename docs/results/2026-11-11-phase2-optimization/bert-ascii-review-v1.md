# BERT ASCII normalization — dependency change for review

**User approved; final gates pending.** The approved Phase 2 plan requires user
review for dependency changes, and the user approved this patch. This candidate adds an ASCII fast path to the
already-vendored pinned `tokenizers` 0.22.2 BERT normalizer. ASCII contains no
CJK or decomposable accents; the path performs cleaning only when needed and
lowercases only when an uppercase ASCII byte is present. Non-ASCII inputs use
the original implementation. The dependency version, assets, schema, oracle,
and numerical tolerances do not change. This is an additional upstream source
modification beyond the previously approved BPE patch.

A CPU profile on the Mac identified normalization transforms, allocations, and
copies among the sampled encode costs. Three matched fresh processes per binary
used the canonical BERT fixture, five warmups, and 30 trials. Median-of-process
encode time changed from **1.03924 to 0.99001 ms on Linux** (ratio 0.95263)
and **1.25658 to 1.20750 ms on Mac** (ratio 0.96094). These are small
diagnostics, not acceptance runs or a guarantee that the earlier Apple p95
failure will close. The measured prototype has the same logic as the review
patch, with formatting/comments and benchmark source-inventory additions in
the latter. Decode differences were small and do not establish a speedup.

The candidate's differential test compares the complete `NormalizedString`,
including its alignments, against the original sequence of normalization steps
for all ASCII bytes, generated mixed strings, every BERT configuration
combination, and pretransformed inputs. The four canonical tokenizer profiles
and rejection cases pass on Linux and Mac. `tools/check.sh --cuda` passes under
umask 077, including 18 real-CUDA operator cases. The exact candidate's Mac
`tools/check.sh --mlx` passes, including 18 real-Metal operator cases; it
selects 373 report tests, with 110 Linux-specific tests excluded and 166 of
the selected tests platform-skipped. The exact source hashes match across the
Linux checkout and isolated Mac review worktree. Both final platform acceptance
matrices remain pending; no failed control is discarded.

The review patch SHA-256 is
`ac42bc3004f978369915eef971fe95fcad5b13642040af80493778b39d96b717`;
`vendor/tokenizers-0.22.2/src/normalizers/bert.rs` SHA-256 is
`1c8b1602de94ed84d3f3a0d05f32d3f6f4f398ce9d5cee67f2bacd28d53bc940`.
The three asset examples add that source file to their existing `source_sha256`
maps; the report schema is unchanged. [Raw measurements, patch, regression,
probes and logs](raw/bert-ascii-review-v1.tar.xz) SHA-256:
`b2e2ca2e11b5d43569bb85ee5f3e70a5a2783125d42448e022dc45703ca80d73`.
The [Mac CPU profile and temporary harness](raw/bert-encode-profile-v1.tar.xz)
has SHA-256 `e5ccee53174add4dfb125a5c52f4b2252cacc6955392fa29f895c69943cfeea9`.
