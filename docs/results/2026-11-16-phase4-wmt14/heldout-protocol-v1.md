# Frozen WMT14 CUDA held-out quality protocol

Frozen before inspecting or scoring rows 128–3002. Rows 0–127 are development
data and include the earlier 32-row and 35-row screens. The baseline statistics
and exact Python/Rust development outputs are recorded in the
[development report](cuda-development-quality-v1.md) and the separate CFG
development record linked there. This protocol does not change the pinned
upstream code, checkpoint, tokenizer, decoder, numerical bands or released
generation settings.

Evaluate base (released CFG=0) and the separate explicit CFG=1 diagnostic with
the same verified source and reference rows, contiguous batches of four and a
three-row final batch. Retain every generated code and raw ID. Use the pinned
tokenizer and the explicitly versioned sacreBLEU 2.5.1 environment described in
the development report. Report cased BLEU, lowercase BLEU, chrF, whitespace
length ratio and full metric signatures for the 2,875 held-out rows and all
3,003 rows separately. Development rows are not a held-out result. Score base
and CFG independently; CFG=1 is not presented as the released setting.

For each mode, the following all gate held-out acceptance:

1. All 2,875 Rust raw-ID rows must exactly match upstream Python. Record all
   intermediate code differences and investigate any even if final IDs agree.
2. For BLEU, lowercase BLEU and chrF, Rust minus Python must be no worse than
   -0.5 points, both as a point estimate and at the lower 2.5th percentile of
   1,000 paired bootstrap resamples of held-out rows with seed 42.
3. The absolute Rust/Python length-ratio difference and both endpoints of its
   paired 95% bootstrap interval must be at most 0.01.
4. Input, tokenizer, scorer, capture and source identities must be verified;
   all numerical trace failures and near ties remain visible. A scoring error,
   missing row, unsupported tie or unverified receipt is a failure, not a pass.

When the paired hypothesis strings are identical, each deterministic paired
bootstrap difference is provably zero; the scorer may record an explicit
identity shortcut instead of recomputing the same metric 2,000 times. For any
nonidentical hypotheses it computes all 1,000 paired resamples normally.

The -0.5-point quality margin is roughly one ninth of the 4.28–4.79-point
widths of the 128-row baseline BLEU/chrF bootstrap intervals across base and
CFG; 0.01 length ratio is below one fifth of their 0.055–0.062 widths. These
strict paired-comparison limits are intentionally much smaller than baseline
sample uncertainty and are supplemented by the exact deterministic raw-ID gate.
All 128 development rows matched exactly in both modes, making a zero paired
difference the observed development result rather than a justification to
relax a future mismatch. These provisional criteria are for this pinned FP32
comparison; changing them after observing held-out data requires explicit
disclosure and approval.

Passing this protocol alone does not establish performance, memory, tokenizer
CLI coverage, Metal, or Phase 4 completion. The separate XSum Metal raw-ID
failure remains unresolved.
