# WMT14 CFG release-surface audit — decision pending

2026-11-20. The Phase 5 plan says unqualified options must be available only through explicitly
named experimental runtime flags. The unified WMT14 command currently accepts any finite
nonnegative `--cfg`, while the legacy `sample-wmt14` alias accepts a positional scale. Neither
path requires an experimental flag for positive values, and help does not identify which WMT14
scales are qualified. The backend additionally rejects scales that overflow FP32. This audit does
not change the CLI or infer a new exception.

For example, candidate binary SHA-256
`6194111fe3860584492676c6d14f50124a77f8dcf18f316a959b8abd16a20278` and the qualified WMT14
manifest `01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81` were invoked with
`--cfg 2` on the unified command and positional `2` on the alias, using a deliberately absent input.
Both passed option routing, reached the missing-input error and returned status 1 with empty stdout.
This is a CPU-side parser/loading check, not successful inference or quality evidence. Raw outputs
remain ignored under `reports/phase5-wmt14-cfg-policy-audit-v1/`.

The frozen WMT14 CUDA and Metal work includes full-corpus top-1 base and CFG=1 candidate retests
([CUDA](../2026-11-16-phase4-wmt14/cuda-fast-argmax-plan-v1.md#source-bound-corpus-and-repeated-timing),
[Metal](../2026-11-16-phase4-wmt14/mlx-corpus-retry-v1.md#cli-and-profile-qualification)); the
earlier CUDA CFG=1 exact-ID failure and later remediation remain recorded. The Phase 5 CUDA retest
so far covers base CFG=0 only ([report](cuda-wmt14-quality-retest-v1.md)). The Metal CLI report
mentions a scale-2 byte-identity control, not full quality qualification for all scales. The README
and draft release notes say WMT14 top-1 but do not define a positive-CFG qualification boundary.

A decision is needed before changing compatibility or declaring the experimental-option gate met:

1. Preserve scales 0 and 1 as qualified syntax and require an explicit experimental WMT14 flag for
   every other positive scale, including on the legacy alias; or
2. Treat **all** positive WMT14 CFG as experimental for this release and require a named flag on
   both commands, including scale 1.

Either choice needs clear help/migration documentation, parser tests and CUDA validation before
Metal validation. The exact frozen quality evidence, thresholds and previous failures must remain
unchanged. No performance or release gate is established by this audit.
