# Phase 5 step 1 interim audit — not a completion audit

2026-11-19. This is a checkpoint against [step 1](../../PHASE5.md#step-1--release-candidate-v010-no-numerical-changes),
not a claim that Phase 5, either runtime gate, or the release is complete. It identifies a new
alias-policy ambiguity and summarizes remaining work. Later steps and the required final
requirement-by-requirement audit remain open.

| Requirement | Current evidence | Open work / status |
| --- | --- | --- |
| Unified CLI, aliases, ordinary GPU features | [`sample.rs`](../../../crates/ncm-cli/src/sample.rs), [feature graph](../../../crates/ncm-cli/Cargo.toml), [CUDA fixed-request routing](cuda-routing-screen-v2.md) | One fixed batch per profile on CUDA; Metal not screened. Current XSum alias retains an unflagged experimental path, below. |
| Explicit experimental flags, unsupported combinations and OWT | [help](../../../crates/ncm-cli/src/help.rs), [unified parser](../../../crates/ncm-cli/src/sample.rs), [rejection tests](cuda-routing-screen-v2.md#cli-rejection-integration-checks) | Unified XSum requires flags; its legacy alias does not. Ten negative cases are not a complete device/option matrix. |
| Decode and text | [CUDA decoding](cuda-decode-screen-v1.md), [corpus encoding and boundaries](cuda-text-input-v1.md), [overlength generation](cuda-text-truncation-v1.md) | Decoder screen is one retained output row per text profile; XSum/WMT14 encoding covers all required corpus rows, with limited end-to-end CUDA examples. LM1B `--text` decision and Metal evidence remain open. |
| `docs/USAGE.md` build, fetch, exports, hashes, examples, fresh clones | [integrated Linux replay](linux-guide-integrated-replay-v1.md), [export investigation](linux-export-repro-v1.md) | Seven profiles were reproduced in one fresh Linux clone, but Sudoku medium/hard qualified manifests were not. The complete Linux and Mac guide gates remain open. |
| README, release notes, limitations | [README](../../../README.md), [draft notes](../../RELEASE-NOTES-v0.1.0.md), [Phase 4 audit](../2026-11-19-phase4-audit/phase4-final-audit-v1.md) | Matrix and recorded limitations are drafted; update to reflect final decisions and qualification. Historical results do not automatically qualify this candidate. |
| Reproducible performance/quality | [reporting policy](../../REPORTING.md), [CUDA Sudoku cold screen](cuda-sudoku-cold-screen-v1.md) | A single-profile exploratory cold-process screen is not the required full comparison or quality gate. Missing workload/device evidence is pending. |
| `v0.1.0` tag and source-only GitHub release | Local `git tag --list 'v0.*'` returned no tags on this checkout. | User approval is required before tagging/publishing. GitHub release status was **not** established: `gh release view v0.1.0 --json tagName` timed out on TLS handshake. |

## XSum alias and experimental-flag ambiguity

The unified `sample` parser requires `--experimental-positive-cfg-xsum` for positive CFG and
`--experimental-stochastic-xsum` with a seed. However, `main_impl` accepts the original
`sample-xsum ARTIFACT DEVICE INPUT [CFG [SEED]]` positional form and forwards CFG/seed to the
backend without either flag. The help explicitly promises original alias arguments. This conflicts
with step 1's requirement that unqualified options remain available **only** as explicitly named
experimental runtime flags. The intended compatibility boundary needs a user decision; no exception
or behavior change is inferred here.

A no-inference check used the qualified XSum artifact and candidate binary SHA-256
`cbd3d8d1fa39cbd878c6ff5f41a4f32aafe726d7706ee7c0dc28e0fd4e3a9b67` with a deliberately absent
input. Legacy alias calls with positional `1`, and `0 7`, proceeded to the missing-input error;
the unified calls using `--cfg 1` or `--seed 7` rejected the missing experimental flag before
input loading. All four returned status 1 with empty stdout. This demonstrates parser routing, not
successful stochastic generation or quality. The candidate's dispatch, parser, help and feature
sources match the current tree; later changes to `decode.rs` were test-only. Captured stderr/stdout
remain ignored under `reports/phase5-xsum-alias-flag-audit-v1/`.

Possible resolutions: require explicit experimental flags even on the alias, rejecting the old
unflagged experimental positional calls while retaining qualified alias behavior; or explicitly
approve a narrowly documented exception for the legacy positional forms. The user's existing
requirements do not authorize silently choosing the exception.
