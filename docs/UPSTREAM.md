# Keeping parity with Python NCM

Ongoing upstream compatibility is a core goal, not a one-time port. The desired
state is a Rust release qualified against an explicit Python NCM commit and
checkpoint revision, with a visible lag behind newly released upstream behavior.
“Latest-compatible” without a pin and evidence is not an acceptable claim.

## Current state

[../upstream.toml](../upstream.toml) holds **research pins only**. There is currently
no validated Rust implementation. The [research inventory](RESEARCH.md) records
release profiles and blockers. Phase 1 adds an executable per-profile compatibility
matrix; until then all profiles are planned or artifact-blocked.

Use separate version identities:

1. Rust package/release version.
2. Supported upstream code commit and HF artifact/tokenizer/data revisions.
3. Canonical artifact schema and reference-fixture schema versions.
4. Candidate upstream update, which is not supported until qualification completes.

## Cadence and detection

- Weekly GitHub Actions watcher compares upstream `main` HEAD and HF checkpoint
  revision with the research/supported pins. It writes a job summary and opens or
  refreshes a single tracking issue if either changes. It fetches metadata only:
  **no upstream code execution, model downloads or automatic pin updates**.
- Monthly human triage, and again before every release. Prioritize correctness,
  checkpoint/config changes, new model profiles, tokenizer changes and training
  code releases. Also monitor upstream issues/releases; the initial watcher checks
  revisions, not all issue discussions or every dataset/tokenizer repository.
- Phase 1 expands the manifest/watcher to dataset and tokenizer revisions, file
  inventories/checksums and dependency/runtime releases. Metadata failure is a
  failed watch run, not “no changes”. Scheduled workflows need Actions enabled and
  a maintained repository; notifications and a human owner remain necessary.

## Update playbook

1. Create a candidate branch/issue; diff old..new code and separately diff checkpoint,
   configuration, inference metadata and tokenizer assets. Record files and profile
   effects, including changes to defaults, masks, math, sampling and evaluation.
2. Classify: docs-only; behavior-preserving; correctness fix; new required operator;
   checkpoint/schema migration; evaluation-protocol change; training-only. Publish
   an impact checklist and explain whether historical outputs should change.
3. Keep the old reference environment and fixtures. Build a separately pinned
   candidate Python environment; reproduce the relevant change with minimal tests.
4. Regenerate **candidate** fixtures and canonical artifacts. Review tensor/config
   diffs and hashes. Never overwrite the sole golden reference before comparison.
5. Port the change in shared Rust logic and both adapters. Keep legacy profile
   support where feasible; otherwise document the breaking boundary and migration.
6. Run old-profile regression, new numerical/model parity, then affected quality
   and performance suites on CUDA and Apple Silicon. Passing local CPU checks
   cannot qualify a GPU change. Compare old Python vs new Python to distinguish intentional
   upstream behavior changes from a Rust regression.
7. Publish the four-way comparison where applicable: old/new upstream Python and
   old/new Rust. Include fixed artifact comparisons and new-artifact comparisons
   separately. See [REPORTING.md](REPORTING.md).
8. Only after review, advance supported pins, fixture/schema versions and the
   compatibility matrix, close the watch issue, and tag the Rust release. Retain
   immutable historical reports. An unsupported new feature stays explicitly pending.

## Compatibility matrix fields (Phase 1 implementation)

For each profile and runtime track: upstream/config/checkpoint hashes, tokenizer
revision, implemented features, numerical status, quality status, benchmark report,
known divergences, hardware/dtype, and last qualified Rust commit. Statuses are
`planned`, `artifact-blocked`, `implemented-unvalidated`, `validated`, or `regressed`.
No global “supported” badge if one required runtime lacks evidence.

## Policy on upstream bugs and API changes

Match intended/released behavior, not unsafe deserialization or accidental aliasing.
Maintain minimal documented reference patches and dedicated regression fixtures.
For numerical changes or upstream bug fixes, explain and version any resulting
output difference rather than silently rewriting historical baselines. Submit clear
repros/fixes upstream when useful; external contributions are separate from this
repository's automated watcher and require review.

Avoid long-lived forks of runtime crates. Prefer upstream contributions for missing
ops; pin reviewed patches narrowly with an exit condition. Model compatibility,
runtime dependencies and compiler upgrades each get independent qualification.
