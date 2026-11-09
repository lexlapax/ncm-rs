# Agent context map

Last updated: 2026-11-09. Read this first; load linked documents only for your task.

## Current state / approval gate

- Phase 0 research/scaffold prepared; repository began as uncommitted Cargo hello-world.
- Repository created: https://github.com/lexlapax/ncm-rs (**public**, as requested), branch `main`,
  remote `origin`. Initial scaffold is the baseline for this context map.
- **No model, backend, converter, numerical tests or benchmark implemented.**
- User requested phased port planning, initial structure and commit/push to
  `lexlapax/ncm-rs`, then plan review **before implementation**. Phase 1 not approved.
- Runtime proposal: shared Rust model + `mlx-rs` on Apple Silicon and Candle CUDA
  on Linux; Python only for offline reference/conversion/evaluation.
- Additional requirements: periodic upstream parity; published performance AND
  quality comparisons on completion and relevant changes; lazy agent context;
  context-map update before every commit; no agent attribution.
- User explicitly requested public visibility; GitHub confirms `PUBLIC`.
  Future reviewed quality/performance reports will be publicly published.
  No force-push/history rewrite authorized.

## Task routing

| Need | Load |
| --- | --- |
| Goals, crate layout, scaffold commands | [README.md](README.md), [Cargo.toml](Cargo.toml) |
| Phases, acceptance gates, next implementation | [docs/PLAN.md](docs/PLAN.md) |
| NCM semantics, release differences, source evidence/bugs | [docs/RESEARCH.md](docs/RESEARCH.md) |
| Backend contract, crate boundaries, optimization design | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Fixtures, numerical/quality tests, CI/hardware safety | [docs/TESTING.md](docs/TESTING.md) |
| Conversion, config/weight formats, trust boundaries | [docs/ARTIFACTS.md](docs/ARTIFACTS.md) |
| Upstream tracking, compatibility and upgrade policy | [docs/UPSTREAM.md](docs/UPSTREAM.md), [upstream.toml](upstream.toml) |
| Benchmarks, regressions, publication | [docs/REPORTING.md](docs/REPORTING.md), [docs/results/README.md](docs/results/README.md) |
| Contribution rules / commit checklist | [AGENTS.md](AGENTS.md), [.github/pull_request_template.md](.github/pull_request_template.md) |
| Scaffold CI / metadata-only weekly watch | [.github/workflows/ci.yml](.github/workflows/ci.yml), [.github/workflows/upstream-watch.yml](.github/workflows/upstream-watch.yml) |

## Pins and blockers

- NCM research pin: `c70274273f489abc73832b2b98584a8483b6b203`.
- HF weights inventory: `d342ce8aaea58830eef3b8ad473664673e41f692`.
  These pins are **researched, not validated Rust compatibility**.
- OWT weights absent at inspected revision (`owt/TODO.md` only); paper/config
  differences exist. Upstream has no full training pipeline.
- Linux RTX 3060 Ti 8 GiB and CUDA compiler available; no Apple Silicon execution
  available. Neither GPU runtime built/tested. Weights not downloaded.

## Work/checks/next actions

- Added seven-package dependency-free workspace, status-only `ncm` binary, research/
  architecture/phases/artifact/test/upstream/reporting documents, MIT notices,
  PR checklist, scaffold CI and weekly metadata watcher. No inference code.
- Passed locally on Rust 1.99.1: `cargo fmt --all -- --check`;
  `cargo clippy --workspace --all-targets --locked -- -D warnings`;
  `cargo test --workspace --locked` (**zero tests**, scaffold compilation only);
  `RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --locked`;
  `cargo run -p ncm-cli --locked`; `git diff --check`.
- TOML parsing/local Markdown links passed; watcher inline Python parsed and ran
  metadata-only against GitHub/HF successfully, without creating an issue.
- Rust 1.88/stable CI configured; no local MSRV, GPU, model-quality or performance
  validation. Confirm remote CI after pushing; no completed remote result claimed here.
- Initial commit/push authorized. After push, report repository/commit, summarize
  phases and await approval. Any next commit must refresh this map again.
- After approval: Phase 1 reference/export contract and BOTH hardware spikes;
  do not jump directly to kernels or claim default CPU CI is GPU validation.
