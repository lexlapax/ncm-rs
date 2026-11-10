# Agent rules

## Start here; load context lazily

1. Read [agent-context-map.md](agent-context-map.md) before planning or editing.
2. Inspect `git status` and the files relevant to the task. Preserve unrelated work.
3. Follow the map's links only as needed; do not load every design document by default.
4. If instructions conflict or the map is stale, inspect the actual source and ask
   about consequential ambiguities. Never treat planned features as implemented.

## Project constraints

- Implement fast, idiomatic Rust NCM with MLX on Apple Silicon and CUDA on Linux.
  Keep production inference Python-free; native MLX/CUDA libraries are expected.
- Maintain periodic parity with explicitly pinned upstream Python NCM. No silent
  upstream pin, artifact, schema, oracle, or numerical-tolerance changes.
- Establish correctness before optimizing; preserve shared model semantics across
  adapters. Unsupported features must fail explicitly, not silently fall back.
- Publish reproducible quality AND performance comparisons for execution/output-
  affecting changes and releases. Missing hardware evidence is pending, not passing.
- The implementation plan is approved. Within EVERY phase, complete Linux/CUDA
  work and validation first, then macOS/MLX. Both runtime gates must pass before
  advancing phases. Consult the context map for the current phase and blockers.
- `ssh macstudio` is authorized for Apple testing. It has competing workloads:
  **notify the user and coordinate availability before starting MLX workloads**.
  Inspect remote state before syncing; never overwrite unrelated work or change
  global tooling unnecessarily. Pull reviewed pushed commits into
  `~/projects/lexlapax/ncm-rs` rather than overwriting that checkout.
- Keep unsafe code narrowly audited, dependencies deliberate, secrets/weights/data
  out of Git, and untrusted code off persistent GPU runners. Respect licenses.
- No agent attribution in code, comments, documentation, generated files or commit
  messages: no “by Claude/Codex/pi”, AI-author credits, or agent co-author trailers.
  Preserve required upstream copyright/license notices and technical provenance;
  these are not agent attribution.

## Work and validation

Read the task-specific contract through the map, make focused changes, add regression
tests, and run relevant checks. Test MLX and CUDA separately on real target hardware;
CPU builds do not prove GPU correctness. Record exact commands/results and limitations.
Do not fabricate benchmarks, claim empty tests establish parity, or regenerate goldens
merely to hide a failure. Distinguish upstream behavior from documented reference fixes.

## Before EVERY commit (including amend/fixup)

1. Update **agent-context-map.md in the same commit** with current state, changed
   areas, checks actually run, blockers, decisions and next actions. Keep it concise;
   link durable details instead of copying them. Update it even for docs-only work.
2. Run the appropriate checks and inspect the staged diff for scope, secrets,
   accidental artifacts and misleading status claims.
3. Confirm the staged context-map update is present. Never commit without it.
4. Commit/push only within the user's authorization; avoid force-push or rewriting
   shared history. Report what was changed, tested, published and left pending.
