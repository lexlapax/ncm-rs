# Current candidate and archived gate bindings

The [binding checker](phase2-closure-bindings-v1.py) verifies the following from
Git history and the retained archives, without running a benchmark:

- Current tracked production crates, vendored dependencies, lockfile, upstream
  pin, reference tools, fixtures and the listed acceptance harness files match
  gated candidate `5200924`. The same tracked paths have no staged or unstaged
  changes. Untracked files are not covered by this source comparison.
- The selected upstream pin, tracked fixtures, CUDA oracle report and artifact
  manifest match accepted baseline `3307dff`. The local qualified Sudoku
  manifest matches the committed anchor, SHA-256
  `ad6a99ad7d5ccc069e0fedf26a85f755e06cd76590463e1e04e17363e35d9832`.
- Both raw archive hashes and the embedded controlled-audit hashes match their
  recorded values. Each audit has 78 passing cells, passing quality, the same
  audit-source hash as the checkout, and unchanged 1.05 median/host-memory and
  1.10 p95 limits. Linux audit SHA-256 is
  `50960a48e6e026b377b547fcc3df39096a5408ed27c576114d3d79a9b52500e3`;
  Apple is `a3ed81700b05f39a50a29a44340ffb00fb826fa738539989eb41e78556d58e4e`.

Run from this checkout with `python3 docs/results/2026-11-11-phase2-optimization/phase2-closure-bindings-v1.py`.
The checks rely on the existing Git history and archives; they do not re-run
the underlying audits, verify untracked executable bytes, qualify hardware, or
establish the separate comparable steady-state live-memory criterion. The
[Apple result](apple-gate18-results-v1.md) preserves the immediate snapshot and
its unsettled-versus-retained ambiguity. Phase 2 remains open, and the named
final platform acceptance reports have not been published.
