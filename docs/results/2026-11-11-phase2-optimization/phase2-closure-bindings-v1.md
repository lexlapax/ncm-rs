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
- Both raw archive hashes, embedded controlled-audit hashes and the published
  [Linux](linux-acceptance.json) and [Apple](mlx-acceptance.json) audit files
  match their recorded values. Each audit has 78 passing cells, passing quality, the same
  audit-source hash as the checkout, and unchanged 1.05 median/host-memory and
  1.10 p95 limits. Linux audit SHA-256 is
  `50960a48e6e026b377b547fcc3df39096a5408ed27c576114d3d79a9b52500e3`;
  Apple is `a3ed81700b05f39a50a29a44340ffb00fb826fa738539989eb41e78556d58e4e`.
- The exact cell inventory is independently checked: 58 timing comparisons
  cover 12 codec scopes, 12 tokenizer scopes and five checked loads against
  both references; 20 host-memory comparisons cover RSS and HWM for all five
  workloads. Every stored controlled median/p95 and upper confidence bound is
  checked against its original limit, as is each host-memory ratio. Controls
  contain nine Linux and 18 Apple processes per runtime and workload. Three
  [focused negative tests](phase2-closure-bindings-test-v1.py) also reject
  duplicate/missing cells, wrong control counts, nonfinite values and altered
  timing or memory bounds, trace inventory corruption, oracle changes and
  decision mismatches.
- The qualification bytes match the exact inputs recorded in each audit. Both
  contain 642 uniquely named traces with zero element mismatches and 2,500
  selection decisions with no disagreements; both disclose two near ties. The
  pinned oracle hash and unchanged numerical policy are checked explicitly.

Run from this checkout with `python3 docs/results/2026-11-11-phase2-optimization/phase2-closure-bindings-v1.py`.
The checks rely on the existing Git history and archives; they do not re-run
the underlying audits, verify untracked executable bytes, qualify hardware, or
establish the separate comparable steady-state live-memory criterion. The
[Apple result](apple-gate18-results-v1.md) preserves the immediate snapshot and
its unsettled-versus-retained ambiguity. The named platform files are exact
copies of the 78-cell audits; they do not include the separate live-memory criterion. Phase 2 remains open.

A fresh clean export of `5200924` with the retained Linux archive reproduces the
controlled audit byte-for-byte. The prior Apple clean export reproduces the
initial failure byte-for-byte and the controlled decision with nine differences
in the last floating-point digit of standard deviations. The [replay checker](phase2-replay-review-v1.py)
rejects other differences; these are offline recomputations, not new GPU
measurements. The [Apple result](apple-gate18-results-v1.md) records the
cross-environment limitation.

For a fresh Linux/Apple audit replay from this repository root:

```sh
phase2_tmp=$(mktemp -d)
mkdir "$phase2_tmp/linux" "$phase2_tmp/apple"
git archive 5200924 | tar -xf - -C "$phase2_tmp/linux"
git archive e69828a | tar -xf - -C "$phase2_tmp/apple"
tar -xJf docs/results/2026-11-11-phase2-optimization/raw/tiny-borrow-linux-gate-v1.tar.xz -C "$phase2_tmp/linux"
tar -xJf docs/results/2026-11-11-phase2-optimization/raw/apple-gate18-resource-memory-v1.tar.xz -C "$phase2_tmp/apple"
(cd "$phase2_tmp/linux" && python3 tools/bench/audit_phase2_performance.py reports/phase2-tiny-borrow-linux-controlled-config-v1.json "$phase2_tmp/linux.json")
(cd "$phase2_tmp/apple" && python3 tools/bench/audit_phase2_performance.py docs/results/2026-11-11-phase2-optimization/apple-gate18-initial-config-v1.json "$phase2_tmp/apple-initial.json" || test "$?" -eq 1)
(cd "$phase2_tmp/apple" && python3 tools/bench/audit_phase2_performance.py docs/results/2026-11-11-phase2-optimization/apple-gate18-controlled-config-v1.json "$phase2_tmp/apple-controlled.json")
python3 docs/results/2026-11-11-phase2-optimization/phase2-replay-review-v1.py "$phase2_tmp/linux.json" "$phase2_tmp/apple-initial.json" "$phase2_tmp/apple-controlled.json"
```

The initial Apple audit is expected to return 1; the review script confirms its
actual retained contents. Results may differ further across Python environments;
the checker fails closed for changes outside tiny standard-deviation rounding.
