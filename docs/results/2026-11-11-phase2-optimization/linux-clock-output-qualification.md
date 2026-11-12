# Clock codec report qualification

**Status: offline report-content component. Linux 72/78; Apple pending.**
`tools/bench/qualify_clock_codec.py::qualify` checks all eighteen supplied full codec
reports against the existing frozen v2 Python reference. It runs no workload, imports
no Torch and does not evaluate performance acceptance or authorize clock collection.

## Contract and coverage

- Exactly eighteen raw reports and complete receipt declarations in the approved
  previous/Python/candidate cyclic order: six per runtime, six repetitions, distinct
  supplied PIDs/report paths. No prefix, copied Python report or selectively omitted
  repetition qualifies. Report size/SHA-256 must match the supplied post-exit receipt.
  Bytes are bounded at the existing 16 MiB/report; JSON rejects duplicate keys and
  nonfinite constants. Raw capture and current-file identity checks remain separate.
- Native source inventories must match separately supplied reviewed declarations,
  with the unchanged complete native-source set and pinned harness layout. Reports
  retain CUDA/FP32, u32 native/i64 Python indices, five warmups/thirty trials and
  **no diagnostic profiling**. Explicit numeric checks reject Boolean timing/logit/
  count substitutions and nonpositive timings; no numerical tolerance changes.
- Current Python runtime/build/upstream/source/version/manifest/workload/protocol
  metadata must exactly match the declared historical metadata **and** all three
  frozen reference reports. This preserves distribution versus runtime-build
  distinctions; metadata agreement does not establish historical interpreter/library
  bytes or a fresh upstream worktree check.
- Six canonically distinct reports per runtime are required across both halves, so
  whitespace or putting a copied report in the other triple does not make it fresh.
  Distinct reports/PIDs alone still cannot prove independent real processes.
- The existing `phase2_summary.codec` checker accepts triples. The new wrapper uses
  fixed repeats **1–3 and 4–6**, never chooses a favorable triple, and checks both
  native runtimes for each half. All current Python reports also go **directly**
  against the same frozen Python reference, not transitively through Rust or a new
  Python report. A third `current_python` group uses Python allocator fields; it does
  not reinterpret those values as Rust pool counters. Four checked groups result.
- The existing archive/member binding and reference selection are unchanged: one
  copy each of frozen Python repeats 1–3. The existing codec checks cover B1/B2/B4,
  all twelve scopes, IDs and retained counts, first decode/reconstruct logits,
  stage/load/trial structure, fixture and artifact-manifest identity. FP32 tolerance
  remains **5e-4 + 2e-4 × abs(reference)**. No golden or oracle is regenerated.

Only `reported_codec_outputs_qualified` is true. Returned evidence includes each raw
report/receipt/canonical-report digest, all four quality groups and the frozen
reference binding. No latency/throughput ratios or acceptance verdict are published.
The shared checker computes descriptive statistics internally, but this wrapper
returns only quality fields. It reads the existing reference archive, fixture and
manifest **after collection**; that is not the final all-frozen-file rehash.

This does not prove capture-file identity, producer authenticity, process freshness,
whole-job lifecycle, clock enforcement/restoration, runtime-library equivalence,
unreported block/snapshot floating outputs or operator/642-trace parity. A future
finalizer must bind the same reports/receipts to the process/lifecycle/content/file
gates before claiming a complete diagnostic. CPU reporting tests close no Linux
acceptance comparison and provide no Apple evidence.

## Tests and limitations

Three new tests include an archive-based suite in a **fresh subprocess** to preserve
the parent suite's RSS-sensitive tests. It verifies the retained v2 archive SHA/size
and all eighteen selected member hashes, using six original `python/` reports—not
the twelve additional copied Python files. It combines those historical report bytes
with explicitly **synthetic current-style receipts**. This is offline validator
coverage, not a new diagnostic, current producer proof or salvage of failed read-only
v1. No historical failure or postflight status is changed.

The archive test checks all four twelve-scope quality groups and rejects **43 altered
cases**: every runtime's sixth report has ID/logit/load/stage/trial/type/schema/profiling
mutations; cross-half copied reports; order/PID/path/size/hash/exit/prefix failures;
Python build/upstream/distribution/source changes; forged matching Python declarations;
a direct-reference tolerance-chain case; duplicate JSON keys and nonfinite constants.
Mutated outputs are reserialized and rehashed where appropriate, so matching hashes
cannot hide bad logits or structure. Cached golden loading is used only for repeated
negative cases; the real archive loader and unchanged numerical checker run for the
positive baseline. Additional small tests reject missing inputs or unreviewed source/
metadata declarations before reference reads.

The initial focused run failed because its last-job order mutation assigned
`python`, which was already that slot's runtime. It was a no-op fixture, not an output
validator bypass. Changing the test mutation to `candidate` makes it exercise the
order guard; no production guard/tolerance was weakened. The initial log is retained.
Final review also made cross-half copied reports use demonstrably different raw JSON
whitespace before rehashing, so rejection exercises canonical equality rather than
only identical bytes. Focused/default checks pass again after that fixture enhancement.

```sh
PYTHONPATH=tools/bench python -m unittest test_qualify_clock_codec test_qualify_diagnostic_codec -v
CARGO_BUILD_JOBS=4 ./tools/check.sh
```

Final **7 focused and 373 reporting tests pass**. No new model/GPU/benchmark/policy
run or current library scan occurred. The earlier unresolved lifecycle-fixture
failure remains recorded, not declared repaired by these passes.

| Private evidence under `reports/` | SHA-256 |
| --- | --- |
| `phase2-clock-codec-qualification-tests-v1.log` (fixture failure) | `c2f09e2d29be5097a86409da3b2e00f37d7ef40f103bdd35b67e203316f8f4a9` |
| `phase2-clock-codec-qualification-tests-v2.log` (7 pass) | `404922d244bee74961f450e6c1e365a108bd6559518f9feac9ea389eaee23072` |
| `phase2-clock-codec-qualification-check-v1.log` (373 pass) | `2ca87935a7c93074af3e7bcdc4adfda848001af457c858133e45e1d4a59197b2` |
| `phase2-clock-codec-qualification-tests-v3.log` (final 7 pass) | `a9c9cf9f203314de42b84cd52208efeb4ed5355decc6e152da219444dfcfd51a` |
| `phase2-clock-codec-qualification-check-v2.log` (final 373 pass) | `907384d9b33ea23ffcd4bfab844f69654c4e68b1d730aa1d2e8ecf48a92b1c62` |

## Next boundary

Raw-capture/report-file identity and role-command binding, then same-evidence
finalization composing telemetry/lifecycle/process/output validators and frozen
rehash **last**. Fresh upstream verification, complete preparation and frozen entry
remain unfinished. Default-policy/no-custom-lock/no-controller confirmation,
permissions/enforcement and a later new quiet window remain unresolved. No live
launch gate, performance comparison or Phase 2 completion is claimed.
