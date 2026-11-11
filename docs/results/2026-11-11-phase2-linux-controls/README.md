# Expanded Phase 2 Linux controls and resource diagnostics

**Linux acceptance remains incomplete: 72/78 controlled comparisons pass.**
No Apple work or Phase 3 advancement is authorized by these results. Production
behavior remains the provisional `d44beaa` candidate; this checkpoint changes
reporting/diagnostic tools, not model/tokenizer execution or the public loader API.

## Fixed matched controls — all results retained

`reports/phase2-optimization-controls-plan-v1.json` was written and hashed before
collection. A fresh three-process full matrix compares accepted Rust `3307dff`
(production `083c1e5`), candidate `d44beaa`, and unchanged qualified Python.
Then exactly **30 fresh processes per runtime** were collected for codec, BERT,
GPT-2 and Qwen: **360 processes**, rotating runtime order, CPU affinity 2, five
warmups and thirty trials each. All execution scopes and checked loads of those
four workloads were predeclared for controls, not selected afterward. Python
samples are shared between comparisons, not independent duplicates.

The initial matrix passes output checks but its unchanged audit is **74/78**.
The controlled audit is **72/78**, under the same 1.05 median/memory and 1.10 p95
bands. Six comparisons against accepted Rust remain open:

| Scope | Controlled median ratio | Controlled p95 ratio | Failing upper bound / condition |
| --- | ---: | ---: | --- |
| B1 resident decoder block | 1.01174 | 1.01323 | Median interval upper **1.10818** |
| GPT-2 encode | 0.99107 | 1.09738 | p95 interval upper **1.50205** |
| GPT-2 decode | 1.00321 | 1.08133 | p95 interval upper **1.73052** |
| GPT-2 skip-special decode | 0.98715 | **1.13875** | Point and interval p95 (upper **1.68636**) |
| Codec checked load | 0.98981 | 0.96715 | p95 interval upper **1.15753** |
| Qwen checked load | 1.01787 | **1.10334** | Point and interval p95 (upper **1.17563**) |

Every comparison against qualified Python and every original-matrix comparable
host-memory cell passes. That is not a waiver for failures against accepted Rust.
The fixed bootstrap remains 10,000 paired whole-process resamples, seed 20261110,
descriptive per-cell 95% intervals, not simultaneous guarantees. More processes
did not automatically resolve uncertainty: they also exposed tail variation in
previously passing point estimates. Nothing was discarded or relabeled passing.

B1 block process medians cluster near 0.177 and 0.198 ms. CUDA event spans show
corresponding clusters too; this is not solely the host stopwatch overhead.
GPU telemetry records changing clocks, but neither those samples nor inspection
of the unchanged block establish a causal explanation. Background CPU work was
observed after collection, not stopped; it cannot retrospectively prove the cause
of a particular tail. No clocks, power settings, services or unrelated processes
were changed. A quiet/stabilized matched test environment is the next resource
question, not justification to widen bands or repeatedly sample until passing.

### Explicit reporting-tool extension

Historical `summarize_codec_load_control.summarize(root, device)` behavior remains
load-only. `include_execution=True` additionally validates and summarizes all
12 execution cells, retaining every declared process. The audit opts in with
`codec_execution_controls: true`; it requires an actual selected matrix
`environment.json`, and binds native binary hashes, native/Python source hashes,
Python environments, runner identity and affinity. Apple native libraries are
also bound when applicable. This changes reporting configuration only, not
model/artifact/workload schemas or numerical tolerances.

Regression tests verify all 13 codec scopes, no lost trials, both native binaries
and Python bindings, and that failing execution intervals propagate into the
unchanged **78-cell** audit. Historical load-only outputs remain reproducible.

## Whole-model lifetime and retained-output memory

New `crates/ncm-backend-cuda/examples/codec_memory.rs` is a CUDA-only diagnostic.
Three fresh processes per native implementation each run ten complete authenticated
model load/execute/drop cycles. The first model executes thirty B1/B2/B4 rounds;
subsequent models execute two rounds: **144 requests per process**. Observations
are synchronized, both with outputs retained and after outputs are dropped.
An independent backend clone stays alive across each model's destruction.

All six captures pass: live CUDA pool usage stabilizes immediately after prepared
state is created and returns to **zero after all 60 model drops**. Results:

| State | Accepted Rust live pool bytes | Candidate live pool bytes |
| --- | ---: | ---: |
| Loaded weights | 59,586,048 | 59,586,048 |
| Prepared model, outputs dropped | 59,586,048 | 59,611,728 |
| B1 outputs retained | 61,779,228 | 60,711,276 |
| B2 outputs retained | 63,972,408 | 61,810,824 |
| B4 outputs retained | 68,358,768 | 64,009,920 |

Prepared state adds exactly **25,680 logical pool-used bytes**. Immutable snapshot
sharing saves a net **1,067,952 / 2,161,584 / 4,348,848 bytes** with B1/B2/B4 outputs
retained. This diagnostic retains native index tensors, unlike the acceptance
harness's host ID vectors; its absolute counters are not substituted into that
matrix. Reserved pool/context/native-library memory and host RSS have different
coverage. Observed whole-process high-water marks are about 772.85/772.83 MB here:
there is **no material host-RSS saving claim**. Reserved pool memory need not fall
to zero after drop.

## CPU, lifecycle, and actual metadata allocation work

Fresh nine-process lifecycle runs validate every context/load/execute/drop phase
and identical output identities. Separately, 45 whole-process resource jobs
(three native/native/Python processes for each workload) complete; all their raw
benchmark outputs also pass the codec/tokenizer validators.

Median total child CPU milliseconds, including imports/context, loading, all
benchmark trials, serialization and exit — **not request CPU**:

| Workload | Accepted Rust | Candidate | Python |
| --- | ---: | ---: | ---: |
| Codec | 9,414.775 | 9,476.988 | 15,919.650 |
| BERT | 281.561 | 282.550 | 1,588.207 |
| GPT-2 | 900.331 | 911.964 | 3,886.105 |
| Qwen | 4,550.973 | 4,441.535 | 9,949.394 |
| TinyStories | 214.361 | 212.085 | 3,184.728 |

These small native differences do not establish a broad CPU improvement. Resource
collection was serial with timing collection; the report auditor ran separately
on CPU 3 during part of the resource-only collection. No acceptance timing was
collected under the debugger or concurrently with the new GPU diagnostics.

`tools/bench/gdb_codec_metadata.py` inspects the **unchanged** native profile
executable using local GDB breakpoints, not production instrumentation. Three
captures agree: two B1 reconstructions still execute **336 direct model formatter
calls**, with **336 malloc and 236 realloc entry calls / 15,984 requested bytes**.
Thus `NoTrace` does not eliminate all dynamic labels/weight names in this binary.
Per reconstruction that is 168 format calls and 7,992 requested bytes, not live
storage. Warm encode records 7,555 allocation and 729 realloc entries in total;
warm resident block records 475 and 56. These latter counts include native library
work and are not Rust-only GlobalAlloc counters or timing attribution.

An initial Rust-shim-only attempt missed inlined allocator paths and observed no
formatter allocations. It is retained as an instrumentation-coverage finding,
not evidence that formatting is allocation-free. Subsequent libc-entry captures,
per-invocation counts, exact executable/symbol/source hashes, ABI disassembly and
four offline bookkeeping/error tests are retained. The tool is explicitly Linux
x86-64 and rejects an incompatible executable/ABI; it launches an owned process,
never attaches to unrelated work. Nested allocator entries and unobserved custom
allocators prevent equating these counts with unique live objects.

**Disposition:** defer additional typed weight records/trace-plumbing changes in
this bounded iteration. The actual transient work is now measured, but no isolated
latency gain has been demonstrated; adding state/API plumbing while the hardware
performance gate remains unsettled is not justified. No zero-cost claim is made.

### Conditional CUDA/ownership decisions for this iteration

- **Defer custom selection/resampling kernels, alternate convolution and permanent
  weight-layout copies.** The unchanged candidate's retained CUPTI profile has
  892 kernels, including 94 `ucopy_f32`, 86 indexed gathers and 26 im2col calls;
  26 value reductions, equality masks, masked-index selections and index minima
  preserve first-index ties. These are residual opportunities, not evidence of
  a particular profitable replacement or a license to alter reduction order.
  No per-weight copy attribution justifies keeping duplicate weight layouts;
  composed activity sums do not predict request speedup. CUDA correctness and
  memory savings are established without adding another kernel/dependency surface.
- **No-go for CUDA graph replay in this bounded iteration.** The two-request
  profile still contains 1,536 allocation and 1,536 free API calls including loading.
  Stable buffer ownership, graph thread restrictions and independently retained
  snapshots are unresolved design work. No capture break-even or bounded replay
  storage benefit is established; no unsafe Send/Sync workaround is acceptable.
- **No-go for level-zero tables or new generic artifact ownership now.** The two
  tables would add 2,916,352 logical bytes, versus the measured 25,680-byte prepared
  state, with no demonstrated startup/latency break-even after reconstruction
  reuse. Selective/streaming ownership would change authentication/lifetime
  boundaries and needs separate review. The raw-byte upload PoC is not that design.

These are bounded no-go/defer decisions, not claims that further optimization is
impossible. MLX-specific decisions remain pending actual Apple work.

## Validation and build-cache incident

Final `tools/check.sh --cuda` passes, including **36 report-tool tests**. A fresh
capture passes **642 trained traces / 2,500 decisions**; cleared-environment
real tokenizer probes pass. Production benchmark hashes match the frozen matrix.

A shared Cargo target directory used for the *separate diagnostic baseline build*
subsequently caused a validation build to reuse old release artifacts. The new
cache integration test failed (no retained geometry), and benchmark executable
hashes identified the accepted-baseline binaries rather than the candidate.
This failed check and hashes are preserved. Cleaning **only** local CUDA/model
release package outputs, rebuilding and rerunning checks restored the exact frozen
candidate hashes and passed the test. No failed check was counted as passing.

Timing/resource/lifecycle collection had already finished before that baseline
build. GDB capture hashes and saved memory-probe binaries bind their actual
implementations. Fresh clean-build qualification and hash verification confirm
the final state. Future cross-worktree builds must use isolated target directories
or explicitly clean/rebuild and verify hashes; source checkout identity alone
is insufficient. Frozen historical binaries/worktrees remain untouched.

## Reproduction and remaining work

[evidence.json](evidence.json) hashes sources and the lossless **XZ** archive;
[summary.json](summary.json) retains resource/memory/metadata summaries. The archive
contains all 405 matrix/control process reports (plus explicitly shared Python
copies), telemetry, fixed plan, both negative audits, fresh checks/qualification,
resource/lifecycle reports, all six memory captures, debugger attempts and symbols.
It contains no executable, licensed weight or trained trace tensor.

```sh
D=docs/results/2026-11-11-phase2-linux-controls
# Extract into a new clean checkout with an empty reports/ directory.
mkdir -p reports
tar -xJf "$D/raw/linux-controls-v1.tar.xz" -C reports
python3 tools/bench/phase2_summary.py reports/phase2-optimization-gate-threeway-v1/previous /tmp/control-previous.json
python3 tools/bench/phase2_summary.py reports/phase2-optimization-gate-threeway-v1/candidate /tmp/control-candidate.json
python3 tools/bench/audit_phase2_performance.py reports/phase2-optimization-gate-v1/controlled-final-config.json /tmp/controlled-final-audit.json
# Expected audit exit 1: exactly 72/78, not a passing result.
```

Native memory diagnostic: build `--example codec_memory --features cuda --release
--locked`; run `codec_memory CANONICAL_BUNDLE NEW_REPORT` in three fresh processes
per implementation. Build the baseline diagnostic in a separate worktree/target,
using the same example source. Debugger launch is documented by the archived
script/symbol map and this invocation (new output required):

```sh
NCM_METADATA_SYMBOLS="$PWD/reports/phase2-optimization-metadata-v1/symbols.json" \
NCM_METADATA_REPORT="$PWD/NEW_METADATA_REPORT.json" taskset -c 2 \
  gdb --batch -nx -nh -iex 'set auto-load off' -ex 'set debuginfod enabled off' \
  -ex 'set disable-randomization off' -ex 'set pagination off' \
  -x tools/bench/gdb_codec_metadata.py --args target/release/examples/codec_profile \
  artifacts/canonical/sudoku-easy-qualified
```

Remaining requirements are explicit in the [working completion audit](../2026-11-11-phase2-optimization/completion-audit.md).
The six Linux failures require a defensible next controlled experiment/resource
arrangement; Apple remains blocked. Raw-upload production/API review remains
unapproved and optional conditional work, not a silent loader change. Final
opportunity dispositions, Apple evidence and dual-platform closure remain pending.
