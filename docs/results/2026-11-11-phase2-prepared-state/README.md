# Phase 2 prepared-state experiments and raw-upload PoC

**Provisional Linux experiments, not acceptance. Phase 2 remains open.**
The accepted comparator remains `3307dff`/production `083c1e5`. These incremental
experiments instead compare to the published lower-risk candidate `bcf2be4` to
attribute additional work. They cannot replace the final 78-cell comparison
against accepted Rust and qualified Python. No Apple run has started.

## Prepared state

First, shared model code uses the validated positional table directly, rather
than uploading sequential IDs and gathering identical rows. Immutable codebook
squared norms are lazily prepared per model/level. Distance ordering remains
`xx + ee - 2*dot`, including the row-constant term and first-index tie semantics.
Initialization errors remain retryable; simultaneous initializers can compute
extra temporaries, but publish one immutable result. Tests cover retry/reuse,
non-Send values, separate owners/drop, and concurrent publication.

Second, CUDA prepares reusable linear/area geometry on its fixed native device.
The cache holds no weights or activations. It has **64-entry and 1 MiB logical
metadata admission bounds**; additional shapes execute the same operator without
retention. These are not GPU allocator or RSS bounds. Backend clones share the
device but start with empty caches, so retaining a caller's backend does not
retain another owner's prepared geometry. Locks are released before execution.
No precision, coordinate formula, reduction order or public snapshot changed.

CPU tests cover reuse across batches, distinct algorithms, admission limits and
invalid sizes. A dedicated real-CUDA test exercises all twelve trained lengths,
repeated use, different inputs, output survival, clone isolation, and three
owner lifetimes. Synchronized CUDA **used** pool memory stabilizes after the first
pass and returns exactly to the pre-owner baseline after each drop. This is not a
claim about reserved pool memory, whole-model host RSS, or MLX lazy lifetimes.

Both stages pass fresh **642-trace / 2,500-decision** qualification under unchanged
budgets and `tools/check.sh --cuda`. Untraced/traced outputs and retained snapshots
also pass the existing exact comparisons. The geometry stage adds three unit
tests and one real-device integration test. Final check log is
`phase2-optimization-geometry-v1/check-cuda-v3.log` in the archive. The earlier
Clippy failure (test module placement) and rejected reversed trace-path invocation
are retained; the latter stopped at "output must not exist", without modifying
the oracle. Neither failed attempt counts as validation.

### Work attribution and timing limitations

CUPTI workload: checked load plus two B1 reconstructions. Instrumented counts and
summed durations are diagnostics, never acceptance latency.

| Candidate | Kernels | Driver events | H2D copies | H2D bytes, including weights |
| --- | ---: | ---: | ---: | ---: |
| Published lower-risk candidate | 918 | 24,418 | See prior report | See prior report |
| Positions + norms | 892 | 23,846 | 764 | 59,663,088 |
| Positions + norms + geometry | 892 | 22,936 | 694 | 59,638,176 |

Geometry eliminates **70 uploads / 24,912 bytes** in this two-request capture.
That agrees with the trained geometry's logical storage; adding 768 bytes of
codebook norms gives the research estimate of 25,680 bytes. Tensor objects,
allocator rounding and temporary initialization allocations are additional.
The prior row comes from the [lower-risk report](../2026-11-11-phase2-low-risk/README.md),
not a new instrumented run. Each new stage has its own fresh rotating three-process
Rust/Rust/Python matrix, CPU affinity 2, five warmups and thirty trials.

Combined prepared-state medians versus `bcf2be4`, milliseconds:

| Batch / scope | Previous | Candidate | Python | Candidate/previous |
| --- | ---: | ---: | ---: | ---: |
| B1 encode | 2.0661 | 1.9408 | 3.5802 | 0.9394 |
| B2 encode | 2.4072 | 2.2985 | 3.4952 | 0.9549 |
| B4 encode | 3.0280 | 2.9183 | 3.9147 | 0.9638 |
| B1 reconstruction | 2.0811 | 2.1731 | 4.6949 | 1.0442 |
| B2 reconstruction | 2.6810 | 2.5719 | 5.1121 | 0.9593 |
| B4 reconstruction | 3.3830 | 3.2785 | 5.9028 | 0.9691 |
| B1 resident block | 0.1774 | 0.1970 | 0.2605 | **1.1106** |

**The B1 resident-block point regression is unresolved.** That scope does not use
prepared geometry/norms, but this does not justify waiving the failure or asserting
its cause is noise. Earlier positions/norms-only reconstruction ratios were
0.8879/0.9746/0.9810; movements in unchanged scopes prevent interpreting differences
between the two matrices as an isolated geometry speedup. Retain all runs and
resolve final-candidate uncertainty with predeclared matched controls. No final
keep decision, full resource acceptance, or Linux gate is claimed here.

## Checked raw-byte upload PoC — production loader unchanged

`crates/ncm-backend-cuda/examples/weight_upload_poc.rs` compares decoded `Vec<f32>`
upload with pinned Candle `Tensor::from_raw_buffer`. It authenticates the pinned
bundle and runs full existing model preflight first. Shape/product/byte-length
and little-endian guards precede native conversion. It adds no unsafe code.
CPU tests exercise aligned/unaligned buffers, bit-exact ownership after source
mutation, truncated elements, size mismatch, zero dimensions and overflow.

Three CUDA processes each run five warmups and thirty alternating trials over
all **58 runtime tensors / 59,586,048 bytes**. Native readback is byte-exact.
Pooled upload-only medians are **13.512509 ms decoded-vector / 7.379914 ms raw**.
These exclude authentication, preflight, model construction and destruction;
they do not establish full-load speedup, RSS savings, or zero-copy GPU storage.

A production implementation would add a validated little-endian byte-upload
method to public `WeightBackend`, keeping the existing FP32 method and complete
preflight before device allocation. **That API decision is awaiting user approval.**
The loader and public backend trait are unchanged; there is no silent new ownership
or trust contract. The example is explicitly CPU-feature-gated for default builds.

## Evidence and reproduction

[evidence.json](evidence.json) fingerprints sources, the archive, stage identities
and all saved snapshot executables. No executable, licensed weight, or trained
trace tensor is committed. The first stage's executable is identified by its raw
matrix hash, not claimed to be retained locally. The second stage's snapshot is
saved at `.reference/phase2-geometry-snapshot-bin/` and matches its matrix hash.
The clean `bcf2be4` source/binaries are preserved at
`.reference/phase2-low-risk-baseline{,-bin}`.

The archive contains both complete matrices, hardware/telemetry/commands/source
hashes, full summaries, qualification reports, CUPTI text/summary, checks,
intermediate patches and all raw-upload samples. Apply `prepared-state.patch`
to `bcf2be4` for the first runtime; additionally apply `geometry.patch` for the
second. Current source adds the tests and diagnostic example declaration.

Recompute both summaries without GPU, weights or Python ML environments:

```sh
D=docs/results/2026-11-11-phase2-prepared-state
mkdir -p /tmp/ncm-prepared-reproduction
# Use a new/empty extraction directory.
tar -xzf "$D/raw/experiments-v1.tar.gz" -C /tmp/ncm-prepared-reproduction
python3 - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, 'tools/bench')
from phase2_summary import codec
from summarize_cupti_trace import summarize
root = Path('/tmp/ncm-prepared-reproduction')
for stage in ('prepared', 'geometry'):
    raw = root / f'phase2-optimization-{stage}-v1'
    matrix = root / f'phase2-optimization-{stage}-threeway-v1'
    result = {name: codec(matrix/name) for name in ('previous', 'candidate')}
    assert result == json.loads((raw/'comparison.json').read_text())
    profile = json.loads((raw/'cupti.json').read_text())
    profile.pop('summarizer_sha256')
    assert summarize((raw/'cupti.txt').read_bytes()) == profile
print('Both stage summaries and profiles reproduced; acceptance remains pending.')
PY
```

Core live commands (output paths must be new):

```sh
NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 ./tools/check.sh --cuda
CARGO_BUILD_JOBS=4 cargo test -p ncm-backend-cuda --features cpu --example weight_upload_poc --locked
# Build release CUDA codec_bench/codec_profile/codec_trace/codec_lifecycle examples.
python3 tools/bench/run_phase2_linux_comparison.py \
  .reference/phase2-low-risk-baseline .reference/phase2-low-risk-baseline-bin \
  NEW_MATRIX --cpu 2 --workloads codec
target/release/examples/codec_trace artifacts/canonical/sudoku-easy-qualified \
  reports/phase2-oracle/v1/sudoku-cuda NEW_CAPTURE cuda
.reference/venv/bin/python tools/reference/qualify_codec.py \
  reports/phase2-oracle/v1/sudoku-cuda NEW_CAPTURE NEW_QUALIFICATION.json
# Run weight_upload_poc in three fresh processes, taskset -c 2, new report each time.
```

Still pending: hot weight/trace-label attribution, API decision/full-load testing,
final CPU/allocation/lifecycle/host-memory evidence, uncertainty controls, full
Linux gate, then Apple testing/vectorization, remaining opportunity dispositions
and the five-step completion audit. Phase 3 remains unstarted.
