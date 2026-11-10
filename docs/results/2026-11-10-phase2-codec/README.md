# Phase 2 complete shared codec: Linux/CUDA parity

**The FP32 Sudoku codec passes the declared CUDA trace gate. Phase 2 is NOT
complete:** tokenizer assets, Linux performance/memory acceptance, Apple codec
qualification and final publication remain pending. This is not a generator,
Sudoku-solving evaluation, inference CLI or inference release.

## Scope and unchanged identity

`ncm-model` now executes the lookup encoder, pre-quant GroupNorm, residual codebook
quantization, learned cascades, Phi refinement, bidirectional decoder, encode,
fixed-code decode and reconstruction over `CodecBackend`. All state is request-local;
returned per-level snapshots remain independently observable after later calls.
Intermediate codes stay on-device for reconstruction. Production execution has no
Python dependency; the native example authenticates reference inputs, runs Rust
and exports observations for a separate offline comparison.

The upstream/checkpoint pins, Phase 1 goldens and predeclared `codec-fp32-v1`
budgets are unchanged. See [PHASE2.md](../../PHASE2.md). Canonical trained reference
is the original CUDA capture, not the optional failing CPU diagnostic. Native
weight readbacks are authenticated against the exact committed inventory before
capture. Reference files supply only input/supplied IDs, never candidate outputs.

## Results

- Full tiny codec passes CPU and real CUDA: all **236 captured tensors** across two
  cases, all integer values exact. Additional checks retain/re-read every API
  snapshot and compare uninstrumented reconstruction/fixed decode. The runner
  checks 252 floating comparisons / 112,212 elements including these repeated
  boundaries; largest absolute error CPU `2.288818359375e-5`, CUDA `1.9073486328125e-5`.
- Trained Sudoku CUDA passes **all 642 captures** across solved B2, blank B1 and
  mixed B2, free-running reconstruction and supplied-code decoding. All **2,500**
  code/output decisions agree exactly; candidate IDs also agree with first-index
  selection from their own captured scores. All fixed input IDs agree.
- Two reference near-margin decisions remain: mixed-board flattened position zero
  at level 1 (gap `0.0046443939208984375`) and level 2 (gap
  `0.0005822181701660156`). Both select the same ID as Python. Their positions,
  margins and stability are retained; they do not excuse any floating failure.
- Two fresh native processes produce the same trace-file SHA-256:
  `a3776cac0ba75eaeb9ebc9a30ebd03c131f0b4470f12f136b7ccc95fc25e0fa0`.
  Uninstrumented code IDs, logits and output IDs equal the instrumented execution.

| Trained boundary | Elements | Max absolute error | Mean absolute error | Max fraction of allowed elementwise error |
| --- | ---: | ---: | ---: | ---: |
| Component | 19,333,560 | 0.0026702881 | 1.31251e-6 | 0.941824 |
| Distances | 25,760 | 1.5 | 0.00172172 | 0.0568543 |
| Accumulated/block/logits | 4,578,160 | 0.0002746582 | 1.48284e-6 | 0.211810 |

Large absolute differences occur on large-magnitude values; acceptance is the
unchanged **absolute plus relative** elementwise budget, not absolute error alone.
The closest boundary is blank-board level-zero learned downsampling, channel 113:
reference `4.46671724319458`, native `4.467652797698975`, error
`0.0009355545043945312`, allowed `0.000993343448638916`. It uses **94.2% of the
budget**; this limited margin must not be hidden or treated as Apple evidence.
The distinct original-Python CPU/CUDA failure remains in the oracle audit.

Raw `qualification.json` retains every tensor's errors/mismatch counts,
policy-wise absolute-error histograms, normalized worst errors, all decisions and
reconstruction agreement. No reference or tolerance was regenerated to pass.

### Reconstruction, not perfect recovery

Native vs Python output agreement is **445/445 tokens and 5/5 sequences** for
free-running reconstruction, and likewise for supplied-code decoding. Recovery
of original input tokens is different: solved 178/178, blank 89/89, mixed 170/178.
Both runtimes lose the same eight mixed-input tokens. Arbitrary supplied codes
recover only 74/445 original tokens, as expected for this diagnostic. No general
lossless-codec or official task-quality claim follows from these synthetic boards.

## Reproduce

On the existing RTX 3060 Ti, Rust 1.99.1, Candle 0.11.1 environment:

```sh
NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 ./tools/check.sh --cuda
.reference/venv/bin/python -m unittest discover -s tools/reference -p 'test_*.py' -v
NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 \
  cargo run --release -p ncm-backend-cuda --features cuda --example codec_trace -- \
  artifacts/canonical/sudoku-easy-qualified reports/phase2-oracle/v1/sudoku-cuda \
  reports/phase2-codec/cuda-v2 cuda
.reference/venv/bin/python tools/reference/qualify_codec.py \
  reports/phase2-oracle/v1/sudoku-cuda reports/phase2-codec/cuda-v2 \
  reports/phase2-codec/cuda-v2-qualified-final.json
```

Use fresh output paths. The native capture verifies the committed reference report,
trace/config hashes and canonical config/weights. Source hashes and the exact
candidate trace hash are in `raw/native-cuda.json`. `raw/qualification.json` records
the comparison and qualification tool hashes. Large raw traces remain ignored
under `reports/phase2-codec/`; durable archive publication is still pending.

Full local `--cuda` checks pass (24 base Rust tests plus real-CUDA suite, seven
report-tool tests, unchanged 18 operator cases, fmt/lint/docs/syntax). All 13 offline
reference tests pass. Source is the accompanying commit; no Apple work was started.
No latency, throughput or memory improvement is claimed: matched uninstrumented
Python/native measurements remain required before Linux Phase 2 acceptance.
