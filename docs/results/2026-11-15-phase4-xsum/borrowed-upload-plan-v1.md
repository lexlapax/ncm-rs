# XSum host upload-copy candidate

The [intrusive Metal heap snapshots](mlx-owned-loader-screen-v1.md) show
substantial resident empty large allocations but do not prove their cause. The
XSum codec has two 77,266,944-byte FP32 weights; the current shared codec loader
constructs a temporary `Vec<f32>` for each weight before upload. The XSum
generator loader similarly constructs one temporary vector per tensor. Avoiding
those transient copies is a testable memory hypothesis, not an established fix.

The candidate adds a direct dependency on already-locked `bytemuck = 1.25.2` and
uses its safe checked alignment cast to borrow FP32 bytes only on little-endian
hosts with suitable alignment. Unaligned or other-endian data use the existing
little-endian decoding copy; malformed byte lengths fail explicitly. Existing
artifact integrity, complete host dtype/shape/finiteness preflight, backend
upload semantics and numerical bits are retained. The shared codec loader and
XSum generator loader use the helper; other generator paths are unchanged.
No tolerance, artifact or upstream pin changes.

The focused aligned/unaligned bit-preservation unit test and full
`setpriv --inh-caps=-all env RUSTC_WRAPPER= CARGO_BUILD_JOBS=4 ./tools/check.sh
--cuda` passed; the latter log is
`reports/phase4-xsum-borrowed-upload-cuda-check-v2.log`, SHA-256
`e111983b8fddf69084caf86060e08547dfe4ff16b88e207f95f808e0d3ebf5b0`.
The first check stopped at a corrected Clippy style error and remains retained.
The initial CLI comparison (`reports/phase4-xsum-borrowed-upload-cli-v1`) used
the previous binary: the combined Cargo command selected examples but did not
rebuild the CLI. Its match is baseline evidence only; an explicit candidate CLI
rebuild and rerun subsequently passed on all 32 reference rows. The corrected
binary SHA-256 is `2b32042990a96be5b6e5661b212b3367f62d12b91a998dd255f59e3b03b649ce`;
`reports/phase4-xsum-borrowed-upload-cli-v2/result.json` hashes to
`9401a2f712aa78864b6b6533b8d14cf5822c69f0bfa46e2849c3ba31c314a625`.
The quality and performance examples were
selected and rebuilt by that command: their captured executable hashes are
`fa10891b75dd4f58f09308a215066a918bbfc308064a0a2b7155ace7f5598713` and
`1834c2935faaf07fc2a05c710bcf93173a308b59027ffa69639366d8a67dd7b0`,
respectively, both distinct from their retained owned-loader predecessors.

On Linux, `setpriv --inh-caps=-all env RUSTC_WRAPPER= cargo run --release --locked -p ncm-io --example
xsum_payload_alignment -- artifacts/canonical/xsum-inference-rope-v2-final`
loaded the checked artifact and found all 53 codec FP32 tensors (242,533,376
bytes) and all 193 generator FP32 tensors (354,091,300 bytes) aligned in that
process. The probe source SHA-256 is
`ac4fc963642bfa882a71a94c98082835153655fc684e501ab19b0954e1db4c32`;
its log is `reports/phase4-xsum-payload-alignment-linux-v2.log`. The first
format check flagged the example's line wrapping; after formatting, the probe,
focused Clippy and format check passed. Alignment is
checked separately on every actual upload, so this observation is not an
assumption about other allocations or platforms.

The shared codec was also exercised on CPU against the retained Sudoku CUDA
oracle: `setpriv --inh-caps=-all env OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib" target/release/examples/codec_trace artifacts/canonical/sudoku-easy-qualified
reports/phase2-oracle/v1/sudoku-cuda reports/phase4-borrowed-upload-sudoku-cpu-v2
cpu`, with process-local `LD_LIBRARY_PATH` set to the existing cuDNN directory,
followed by `tools/reference/qualify_codec.py` on those two trace directories.
All 642 traces qualified with zero ID disagreements and two recorded near ties
(`reports/phase4-borrowed-upload-sudoku-cpu-v2-qualification.json`). The initial
v1 launch failed before execution because that library path was absent; its log
is retained. The qualification SHA-256 is
`c57ff01a88600078c393e2dc8a8c1ac5fe11dafcc702067c43b7cf664a610771`.
This CPU result does not replace real CUDA or Metal model checks.

Both strict-CUDA XSum quality shards completed with source and linked-library
postflight checks. The receipt-validated comparison against the previously
scored Rust CUDA capture found zero code and zero raw-ID differences across all
11,334 rows. `reports/phase4-xsum-borrowed-upload-quality-v1-comparison.json`
hashes to `74bd11e26b8834967c650a52f8e2c682c3a716961bbe5637753170244543950b`;
the candidate shard manifest hashes are
`90845c2a31886dc704252c27954aa1331b53d6be790570495c56b19a7eeeaf6b` and
`74625f110fe30457ae7c8a35800917ab21dafb09a3dae65ca4a979af75cabb88`.
This is an exact output comparison to a previously scored capture, not a fresh
scorer run, and does not change the prior stochastic exceptions.

The corresponding real-CUDA Sudoku codec trace, using the same command as the
CPU trace with `cuda` and output directory
`reports/phase4-borrowed-upload-sudoku-cuda-v1`, also qualified all 642 traces
with zero ID disagreements and two near ties. Its qualification SHA-256 is
`6a27bf1b1e5bc321e7eae509ecb886640f9b65080b87b30b20d1e25325b303d7`.
The repeated CUDA latency and RSS comparison used the pinned Python baseline,
three fresh processes per runtime and batch, five warmups and 30 synchronized
trials per process. The command was `setpriv --inh-caps=-all env
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .reference/venv/bin/python
tools/bench/run_phase4_xsum_bench.py --profile v2 --repetitions 3 --batches 1 4
32 --cudnn-dir .reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib
--output reports/phase4-xsum-borrowed-upload-performance-v1`.

| Batch | Runtime | Median of process medians ms | Pooled p95 ms | Median peak RSS MiB | Median sampled GPU peak MiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Rust | 157.33 | 158.00 | 994.19 | 838 |
| 1 | Python | 190.31 | 191.01 | 1840.39 | 990 |
| 4 | Rust | 568.70 | 569.86 | 990.24 | 1062 |
| 4 | Python | 690.95 | 692.31 | 1840.00 | 1624 |
| 32 | Rust | 4404.38 | 4414.09 | 991.90 | 3080 |
| 32 | Python | 5338.88 | 5347.13 | 1847.85 | 5672 |

Timing excludes load; process RSS includes it. GPU memory is sampled and can
miss peaks. Across 1,209 activity samples, all queries succeeded and four other
GPU processes occupied 488–490 MiB; the GPU was not exclusive. These results
favor Rust within the measured scope, but are not a paired old/new loader
comparison and do not establish a Metal memory improvement. The manifest,
timing summary and GPU summary SHA-256 values are respectively
`1218e6c669ef25ea6d43a63ca7989c504836227bc7962a684ed8ec0e094a7361`,
`7f4e666447811cd9c38c164e6445ae4b7820768f31235cecf48d2bf456855b04`, and
`8b37bd39b4330d30ca39b43ae7126b08d57017944a6d27e54af018675b72d751`.
Metal candidate testing and its still-failed exact raw-ID gate remain open.

Before any Metal candidate run, run full Linux CUDA checks and meaningful real
model parity and source-bound performance/quality comparisons. Then test on
Metal, including the existing raw-ID failure, full regression and RSS; keep any
negative result. The aligned/misaligned/bit-preservation unit regression is
prepared. The candidate remains unqualified until those checks and measurements
finish; a reduction in allocation copies is not by itself an RSS or speed result.
