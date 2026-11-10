# Offline reference tools

`operators.py` generates synthetic FP32 PyTorch operator fixtures and runs the same
cases as synchronized eager CUDA microprobes. It does not load checkpoints or
execute the NCM model. Separate `sudoku_artifact.py` and `sudoku_trace.py` tools now
support the scoped Sudoku-easy inspection profile; no general/data importer exists.

Use an isolated Linux Python 3.14 virtual environment:

```sh
python3 -m venv .reference/venv
.reference/venv/bin/python -m pip install -r tools/reference/requirements-linux.lock
# Review a NEW fixture file; never overwrite the established reference implicitly.
.reference/venv/bin/python tools/reference/operators.py generate --fixture .reference/new-operators.json
.reference/venv/bin/python tools/reference/operators.py run \
  --fixture tests/fixtures/operators-v1.json --device cuda --output reports/new-python-run.json
```

Create output directories first. Writers refuse existing output files. The Linux
lock includes NVIDIA dependencies and is not an Apple requirements file. Torch is
pinned to 2.15.0; reference attention explicitly uses the math SDPA backend and
TF32 is disabled. CUDA requests fail rather than silently falling back to CPU.
These tools never become a production Rust dependency.

## Sudoku-easy checkpoint and oracle checks

Explicit downloads total about 162 MB; no evaluation pickle is needed. Fetch only
these three files, then verify the hardcoded sizes/SHA-256 through the exporter.
Never substitute an unpinned revision or retry unrestricted pickle.

```sh
mkdir -p artifacts/upstream/sudoku/easy reports/sudoku-new
revision=d342ce8aaea58830eef3b8ad473664673e41f692
for name in config.yaml tokenizer.pt generator.pt; do
  destination="artifacts/upstream/sudoku/easy/$name"
  test ! -e "$destination" || exit 1
  curl --fail --location --max-filesize 103000000 \
    "https://huggingface.co/hazyresearch/ncm-checkpoints/resolve/$revision/sudoku/easy/$name" \
    --output "$destination" || exit 1
done
# Reuse an already verified clean checkout, or clone explicitly:
git clone https://github.com/HazyResearch/ncm.git .reference/upstream-ncm
git -C .reference/upstream-ncm checkout --detach c70274273f489abc73832b2b98584a8483b6b203
.reference/venv/bin/python -m unittest discover -s tools/reference -p 'test_*.py' -v
.reference/venv/bin/python tools/reference/sudoku_artifact.py \
  --source artifacts/upstream/sudoku/easy --output artifacts/canonical/sudoku-new \
  --report reports/sudoku-new/export.json
.reference/venv/bin/python tools/reference/sudoku_trace.py \
  --upstream .reference/upstream-ncm --source artifacts/upstream/sudoku/easy \
  --bundle artifacts/canonical/sudoku-new --traces reports/sudoku-new/traces.safetensors \
  --report reports/sudoku-new/oracle.json \
  --tokenizer-fixture reports/sudoku-new/tokenizer.json
cmp tests/fixtures/sudoku-tokenizer-v1.json reports/sudoku-new/tokenizer.json
cargo run -p ncm-backend-cuda --features cpu --example artifact_probe --release --locked -- \
  cpu artifacts/canonical/sudoku-new reports/sudoku-new/candle-cpu.json
NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4 \
  cargo run -p ncm-backend-cuda --features cuda --example artifact_probe --release --locked -- \
  cuda artifacts/canonical/sudoku-new reports/sudoku-new/candle-cuda.json
```

The trace tool requires a clean pinned checkout and real CUDA. Two deterministic
synthetic boards exercise codec encode/decode and all 12 dense, uncached generator
levels. Original verified checkpoints and canonical reloads must match bit-for-bit
at 56 captured tensors. Captures clone per-level accumulators rather than invoking
the known aliased snapshot helper; upstream source is unmodified. Math SDPA, FP32,
TF32 off, deterministic cuDNN, seed 42 and greedy generation are explicit. These
are conversion checks, **not** Rust model parity or official Sudoku quality.

All writers require fresh destinations. The exporter records its base Git commit
and exact source hash, so provenance/manifest hashes change across source revisions;
compare tensor hashes separately. Large traces and weights remain ignored; only
small fixtures, inventories and reports are checked in. The optional Torch test
suite above is separate from `tools/check.sh`, which needs no reference venv.

## Phase 2 codec references

`codec_trace.py` captures the pinned unmodified codec with read-only observations,
cloned cumulative snapshots and fixed-code decoding. `compare_codec_traces.py`
checks source/config/weight identity and the new predeclared budgets; it is not a
Rust parity claim. Canonical oracles are the tiny CPU fixture and trained Sudoku
CUDA captures. A full-checkpoint CPU/CUDA diagnostic has one internal mismatch,
retained without changing tolerances. See the
[contract](../../docs/PHASE2.md) and [audit/reproduction commands](../../docs/results/2026-11-10-phase2-oracle/README.md).

The native `codec_trace` example exports the shared Rust codec's observations.
`qualify_codec.py` verifies floating budgets, first-index selection consistency,
stable IDs and all near ties, and records error distributions plus reconstruction
agreement. It does not waive floating failures for near ties or claim performance.
See [complete Linux codec results](../../docs/results/2026-11-10-phase2-codec/README.md)
for exact commands and current limitations.

The tiny fixture stores deduplicated typed little-endian hex arrays. Raw Sudoku
traces remain ignored until durable archive publication; never replace established
fixtures with a new generation merely to hide a failure. The generator authenticates
config/codec/training files against the committed Phase 1 inventory, not an arbitrary
replacement manifest. It requires the pinned clean source checkout and refuses
existing output directories. It never loads pickle or starts an MLX job.

## Phase 2 tokenizer assets

`fetch_tokenizers.py OUTPUT --lock tools/reference/tokenizer-sources.lock.json`
fetches only immutable tokenizer/config/license files, verifies Git/LFS identities
and checks the reviewed size/SHA-256 manifest. It downloads no text-model weights.
`convert_tokenizers.py SOURCE UPSTREAM OUTPUT` then verifies every source hash,
interprets TinyStories' restricted data-only pickle opcodes without unpickling,
and runs pinned upstream wrappers locally/offline. No unrestricted loading or
remote-code fallback is allowed. Both commands require fresh output directories.

Use the separate `requirements-tokenizers.lock` environment; do not replace the
existing Torch reference environment. Canonical payload/manifest hashes are in
`tokenizers-v1.lock.json`, with original license notices copied into each bundle.
`tests/fixtures/tokenizers-v1.json` is the exact encode/decode oracle; the native
`asset_probe` checks all four profiles and rejects tampering/symlinks. See the
[full asset/parity report](../../docs/results/2026-11-10-phase2-tokenizers/README.md)
for commands and the explicit Unicode-16/reference-version policy. This supports
these wrappers, not arbitrary HF imports or full text-model inference.

## Phase 2 Linux measurements

`codec_bench.py` measures pinned eager CUDA codec execution and checked loading,
with explicitly snapshot-safe decode checked against the unmodified final decoder.
`codec_stages.py` adds separate instrumented stream spans; these are not headline
API timing. `codec_compile_probe.py` retains the failed optional fullgraph Inductor
preflight and never marks a compiled baseline qualified from output checks alone.
`text_tokenizer_bench.py` uses the original pinned wrappers and safe source loading;
canonical/native vs original/Python load formats are disclosed rather than hidden.

Build the native `codec_bench` and `asset_bench` examples, then use
`tools/bench/run_phase2_linux.py` for sequential alternating fresh-process runs.
`phase2_summary.py` checks complete matrices, matching outputs/scopes, fixed logit
budgets and timing counts before aggregating. See the [Linux gate report](../../docs/results/2026-11-10-phase2-linux/README.md)
for exact commands, raw trials, telemetry, memory scopes and negative results.

## Phase 2 MLX codec parity

The native `mlx_codec_trace` example uses the same authenticated trained-capture
harness as CUDA. Copy the pinned CUDA reference (not a newly generated Apple
oracle), run native capture, then qualify offline with `qualify_codec.py`.
See the [MLX correctness milestone](../../docs/results/2026-11-10-phase2-mlx-codec/README.md)
for exact commands, first-index/snapshot checks and the retained initial failure.
The independent Apple Python timing baseline and measurements remain pending.

## Apple Silicon / MLX Phase 1

Tested tuple: Rust 1.99.1, mlx-rs Git pin in Cargo.lock (0.34.0), native MLX
0.32.2, macOS 26.7.3 / Xcode 27.2, Python 3.14.10 for offline comparisons.
Do not install the Linux Torch lock on the Mac. Rust never invokes Python.

The tested Xcode installation lacked the optional Metal compiler component. Use
the matching prebuilt native libraries instead of changing global Xcode tooling:

```sh
export PATH=/opt/homebrew/bin:$PATH
uv venv --python 3.14.10 .reference/mlx-venv
uv pip install --python .reference/mlx-venv/bin/python -r tools/reference/requirements-macos.lock
.reference/mlx-venv/bin/python - <<'PY'
import pathlib, shutil, sysconfig
source = pathlib.Path(sysconfig.get_path('purelib')) / 'mlx'
target = pathlib.Path('.reference/mlx-native')
target.mkdir(exist_ok=False)
for name in ('include', 'share', 'lib'):
    shutil.copytree(source / name, target / name)
PY
export MLX_SYS_PREBUILT="$PWD/.reference/mlx-native"
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$MLX_SYS_PREBUILT/lib"
export RUSTDOCFLAGS="$RUSTFLAGS"
export CARGO_BUILD_JOBS=8
./tools/check.sh --mlx
cargo build -p ncm-backend-mlx --features mlx --examples --release --locked
mkdir -p reports/mlx-new
target/release/examples/mlx_probe tests/fixtures/operators-v1.json reports/mlx-new/rust-operators.json
.reference/mlx-venv/bin/python tools/reference/mlx_operators.py tests/fixtures/operators-v1.json reports/mlx-new/python-operators.json
# Use the same verified canonical artifact from the Linux exporter; no pickle on Mac.
target/release/examples/mlx_load_probe artifacts/canonical/sudoku-easy-qualified reports/mlx-new/rust-load.json
.reference/mlx-venv/bin/python tools/reference/mlx_load_probe.py artifacts/canonical/sudoku-easy-qualified reports/mlx-new/python-load.json
```

The prebuilt native prefix contains only native include/share/lib files; it remains
usable without the Python venv. The selected `macosx_26_0_arm64` wheel requires at
least macOS 26.2; older OS/wheel tuples are unqualified. Do not reuse other projects'
MLX dylibs. Match the native hashes in the [Apple report](../../docs/results/2026-11-10-mlx-phase1/README.md).
A source build with an installed Metal compiler is possible but was not qualified
here. The absolute local rpath is for this spike, not redistributable packaging.

For reports, repeat in three fresh processes (five warmups, thirty trials each),
alternating runtime order. Run the existing tokenizer probes on the Mac too. Use
`/usr/bin/time -l` to record Rust process RSS and preserve the native MLX counters
separately; unified memory is not CUDA VRAM. `tools/bench/phase1_summary.py` validates
and aggregates the documented `rust/python-{operators,load,tokenizer}-{1,2,3}.json`
layout. It rejects incomplete runs, failed checks, changed inputs/sources and
nonfinite timing values. Python MLX is an **adapted** baseline, not upstream NCM.

See [artifact security](../../docs/ARTIFACTS.md), [testing](../../docs/TESTING.md),
and the [first measured report](../../docs/results/2026-11-09-cuda-operators/README.md).
