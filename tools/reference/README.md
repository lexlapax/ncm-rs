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

See [artifact security](../../docs/ARTIFACTS.md), [testing](../../docs/TESTING.md),
and the [first measured report](../../docs/results/2026-11-09-cuda-operators/README.md).
