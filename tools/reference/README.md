# Offline reference tools

`operators.py` generates synthetic FP32 PyTorch operator fixtures and runs the same
cases as synchronized eager CUDA microprobes. It does not load checkpoints or
execute the NCM model. The checkpoint/data exporter is still pending.

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

See [artifact security](../../docs/ARTIFACTS.md), [testing](../../docs/TESTING.md),
and the [first measured report](../../docs/results/2026-11-09-cuda-operators/README.md).
