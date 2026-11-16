# XSum CUDA classifier-free guidance

The restricted temperature-1, top-k-1, top-p-0 path now supports nonnegative
CFG scales. Positive scales use the pinned Python document/null prefix batch,
level-dependent `(1+t)*conditioned - t*null` scores, and shared selected codes
in both branches. Scale zero uses the existing path. Positive CFG accepts up to
64 input rows because the internal batch doubles. Invalid scales, nonfinite
scores and exact head ties fail explicitly; other sampling modes remain unsupported.

On an RTX 3060 Ti, a two-row synthetic padded input at scales 1 and 2 matched
pinned Python CUDA on all 400 intermediate codes and 96 penalized output IDs
per scale. The comparison exercises null prompting and all nine levels; it is
not a full-dataset quality result or a general proof for other inputs. The
previous full XSum quality result was obtained at CFG zero on the earlier
source; the new source needs its own source-bound comparison.

The two traces under `reports/phase4-xsum-cfg-reference-v1/` and `-v2/` have
SHA-256 `57d2be1db75f282d6ed47273580c6a3619a41495ee4a2b626e986911b1e46a55`
and `97d5a79424cdd44bbf51db39517b8cc32bc21db8f9ea0008fadc01b4ee3916e0`;
their native JSON files hash to
`b68f2e04d0b2d77e66664b62d8ac7d34d8e6d4c4e7a57592b92c043bf3f518dd` and
`eb5138e5dc2dc58b982f378bf2e387d0e50eb797e233555eacb11e9e3a494a0b`.
The release CLI binary hash was
`43b8fb92a5c2c2e4114f8cf76d78b5500df06ecdfdab338fc3bfd9cb0733465c`.
The reference script verifies the pinned upstream revision and input checkpoint
hashes. Native-library identity was not separately captured for this sample.

For each scale, generate a new trace with:

```sh
OMP_NUM_THREADS=1 .reference/venv/bin/python tools/reference/xsum_cfg_trace.py --upstream .reference/upstream-ncm --source artifacts/upstream/xsum-phase4 --output NEW_REFERENCE --cfg-scale SCALE
```

Convert `generator.prefix_ids` in the trace to a JSON list of rows. Run
`ncm sample-xsum artifacts/canonical/xsum-inference-rope-v2-final cuda-cudnn INPUT.json SCALE`
with the local cuDNN library path and save stdout. Then use
`tools/reference/xsum_cfg_compare.py --reference NEW_REFERENCE/trace.safetensors --native OUTPUT.json`
with `.reference/venv/bin/python`; it checks shapes and exact codes/IDs.

`tools/check.sh --cuda` passed on this source with all 18 real-device operator
cases; the log is `reports/phase4-xsum-cfg-check-v1.log` (SHA-256
`8f583e3e8a7017e3cb05d0847a781d27df6babff9ffe733ab64c54d3908f8d25`).
Strict CUDA/cuDNN CLI Clippy and Python syntax checks also passed. The broader
CUDA performance and quality gates and XSum MLX validation remain pending.
