# Inference setup — working draft

**Not a verified release recipe.** Phase 5 is still in progress. The commands below document
current development inputs and builds; they have not been followed end-to-end from a fresh clone
on Linux or macOS. In particular, qualified Sudoku medium/hard manifest reproduction and LM1B
text-input behavior need decisions. Do not treat this document as the v0.1.0 setup gate.
See the [current state](../agent-context-map.md) and [export investigation](results/2026-11-19-phase5-cli/linux-export-repro-v1.md).

Production inference is Rust plus native CUDA/cuDNN or MLX. Python and network access below are
for **offline preparation only**. No model weights, evaluation data or native binaries are in Git.
Check the model and tokenizer licenses before downloading or using them. Use a trusted, stable
artifact directory; the loader validates integrity but the manifest itself must have a trusted hash.

## Toolchains and local builds

Rust 1.99.1 or newer, rustfmt and Clippy are required. The default workspace does not require a
GPU SDK. A GPU build must explicitly select the corresponding feature and device; there is no
fallback. The Python locks below are platform-specific and are not production runtime dependencies.

### Linux / NVIDIA CUDA

The tested development machine uses an RTX 3060 Ti (compute capability 8.6), CUDA 13.4, a
compatible driver, GCC 15 and cuDNN from the pinned reference environment. Use the capability and
host compiler appropriate to your toolkit; a build on another tuple is not qualified by this record.
The following demonstrates the existing local setup, not a system-wide installation:

```sh
python3 -m venv .reference/venv
.reference/venv/bin/python -m pip install -r tools/reference/requirements-linux.lock
mkdir -p .reference/cudnn-link
# First confirm this path and the cuDNN version on your system. Never overwrite an existing link.
cudnn="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib"
test -f "$cudnn/libcudnn.so.9" || { echo 'cuDNN library not found' >&2; exit 1; }
test ! -e .reference/cudnn-link/libcudnn.so || { echo 'cuDNN link already exists' >&2; exit 1; }
test ! -L .reference/cudnn-link/libcudnn.so || { echo 'cuDNN link already exists' >&2; exit 1; }
ln -s "$cudnn/libcudnn.so.9" .reference/cudnn-link/libcudnn.so || exit 1
export LIBRARY_PATH="$PWD/.reference/cudnn-link"
export LD_LIBRARY_PATH="$cudnn"
export NVCC_CCBIN=/usr/bin/g++-15 CUDA_COMPUTE_CAP=86 CARGO_BUILD_JOBS=4
cargo build --release --locked -p ncm-cli --features cuda-cudnn
```

The CLI uses `cuda-cudnn` for XSum/LM1B and strict TinyStories; qualified WMT14 selects `cuda`
from the same `cuda-cudnn` build. Sudoku selects `cuda`. `cuda` without cuDNN is a separate build
and does not enable all of these paths. For local validation, unset `LD_LIBRARY_PATH` before running
report-tool tests that reject injected library paths, then run the documented CUDA check:

```sh
unset LD_LIBRARY_PATH
setpriv --inh-caps=-all ./tools/check.sh --cuda
```

The development machine's test shell requires `setpriv --inh-caps=-all`; CUDA binaries using the
reference cuDNN need `LD_LIBRARY_PATH` set again for inference. A batch accepted by the CLI may
still exceed available GPU memory.

### Apple Silicon / MLX

The [native MLX setup](../tools/reference/README.md#apple-silicon--mlx-phase-1) records the tested
M1 Ultra, macOS, Rust, MLX wheel and project-local prebuilt native-library procedure. It uses
`MLX_SYS_PREBUILT`, an absolute local rpath and `cargo` feature `mlx`; the wheel requires macOS
26.2 or later. It is a development configuration, not a redistributable prebuilt binary. After
preparing the matching native prefix as described there:

```sh
export MLX_SYS_PREBUILT="$PWD/.reference/mlx-native"
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$MLX_SYS_PREBUILT/lib"
export RUSTDOCFLAGS="$RUSTFLAGS" CARGO_BUILD_JOBS=8
cargo build --release --locked -p ncm-cli --features mlx
./tools/check.sh --mlx
```

The Phase 5 unified CLI and these setup instructions have not yet passed a fresh-clone Metal gate.
Do not install the Linux Torch lock on macOS. Offline artifact preparation may be done separately
on Linux and the verified artifacts transferred; a self-contained Mac preparation recipe is pending.

## Pinned sources and tokenizer assets

Upstream code is pinned to `c70274273f489abc73832b2b98584a8483b6b203` and checkpoints to
`hazyresearch/ncm-checkpoints` revision `d342ce8aaea58830eef3b8ad473664673e41f692`.
The tokenizer sources are separately locked. For a new checkout and fresh destinations:

```sh
mkdir -p .reference artifacts/upstream artifacts/canonical reports
git clone https://github.com/HazyResearch/ncm.git .reference/upstream-ncm
git -C .reference/upstream-ncm checkout --detach c70274273f489abc73832b2b98584a8483b6b203
python3 -m venv .reference/tokenizers-venv
.reference/tokenizers-venv/bin/python -m pip install -r tools/reference/requirements-tokenizers.lock
python3 tools/reference/fetch_tokenizers.py artifacts/upstream/tokenizers-v2 \
  --lock tools/reference/tokenizer-sources.lock.json
.reference/tokenizers-venv/bin/python tools/reference/convert_tokenizers.py \
  artifacts/upstream/tokenizers-v2 .reference/upstream-ncm artifacts/canonical/tokenizers-qualified
```

The tokenizer fetch is separate from the much larger checkpoint downloads. The current scripts
verify source hashes and refuse existing output destinations. The checked sources and converters,
including licenses, are described in the [tokenizer report](results/2026-11-10-phase2-tokenizers/README.md).

## Checkpoint downloads

`tools/reference/fetch_checkpoints.py` imports the exporters' existing source sizes and SHA-256
pins in the Linux reference environment. It fetches one explicit profile at the existing immutable
revision, writes at most the pinned bytes per file (reading one extra byte to detect overflow),
checks each size/hash and rejects non-HTTPS final redirects. A caller-provided total byte budget
and free-space check are required. A plan prints files, hashes and total without creating a directory
or using the network. Parent directories
must be trusted and stable. Download failure leaves the new partial directory for inspection; the
tool never overwrites an existing destination. Free space is not reserved and is not a bound on
subsequent export, memory use or other concurrent work.

For example, with the Linux reference environment installed and a fresh destination:

```sh
mkdir -p artifacts/upstream
.reference/venv/bin/python tools/reference/fetch_checkpoints.py \
  --profile sudoku-easy --output artifacts/upstream/sudoku-easy --plan
.reference/venv/bin/python tools/reference/fetch_checkpoints.py \
  --profile sudoku-easy --output artifacts/upstream/sudoku-easy --budget-bytes 162378392
.reference/venv/bin/python tools/reference/fetch_checkpoints.py \
  --profile sudoku-easy --output artifacts/upstream/sudoku-easy --verify
```

Choose the desired profile and budget explicitly. `--plan` prints the per-file identities; the
currently pinned total source sizes in bytes are:

| Profile | Bytes |
| --- | ---: |
| `sudoku-easy` | 162,378,392 |
| `sudoku-medium` | 162,378,394 |
| `sudoku-hard` | 162,378,392 |
| `tinystories` | 578,228,368 |
| `xsum` | 653,983,635 |
| `wmt14` | 1,126,375,067 |
| `lm1b-small` | 987,981,523 |
| `lm1b-medium` | 1,865,857,368 |
| `lm1b-large` | 3,499,027,351 |

Each LM1B size download includes a copy of the shared tokenizer checkpoint; plan disk space
accordingly. The exporters recheck the source sizes and SHA-256 before opening checkpoints using
`torch.load(weights_only=True)`; they never retry unrestricted pickle. The downloader was screened
against the retained inputs for all profiles and with one real Sudoku-easy download. A complete
fresh-clone download/export workflow on both machines remains unverified. The
[download screen](results/2026-11-19-phase5-cli/checkpoint-download-screen-v1.md) records its exact scope.

## Export and manifest identities — incomplete

Exporters and all output paths are explicit and refuse existing artifacts. XSum v2 additionally
needs the pinned offline rotary reference; LM1B needs the hash-pinned sidecar outside each artifact.
A local investigation reproduced seven qualified manifests, using different historical exporter
commits and Python patch versions; Sudoku medium/hard's historical exporter has not been located.
The exact identities, provenance-only differences and failed Python 3.14.9 `_ssl` installation are
recorded in the [investigation](results/2026-11-19-phase5-cli/linux-export-repro-v1.md).

**Do not substitute a new manifest hash for a qualified one.** Complete, reviewed commands for
all exporters, a portable reference environment, explicit manifest-hash checks and a clean-clone
replay are still required here. Existing [Sudoku export](../tools/reference/README.md#sudoku-easy-checkpoint-and-oracle-checks),
[XSum rotary export](results/2026-11-15-phase4-xsum/rotary-artifact-profile-v2.md),
[WMT14 export](results/2026-11-16-phase4-wmt14/artifact-plan-v1.md) and
[LM1B export](results/2026-11-16-phase4-lm1b/medium-large-artifacts-v1.md) reports are historical
records, not a substitute for the missing release recipe.

## Inference examples — pending

Run `target/release/ncm --help` and `target/release/ncm sample --help` for the current interface.
Fully reproducible examples, including safe creation of input files and the LM1B rotary sidecar,
will be added and tested from fresh clones on both machines before release. The CUDA-only
[CLI equivalence screen](results/2026-11-19-phase5-cli/cuda-routing-screen-v2.md) and
[text-input screen](results/2026-11-19-phase5-cli/cuda-text-input-v1.md) describe the current
limited development checks, not the release gate.
