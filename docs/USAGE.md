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
all exporters, a portable reference environment and a full clean-clone replay are still required.
The limited Sudoku-easy Linux replay below does not close those requirements. Existing [Sudoku export](../tools/reference/README.md#sudoku-easy-checkpoint-and-oracle-checks),
[XSum rotary export](results/2026-11-15-phase4-xsum/rotary-artifact-profile-v2.md),
[WMT14 export](results/2026-11-16-phase4-wmt14/artifact-plan-v1.md) and
[LM1B export](results/2026-11-16-phase4-lm1b/medium-large-artifacts-v1.md) reports are historical
records, not a substitute for the missing release recipe.

### Linux Sudoku-easy development replay

This is the **only** fresh-clone exporter/inference slice screened so far. It reproduces the
qualified easy manifest, not the unresolved medium/hard artifacts. The recorded Linux reference
uses Python 3.14.9 with Torch 2.15.0+cu132, the historical Git base and the exact exporter bytes
shown below. The newer reference venv from the CUDA section supplies the offline dependencies;
this build-tree Python is not a general-purpose environment. On the tested Arch host its `_ssl`
module failed to build, so use the system Python for downloads and pip; use this interpreter only
for the offline export. See the [screen](results/2026-11-19-phase5-cli/linux-fresh-clone-sudoku-v1.md)
for exact environment and limitations.

The following assumes a **full Git history**, the Linux reference venv and the freshly downloaded
`artifacts/upstream/sudoku-easy` source from above. It creates a project-local CPython source build
and a separate temporary exporter worktree. It does not replace existing files:

```sh
test ! -e .reference/python-3.14.9-src.tar.xz && test ! -L .reference/python-3.14.9-src.tar.xz || exit 1
test ! -e .reference/Python-3.14.9 && test ! -L .reference/Python-3.14.9 || exit 1
curl --fail --location --max-filesize 25000000 \
  --output .reference/python-3.14.9-src.tar.xz \
  https://www.python.org/ftp/python/3.14.9/Python-3.14.9.tar.xz
printf '%s  %s\n' e02d9ac21cebda6e46435369744a92c65cc7cd8669697694122cf97ffacd1e53 \
  .reference/python-3.14.9-src.tar.xz | sha256sum -c - || exit 1
python3 - <<'PY' || exit 1
import pathlib, tarfile
with tarfile.open('.reference/python-3.14.9-src.tar.xz', 'r:xz') as archive:
    for member in archive.getmembers():
        path = pathlib.PurePosixPath(member.name)
        if (path.is_absolute() or '..' in path.parts or not path.parts or
                path.parts[0] != 'Python-3.14.9' or member.issym() or
                member.islnk() or member.isdev()):
            raise ValueError(member.name)
PY
tar -xJf .reference/python-3.14.9-src.tar.xz -C .reference || exit 1
mkdir -p .reference/tmp
root="$PWD"
(
  cd .reference/Python-3.14.9 || exit 1
  TMPDIR="$root/.reference/tmp" ./configure --prefix="$root/.reference/python-3.14.9" --without-ensurepip || exit 1
  TMPDIR="$root/.reference/tmp" make -j4 || exit 1
) || exit 1
PYTHONPATH="$PWD/.reference/venv/lib/python3.14/site-packages" \
  .reference/Python-3.14.9/python -c 'import platform,torch; assert platform.python_version()=="3.14.9" and torch.__version__=="2.15.0+cu132"' || exit 1

test ! -e reports/easy-exporter-worktree && test ! -L reports/easy-exporter-worktree || exit 1
git worktree add --detach reports/easy-exporter-worktree f7aad49c3b904f76c86a3d5e36c3b552d2995000 || exit 1
test ! -e reports/easy-exporter-worktree/tools/reference/sudoku_artifact.py || exit 1
git show bbdf4222b734c4ebcb6cfa3ed1f23dbdf0c5f58c:tools/reference/sudoku_artifact.py \
  > reports/easy-exporter-worktree/tools/reference/sudoku_artifact.py || exit 1
printf '%s  %s\n' cfcae7d716fa7a9692c57019454bd28da5f8098c6d3bec32698795076dc0034e \
  reports/easy-exporter-worktree/tools/reference/sudoku_artifact.py | sha256sum -c - || exit 1
root="$PWD"
(
  cd reports/easy-exporter-worktree || exit 1
  PYTHONPATH="$root/.reference/venv/lib/python3.14/site-packages" \
    "$root/.reference/Python-3.14.9/python" tools/reference/sudoku_artifact.py \
    --source "$root/artifacts/upstream/sudoku-easy" \
    --output "$root/artifacts/canonical/sudoku-easy-qualified" \
    --report "$root/reports/sudoku-easy-export.json"
) || exit 1
printf '%s  %s\n' ad6a99ad7d5ccc069e0fedf26a85f755e06cd76590463e1e04e17363e35d9832 \
  artifacts/canonical/sudoku-easy-qualified/manifest.json | sha256sum -c -
```

The exporter worktree intentionally contains the historical source file restored from a later
commit; its contents and hash are part of provenance. Inspect its status before removing it. The
remaining setup guide does not yet specify all artifact exporters or how to reproduce their hashes.

### Linux TinyStories development replay

This offline export uses the same build-tree Python 3.14.9 and Linux reference venv as the
Sudoku-easy replay. Download `tinystories` using the checkpoint downloader and the budget above;
verify the source before export. This command expects the converted `tinystories` tokenizer from
the tokenizer section and a full Git history. It reconstructs the historical exporter in a
separate worktree and checks the qualified manifest, without changing the current exporter.
The export commands were replayed using retained source and tokenizer assets, **not** a fresh
clone or a fresh download; see the [replay evidence](results/2026-11-19-phase5-cli/linux-export-repro-v1.md#tinystories-guide-command-replay).

```sh
.reference/venv/bin/python tools/reference/fetch_checkpoints.py \
  --profile tinystories --output artifacts/upstream/tinystories --verify || exit 1
test ! -e reports/tiny-exporter-worktree && test ! -L reports/tiny-exporter-worktree || exit 1
git worktree add --detach reports/tiny-exporter-worktree fdc7db4b6d0316bfef74085994b26a6619cd6847 || exit 1
test ! -e reports/tiny-exporter-worktree/tools/reference/tinystories_artifact.py || exit 1
git show 93ae5eb6d51d43a02ae764b2dc928c57d9bd7335:tools/reference/tinystories_artifact.py \
  > reports/tiny-exporter-worktree/tools/reference/tinystories_artifact.py || exit 1
printf '%s  %s\n' f757c3cb694922c2f8cf96a3f6cb40c47ea6bfa8dae262de3b957ee5e204754d \
  reports/tiny-exporter-worktree/tools/reference/tinystories_artifact.py | sha256sum -c - || exit 1
root="$PWD"
(
  cd reports/tiny-exporter-worktree || exit 1
  PYTHONPATH="$root/.reference/venv/lib/python3.14/site-packages" \
    "$root/.reference/Python-3.14.9/python" tools/reference/tinystories_artifact.py \
    --source "$root/artifacts/upstream/tinystories" \
    --tokenizer-asset "$root/artifacts/canonical/tokenizers-qualified/tinystories" \
    --output "$root/artifacts/canonical/tinystories-qualified" \
    --report "$root/reports/tinystories-export.json"
) || exit 1
printf '%s  %s\n' 0337e4d2ab2e52d6808123847ad945749276a616d5865e3db5f9861f4ff48db1 \
  artifacts/canonical/tinystories-qualified/manifest.json | sha256sum -c -
```

The worktree deliberately contains the restored untracked exporter. Inspect its status and contents
before removing it; do not overwrite or discard unrelated work. This is an export identity check,
not a text-generation or quality gate.

### Synthetic Sudoku-easy request

For the Linux CUDA build above, the following creates a new all-blank synthetic board, compares the
unified and legacy commands and validates output shapes. It is an operator/inference smoke request,
not a Sudoku quality or performance benchmark:

```sh
python3 - <<'PY' || exit 1
import json
from pathlib import Path
row = [11]
for i in range(9):
    if i:
        row.append(10)
    row.extend([0] * 9)
row.append(11)
assert len(row) == 91
with Path('reports/synthetic-sudoku-input.json').open('x') as stream:
    json.dump([row], stream, separators=(',', ':'))
    stream.write('\n')
PY
cudnn="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib"
test ! -e reports/synthetic-sudoku-sample.json && test ! -L reports/synthetic-sudoku-sample.json || exit 1
test ! -e reports/synthetic-sudoku-alias.json && test ! -L reports/synthetic-sudoku-alias.json || exit 1
LD_LIBRARY_PATH="$cudnn" target/release/ncm sample artifacts/canonical/sudoku-easy-qualified cuda \
  reports/synthetic-sudoku-input.json > reports/synthetic-sudoku-sample.json || exit 1
LD_LIBRARY_PATH="$cudnn" target/release/ncm generate artifacts/canonical/sudoku-easy-qualified cuda \
  reports/synthetic-sudoku-input.json > reports/synthetic-sudoku-alias.json || exit 1
cmp reports/synthetic-sudoku-sample.json reports/synthetic-sudoku-alias.json || exit 1
python3 - <<'PY'
import json
with open('reports/synthetic-sudoku-sample.json') as stream:
    result = json.load(stream)
assert set(result) == {'codes', 'ids'}
assert len(result['codes']) == len(result['ids']) == 1
assert len(result['codes'][0]) == 322 and len(result['ids'][0]) == 89
PY
```

## Other inference examples — pending

Run `target/release/ncm --help` and `target/release/ncm sample --help` for the current interface.
Additional reproducible examples, including the LM1B rotary sidecar, must be tested from fresh
clones on both machines before release. The CUDA-only
[CLI equivalence screen](results/2026-11-19-phase5-cli/cuda-routing-screen-v2.md) and
[text-input screen](results/2026-11-19-phase5-cli/cuda-text-input-v1.md) describe the current
limited development checks, not the release gate.
