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

## Checkpoint download layout

The exporters validate exact file sizes and SHA-256 before opening checkpoints using
`torch.load(weights_only=True)`; they do not retry unrestricted pickle. Download only the profiles
you need, from the immutable checkpoint revision above. The following shell function **refuses
existing destination files**, creates parent directories and downloads a named path. A failed
or interrupted download may leave a partial file that must be inspected and removed before retry.
It does not replace the exporters' mandatory hash checks.

```sh
revision=d342ce8aaea58830eef3b8ad473664673e41f692
fetch_checkpoint() {
  remote=$1 destination=$2
  if test -e "$destination" || test -L "$destination"; then
    printf 'Refusing existing destination: %s\n' "$destination" >&2; return 1
  fi
  mkdir -p "$(dirname "$destination")" || return 1
  curl --fail --location --output "$destination" \
    "https://huggingface.co/hazyresearch/ncm-checkpoints/resolve/$revision/$remote"
}
# Choose the desired difficulties/profiles; weights can require multiple GiB.
for difficulty in easy medium hard; do
  for file in config.yaml tokenizer.pt generator.pt; do
    fetch_checkpoint "sudoku/$difficulty/$file" "artifacts/upstream/sudoku/$difficulty/$file" || exit 1
  done
done
for profile in tinystories xsum wmt14; do
  for file in config.yaml tokenizer.pt generator.pt; do
    fetch_checkpoint "$profile/$file" "artifacts/upstream/$profile-phase4/$file" || exit 1
  done
done
for file in bpe_vocab_8192.pickle bpe_merges_8192.pickle; do
  fetch_checkpoint "tinystories/$file" "artifacts/upstream/tinystories-phase4/$file" || exit 1
done
fetch_checkpoint lm1b/tokenizer.pt artifacts/upstream/lm1b-phase4/tokenizer.pt || exit 1
for size in small medium large; do
  for file in config.yaml generator.pt; do
    fetch_checkpoint "lm1b/$size/$file" "artifacts/upstream/lm1b-phase4/$size/$file" || exit 1
  done
done
```

The pinned file sizes and hashes live in `tools/reference/sudoku_artifact.py`,
`tinystories_artifact.py`, `xsum_artifact.py`, `wmt14_inventory.py`, `lm1b_inventory.py` and
`lm1b_export_preparation.py`. The complete commands above have **not** been exercised as a fresh
download in this Phase 5 investigation. A limited URL screen fetched and hash-verified the nine small profile config files
and two TinyStories BPE inputs; no checkpoint weights were downloaded in that screen. A bounded,
verified download workflow and full source/disk-budget checks remain to be completed before release.

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
