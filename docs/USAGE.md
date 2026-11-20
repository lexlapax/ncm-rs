# Inference setup — working draft

**Not a verified release recipe.** Phase 5 is still in progress. The commands below document
current development inputs and builds; they have not been followed end-to-end from a fresh clone
on Linux or macOS. In particular, the approved Sudoku medium/hard provenance-only re-exports
and LM1B text-input behavior still need full qualification. Do not treat this document as the v0.1.0 setup gate.
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
A [Linux fresh-clone slice](results/2026-11-19-phase5-cli/linux-fresh-clone-tokenizers-v1.md)
replayed this section and checked all four converted tokenizer identities; it is not a complete
guide replay or a Metal gate. Check those identities against the tracked lock:

```sh
python3 - <<'PY'
import hashlib,json
from pathlib import Path
lock=json.loads(Path('tools/reference/tokenizers-v1.lock.json').read_text())
assert lock['schema_version']==1 and set(lock['profiles'])=={'bert','gpt2','qwen3','tinystories'}
for profile,expected in sorted(lock['profiles'].items()):
    for name,key in (('manifest.json','manifest_sha256'),('tokenizer.json','payload_sha256')):
        path=Path('artifacts/canonical/tokenizers-qualified')/profile/name
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=expected[key]:
            raise ValueError(f'{path}: {actual} != {expected[key]}')
        print(f'{actual}  {path}')
PY
```

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

For the six supported text profiles, this loop applies the exact budgets above to separate fresh
destinations. It downloads approximately 8.7 GB in total, including the three shared-tokenizer
copies; plan disk space for both downloads and exports. Partial directories are retained on failure.

```sh
for entry in tinystories:578228368 xsum:653983635 wmt14:1126375067 lm1b-small:987981523 lm1b-medium:1865857368 lm1b-large:3499027351; do
  profile=${entry%:*}; budget=${entry#*:}
  .reference/venv/bin/python tools/reference/fetch_checkpoints.py --profile "$profile" --output "artifacts/upstream/$profile" --budget-bytes "$budget" || exit 1
done
```

Each LM1B size download includes a copy of the shared tokenizer checkpoint; plan disk space
accordingly. The exporters recheck the source sizes and SHA-256 before opening checkpoints using
`torch.load(weights_only=True)`; they never retry unrestricted pickle. The downloader was screened
against retained inputs for all profiles; the [integrated Linux replay](results/2026-11-19-phase5-cli/linux-guide-integrated-replay-v1.md)
freshly downloaded the seven profiles with reproducible manifests. A complete fresh-clone workflow
on both machines remains unverified. The earlier [download screen](results/2026-11-19-phase5-cli/checkpoint-download-screen-v1.md)
records its narrower scope.

## Export and manifest identities — incomplete

Exporters and all output paths are explicit and refuse existing artifacts. XSum v2 additionally
needs the pinned offline rotary reference; LM1B needs the hash-pinned sidecar outside each artifact.
A local investigation reproduced seven qualified manifests, using different historical exporter
commits and Python patch versions; Sudoku medium/hard's historical exporter has not been located.
The exact identities, provenance-only differences and failed Python 3.14.9 `_ssl` installation are
recorded in the [investigation](results/2026-11-19-phase5-cli/linux-export-repro-v1.md).

**Do not silently substitute a new manifest hash for a qualified one.** The user approved the
provenance-only medium/hard candidates below for further qualification; the original identities
remain recorded. Complete review, a portable reference environment and a full clean-clone replay
are still required.
The limited Sudoku-easy Linux replay below does not close those requirements. Existing [Sudoku export](../tools/reference/README.md#sudoku-easy-checkpoint-and-oracle-checks),
[XSum rotary export](results/2026-11-15-phase4-xsum/rotary-artifact-profile-v2.md),
[WMT14 export](results/2026-11-16-phase4-wmt14/artifact-plan-v1.md) and
[LM1B export](results/2026-11-16-phase4-lm1b/medium-large-artifacts-v1.md) reports are historical
records, not a substitute for the missing release recipe.

### Linux Sudoku-easy development replay

The build, CUDA check, download, export and synthetic example blocks were replayed as written in a
[fresh-clone slice](results/2026-11-19-phase5-cli/linux-guide-sudoku-replay-v1.md) and in the
[integrated Linux replay](results/2026-11-19-phase5-cli/linux-guide-integrated-replay-v1.md); neither
is a complete guide replay. It reproduces the qualified easy manifest, not the unresolved medium/hard
artifacts. The recorded Linux reference uses Python 3.14.9 with Torch 2.15.0+cu132, the historical Git base and the exact exporter bytes
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

### Linux Sudoku medium/hard development replay

The user approved the following provenance-only candidates for further qualification. The original
medium/hard exporter is missing; the only provenance field changed is `converter_sha256`, and
its file record consequently differs in the manifest. All other payload bytes and manifest fields
matched the retained originals in the [investigation](results/2026-11-19-phase5-cli/linux-export-repro-v1.md).
Their CUDA two-row [screen](results/2026-11-19-phase5-cli/cuda-sudoku-reexports-v1.md) is not full
qualification. The block below passed with fresh checkpoint downloads in the existing Linux checkout
([evidence](results/2026-11-19-phase5-cli/linux-export-repro-v1.md#mediumhard-draft-guide-replay-with-fresh-downloads));
it has not yet been followed from a fresh clone or on macOS. It uses the same Python 3.14.9 build
and reference environment prepared in the easy section above.

```sh
for entry in medium:162378394 hard:162378392; do
  difficulty=${entry%:*}; budget=${entry#*:}
  .reference/venv/bin/python tools/reference/fetch_checkpoints.py --profile "sudoku-$difficulty" \
    --output "artifacts/upstream/sudoku-$difficulty" --budget-bytes "$budget" || exit 1
done

test ! -e reports/sudoku-reexport-worktree && test ! -L reports/sudoku-reexport-worktree || exit 1
git worktree add --detach reports/sudoku-reexport-worktree e1ffb121a8094505173c03d27669b6486f9b9abd || exit 1
test ! -L reports/sudoku-reexport-worktree/tools/reference/sudoku_artifact.py || exit 1
printf '%s  %s\n' cfcae7d716fa7a9692c57019454bd28da5f8098c6d3bec32698795076dc0034e \
  reports/sudoku-reexport-worktree/tools/reference/sudoku_artifact.py | sha256sum -c - || exit 1
git show cb0162cea7cae00060bb8f06057800e8641cd90a:tools/reference/sudoku_artifact.py \
  > reports/sudoku-reexport-worktree/tools/reference/sudoku_artifact.py || exit 1
printf '%s  %s\n' e639675a452da41ad4f89c43db094b560a7d71250219ce44774559d19501aade \
  reports/sudoku-reexport-worktree/tools/reference/sudoku_artifact.py | sha256sum -c - || exit 1
root="$PWD"
for difficulty in medium hard; do
  (
    cd reports/sudoku-reexport-worktree || exit 1
    PYTHONPATH="$root/.reference/venv/lib/python3.14/site-packages" \
      "$root/.reference/Python-3.14.9/python" tools/reference/sudoku_artifact.py \
      --difficulty "$difficulty" --source "$root/artifacts/upstream/sudoku-$difficulty" \
      --output "$root/artifacts/canonical/sudoku-$difficulty-reexport" \
      --report "$root/reports/sudoku-$difficulty-reexport.json"
  ) || exit 1
done
printf '%s  %s\n' 9bc8e30673e426d6b10ef14204e2b0a0162fdc6a38c92e453fe5c156fab375c4 \
  artifacts/canonical/sudoku-medium-reexport/manifest.json | sha256sum -c - || exit 1
printf '%s  %s\n' 25ec837745601d69435a41f7ef5131b70d5fe17e1ca908e7b660d15531031158 \
  artifacts/canonical/sudoku-hard-reexport/manifest.json | sha256sum -c - || exit 1
```

The exporter worktree intentionally has one tracked file replaced with the pinned exporter; inspect
its status and contents before removing it. The original qualified manifest hashes remain in the
[investigation](results/2026-11-19-phase5-cli/linux-export-repro-v1.md).

### Linux TinyStories development replay

This offline export uses the same build-tree Python 3.14.9 and Linux reference venv as the
Sudoku-easy replay. Download `tinystories` using the checkpoint downloader and the budget above;
verify the source before export. This command expects the converted `tinystories` tokenizer from
the tokenizer section and a full Git history. It reconstructs the historical exporter in a
separate worktree and checks the qualified manifest, without changing the current exporter.
The export commands passed with retained inputs and in the [integrated Linux fresh-clone replay](results/2026-11-19-phase5-cli/linux-guide-integrated-replay-v1.md).

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

### Linux XSum development replay

This offline export uses the same build-tree Python 3.14.9 and Linux reference venv as the
Sudoku-easy replay. Download `xsum` with the checkpoint downloader and budget above, and prepare
the converted `gpt2` tokenizer as described earlier. The rotary reference is generated by the
current reference environment from the pinned upstream checkout; its exact payload hash is
required. The XSum exporter is reconstructed at its recorded base commit and its qualified
manifest is checked. This sequence passed with retained inputs and in the
[integrated Linux fresh-clone replay](results/2026-11-19-phase5-cli/linux-guide-integrated-replay-v1.md).

```sh
.reference/venv/bin/python tools/reference/fetch_checkpoints.py \
  --profile xsum --output artifacts/upstream/xsum --verify || exit 1
.reference/venv/bin/python tools/reference/xsum_rope_reference.py \
  --upstream .reference/upstream-ncm --output reports/xsum-rope-reference || exit 1
printf '%s  %s\n' 5d205c1c42b467c00e10d03739e18b6433185bcc4c4f9b066d7c6be69ba8346a \
  reports/xsum-rope-reference/rope.safetensors | sha256sum -c - || exit 1
test ! -e reports/xsum-exporter-worktree && test ! -L reports/xsum-exporter-worktree || exit 1
git worktree add --detach reports/xsum-exporter-worktree 3bdb96b2981145ea71a387b10e662318af85cdc3 || exit 1
printf '%s  %s\n' 99173aa7b83907031fc69a15d23e9546f4a9b3de3a876e301c58d7e7a13b51d8 \
  reports/xsum-exporter-worktree/tools/reference/xsum_artifact.py | sha256sum -c - || exit 1
git -C reports/xsum-exporter-worktree diff --quiet -- tools/reference/xsum_artifact.py || exit 1
git show a92a27516957ae9f3e2e8e52e8d6bf189a600dfc:tools/reference/xsum_artifact.py \
  > reports/xsum-exporter-worktree/tools/reference/xsum_artifact.py || exit 1
printf '%s  %s\n' eb90493398ca55087b775a9207200e11d6f312a8e0cd154284569f64cc244319 \
  reports/xsum-exporter-worktree/tools/reference/xsum_artifact.py | sha256sum -c - || exit 1
root="$PWD"
(
  cd reports/xsum-exporter-worktree || exit 1
  PYTHONPATH="$root/.reference/venv/lib/python3.14/site-packages" \
    "$root/.reference/Python-3.14.9/python" tools/reference/xsum_artifact.py \
    --source "$root/artifacts/upstream/xsum" \
    --tokenizer-asset "$root/artifacts/canonical/tokenizers-qualified/gpt2" \
    --rope-reference "$root/reports/xsum-rope-reference" \
    --output "$root/artifacts/canonical/xsum-qualified" \
    --report "$root/reports/xsum-export.json"
) || exit 1
printf '%s  %s\n' 8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4 \
  artifacts/canonical/xsum-qualified/manifest.json | sha256sum -c -
```

Unlike the TinyStories worktree, the XSum base already contains an older tracked exporter: the
recipe checks its pristine hash before replacing it in the new worktree. Inspect status and
contents before removing that worktree; do not discard unrelated changes. This is an export
identity check, not a text-generation or quality gate.

### Linux WMT14 and LM1B development replays

These exports use the Linux reference venv with **Python 3.14.10 and Torch 2.15.0+cu132**;
these versions are recorded in provenance, so other versions can change the manifest. Download
`wmt14` and each needed `lm1b-SIZE` separately with the budgets above. Each LM1B download has
its own tokenizer checkpoint. Prepare the converted `qwen3` and `bert` tokenizers first. These
commands passed with retained inputs and in the [integrated Linux fresh-clone replay](results/2026-11-19-phase5-cli/linux-guide-integrated-replay-v1.md).
They check artifact identities only; the rotary sidecar is prepared separately below.

```sh
.reference/venv/bin/python -c 'import platform,torch; assert platform.python_version()=="3.14.10" and torch.__version__=="2.15.0+cu132"' || exit 1
.reference/venv/bin/python tools/reference/fetch_checkpoints.py \
  --profile wmt14 --output artifacts/upstream/wmt14 --verify || exit 1
setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  .reference/venv/bin/python tools/reference/wmt14_runtime_artifact.py \
  --source artifacts/upstream/wmt14 --tokenizer-asset artifacts/canonical/tokenizers-qualified/qwen3 \
  --output artifacts/canonical/wmt14-qualified || exit 1
printf '%s  %s\n' 01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81 \
  artifacts/canonical/wmt14-qualified/manifest.json | sha256sum -c - || exit 1

for size in medium large; do
  .reference/venv/bin/python tools/reference/fetch_checkpoints.py \
    --profile "lm1b-$size" --output "artifacts/upstream/lm1b-$size" --verify || exit 1
  setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
    .reference/venv/bin/python tools/reference/lm1b_small_runtime_artifact.py \
    --size "$size" --source "artifacts/upstream/lm1b-$size" \
    --tokenizer-asset artifacts/canonical/tokenizers-qualified/bert \
    --output "artifacts/canonical/lm1b-$size-qualified" || exit 1
done
printf '%s  %s\n%s  %s\n' \
  2fea4a252de0cecbe6ab12b2c4a0077cff66498e6344894694ba02876a38c30c artifacts/canonical/lm1b-medium-qualified/manifest.json \
  86676fed0f9813b3ec293aca87133eca61fc609d10d620978375d12c0b03698a artifacts/canonical/lm1b-large-qualified/manifest.json | sha256sum -c - || exit 1
```

The small qualified manifest records an earlier exporter, so use its historical worktree instead
of substituting the generalized exporter. Its exporter requires output under that worktree's own
`artifacts/canonical`; the command moves the checked result to the main artifact directory.
Inspect the worktree, including ignored files, before removing it; preserve unrelated work.

```sh
.reference/venv/bin/python tools/reference/fetch_checkpoints.py \
  --profile lm1b-small --output artifacts/upstream/lm1b-small --verify || exit 1
test ! -e reports/lm1b-small-exporter-worktree && test ! -L reports/lm1b-small-exporter-worktree || exit 1
git worktree add --detach reports/lm1b-small-exporter-worktree 9d791f88c75614bf531d897603d1896394145a79 || exit 1
printf '%s  %s\n' a8dd547b02b06a557c581c885e4eea289a5a8ffa60a6682556737409c06caf52 \
  reports/lm1b-small-exporter-worktree/tools/reference/lm1b_small_runtime_artifact.py | sha256sum -c - || exit 1
root="$PWD"
(
  cd reports/lm1b-small-exporter-worktree || exit 1
  setpriv --inh-caps=-all env OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
    "$root/.reference/venv/bin/python" tools/reference/lm1b_small_runtime_artifact.py \
    --source "$root/artifacts/upstream/lm1b-small" \
    --tokenizer-asset "$root/artifacts/canonical/tokenizers-qualified/bert" \
    --output artifacts/canonical/lm1b-small-qualified
) || exit 1
printf '%s  %s\n' 58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76 \
  reports/lm1b-small-exporter-worktree/artifacts/canonical/lm1b-small-qualified/manifest.json | sha256sum -c - || exit 1
test ! -e artifacts/canonical/lm1b-small-qualified && test ! -L artifacts/canonical/lm1b-small-qualified || exit 1
mv -T -n reports/lm1b-small-exporter-worktree/artifacts/canonical/lm1b-small-qualified \
  artifacts/canonical/lm1b-small-qualified || exit 1
test ! -e reports/lm1b-small-exporter-worktree/artifacts/canonical/lm1b-small-qualified || exit 1
printf '%s  %s\n' 58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76 \
  artifacts/canonical/lm1b-small-qualified/manifest.json | sha256sum -c -
```

For all three LM1B artifacts in `artifacts/canonical`, prepare the shared pinned sidecar in that
same directory. The default CLI looks beside the artifact directory; the alias still accepts an
explicit sidecar path. Check that the output is absent, including a dangling symlink, and use a
trusted, stable directory. The hash must match exactly; do not substitute a numerical approximation.

```sh
test ! -e artifacts/canonical/lm1b-reference-rope-v1.safetensors && test ! -L artifacts/canonical/lm1b-reference-rope-v1.safetensors || exit 1
setpriv --inh-caps=-all .reference/venv/bin/python tools/reference/lm1b_reference_rope_diagnostic.py \
  --upstream .reference/upstream-ncm \
  --output artifacts/canonical/lm1b-reference-rope-v1.safetensors || exit 1
printf '%s  %s\n' 9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc \
  artifacts/canonical/lm1b-reference-rope-v1.safetensors | sha256sum -c -
```

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

## Synthetic text examples — CUDA development screen

The examples below assume the Linux CUDA build and the four qualified artifacts and shared LM1B
sidecar prepared above. They create fresh synthetic inputs and one-row outputs; they are not
quality, performance or Python-parity tests. TinyStories text must encode to exactly one non-padding
token; XSum/WMT14 text is truncated and padded per profile. LM1B currently uses the fixed `[[101]]`
ID input; its `--text` policy is unresolved. WMT14 uses `cuda`, while the other examples use
`cuda-cudnn`. A [Linux CUDA screen](results/2026-11-19-phase5-cli/cuda-synthetic-text-examples-v1.md)
used the same requests with retained artifact paths; the [integrated Linux fresh-clone replay](results/2026-11-19-phase5-cli/linux-guide-integrated-replay-v1.md)
also ran these examples. Full-guide and Metal replays remain pending.

```sh
mkdir reports/text-examples || exit 1
python3 - <<'PY' || exit 1
import json
from pathlib import Path
root=Path('reports/text-examples')
for name,value in {'tiny-input':['hello'],
                   'xsum-input':['The council opened a new park on Tuesday.'],
                   'wmt14-input':['Das ist ein kleines Beispiel.'],
                   'lm1b-input':[[101]]}.items():
    with (root/f'{name}.json').open('x') as stream:
        json.dump(value,stream,ensure_ascii=False)
        stream.write('\n')
PY
cudnn="$PWD/.reference/venv/lib/python3.14/site-packages/nvidia/cudnn/lib"
(
  set -eC # refuse to overwrite output files
  export LD_LIBRARY_PATH="$cudnn" OMP_NUM_THREADS=1
  cli=target/release/ncm
  dir=reports/text-examples
  "$cli" sample artifacts/canonical/tinystories-qualified cuda-cudnn "$dir/tiny-input.json" --text --seed 42 --decode > "$dir/tiny-output.json"
  "$cli" sample artifacts/canonical/xsum-qualified cuda-cudnn "$dir/xsum-input.json" --text --decode > "$dir/xsum-output.json"
  "$cli" sample artifacts/canonical/wmt14-qualified cuda "$dir/wmt14-input.json" --text --decode > "$dir/wmt14-output.json"
  "$cli" sample artifacts/canonical/lm1b-small-qualified cuda-cudnn "$dir/lm1b-input.json" --seed 42 --decode > "$dir/lm1b-output.json"
) || exit 1
python3 - <<'PY'
import json
from pathlib import Path
for name in ('tiny','xsum','wmt14','lm1b'):
    result=json.loads((Path('reports/text-examples')/f'{name}-output.json').read_text())
    if not all(isinstance(result.get(key),list) and len(result[key])==1 for key in ('codes','ids','texts')):
        raise ValueError(f'{name}: unexpected output shape')
    if not isinstance(result['texts'][0],str):
        raise ValueError(f'{name}: decoded output is not text')
    print(name, 'one decoded row')
PY
```

Run `target/release/ncm --help` and `target/release/ncm sample --help` for the complete current interface. These
examples must still be tested from a fresh clone on Mac before release. The CUDA-only
[CLI equivalence screen](results/2026-11-19-phase5-cli/cuda-routing-screen-v2.md) and
[text-input screen](results/2026-11-19-phase5-cli/cuda-text-input-v1.md) describe separate limited
checks, not the release gate.
