#!/usr/bin/env bash
# Reproducibly populate .reference/mlx-native on Apple Silicon macOS from the pinned MLX lockfile.
set -euo pipefail

usage() {
    printf '%s\n' \
        'Usage: tools/setup-mlx-native.sh [--force] [--target DIR]' \
        '' \
        'Extracts pinned native MLX headers, libraries (libmlx.dylib, libjaccl.dylib,' \
        'mlx.metallib), and license metadata on Apple Silicon macOS using' \
        'tools/reference/requirements-macos.lock.'
}

force=false
target=""
while (( $# > 0 )); do
    case "$1" in
        --force) force=true; shift ;;
        --target)
            shift
            if (( $# == 0 )); then usage >&2; exit 2; fi
            target="$1"; shift
            ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$(uname -s)" != Darwin ]] || [[ "$(uname -m)" != arm64 ]]; then
    printf '%s\n' 'tools/setup-mlx-native.sh requires Apple Silicon macOS (arm64).' >&2
    exit 2
fi

root=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
target="${target:-$root/.reference/mlx-native}"
venv="$root/.reference/mlx-venv"
lock="$root/tools/reference/requirements-macos.lock"

if [[ ! -f "$lock" ]]; then
    printf 'Missing lockfile: %s\n' "$lock" >&2
    exit 1
fi

if [[ -d "$target" ]]; then
    if "$force"; then
        rm -rf "$target"
    elif [[ -f "$target/lib/libmlx.dylib" && -f "$target/lib/mlx.metallib" ]]; then
        printf 'Native MLX already present at %s (pass --force to recreate).\n' "$target"
        printf '  export MLX_SYS_PREBUILT=%q\n' "$target"
        printf '  export RUSTFLAGS="-C link-arg=-Wl,-rpath,%s/lib"\n' "$target"
        exit 0
    else
        rm -rf "$target"
    fi
fi

mkdir -p "$(dirname -- "$target")"

if command -v uv >/dev/null 2>&1; then
    if [[ ! -x "$venv/bin/python" ]]; then
        uv venv "$venv"
    fi
    uv pip install --python "$venv/bin/python" -r "$lock"
elif command -v python3 >/dev/null 2>&1; then
    if [[ ! -x "$venv/bin/python" ]]; then
        python3 -m venv "$venv"
    fi
    "$venv/bin/python" -m pip install --upgrade pip
    "$venv/bin/python" -m pip install -r "$lock"
else
    printf '%s\n' 'Neither uv nor python3 was found in PATH.' >&2
    exit 1
fi

"$venv/bin/python" - "$target" <<'PY'
import pathlib
import shutil
import sys
import sysconfig

target = pathlib.Path(sys.argv[1])
purelib = pathlib.Path(sysconfig.get_path("purelib"))
source = purelib / "mlx"
if not source.is_dir():
    raise SystemExit(f"installed mlx package not found at {source}")

target.mkdir(parents=True, exist_ok=False)
for name in ("include", "share", "lib"):
    src = source / name
    if src.is_dir():
        shutil.copytree(src, target / name)

for dist_info in sorted(purelib.glob("mlx-*.dist-info")):
    for lic in sorted(dist_info.glob("LICENSE*")) + sorted(dist_info.glob("licenses/LICENSE*")):
        if lic.is_file():
            shutil.copy2(lic, target / "LICENSE-MLX")
            break
    if (target / "LICENSE-MLX").is_file():
        break
PY

for req in "$target/lib/libmlx.dylib" "$target/lib/mlx.metallib"; do
    if [[ ! -f "$req" ]]; then
        printf 'Expected native MLX file missing after extraction: %s\n' "$req" >&2
        exit 1
    fi
done

printf 'Prepared native MLX at %s\n' "$target"
printf 'Next:\n'
printf '  export MLX_SYS_PREBUILT=%q\n' "$target"
printf '  export RUSTFLAGS="-C link-arg=-Wl,-rpath,%s/lib"\n' "$target"
printf '  cargo build --release --locked -p ncm-cli --features mlx --bin ncm\n'
