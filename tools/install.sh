#!/usr/bin/env bash
# Install ncm from a local release tarball or from the current repository checkout.
set -euo pipefail

usage() {
    printf '%s\n' \
        'Usage: tools/install.sh [OPTIONS]' \
        '' \
        'Options:' \
        '  --archive PATH   Install from a release .tar.gz archive (verifies sibling .sha256 if present)' \
        '  --prefix DIR     Installation prefix (default: ~/.local; installs bin/ncm and lib/*)' \
        '  --skip-doctor    Do not run ncm doctor after installing' \
        '  -h, --help       Show this help'
}

archive=""
prefix="${HOME}/.local"
run_doctor=true

while (( $# > 0 )); do
    case "$1" in
        --archive)
            shift
            if (( $# == 0 )); then usage >&2; exit 2; fi
            archive="$1"; shift
            ;;
        --prefix)
            shift
            if (( $# == 0 )); then usage >&2; exit 2; fi
            prefix="$1"; shift
            ;;
        --skip-doctor) run_doctor=false; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

verify_sha256() {
    local file="$1"
    local sha_file="${file}.sha256"
    if [[ ! -f "$sha_file" ]]; then
        return 0
    fi
    printf 'Verifying %s ...\n' "$sha_file"
    if command -v sha256sum >/dev/null 2>&1; then
        (cd "$(dirname -- "$file")" && sha256sum -c "$(basename -- "$sha_file")")
    elif command -v shasum >/dev/null 2>&1; then
        (cd "$(dirname -- "$file")" && shasum -a 256 -c "$(basename -- "$sha_file")")
    else
        printf 'Warning: neither sha256sum nor shasum found; skipping checksum check\n' >&2
    fi
}

mkdir -p "$prefix/bin"

if [[ -n "$archive" ]]; then
    if [[ ! -f "$archive" ]]; then
        printf 'Archive not found: %s\n' "$archive" >&2
        exit 1
    fi
    verify_sha256 "$archive"
    tmp=$(mktemp -d "${TMPDIR:-/tmp}/ncm-install-XXXXXX")
    trap 'rm -rf "$tmp"' EXIT
    tar -C "$tmp" -xzf "$archive"
    bundle_dir=$(find "$tmp" -mindepth 1 -maxdepth 1 -type d | head -n 1)
    if [[ -z "$bundle_dir" || ! -x "$bundle_dir/bin/ncm" ]]; then
        printf 'Archive does not contain expected <bundle>/bin/ncm layout\n' >&2
        exit 1
    fi
    cp "$bundle_dir/bin/ncm" "$prefix/bin/ncm"
    chmod 0755 "$prefix/bin/ncm"
    if [[ -d "$bundle_dir/lib" ]]; then
        mkdir -p "$prefix/lib"
        cp -R "$bundle_dir/lib/." "$prefix/lib/"
    fi
else
    root=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
    bin_src="$root/target/release/ncm"
    if [[ ! -x "$bin_src" ]]; then
        printf 'No built binary at %s; run tools/package.sh or cargo build --release first.\n' "$bin_src" >&2
        exit 1
    fi
    cp "$bin_src" "$prefix/bin/ncm"
    chmod 0755 "$prefix/bin/ncm"
    if [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]]; then
        mlx_prebuilt="${MLX_SYS_PREBUILT:-$root/.reference/mlx-native}"
        if [[ -f "$mlx_prebuilt/lib/libmlx.dylib" ]]; then
            mkdir -p "$prefix/lib"
            cp "$mlx_prebuilt/lib/libmlx.dylib" "$prefix/lib/libmlx.dylib"
            [[ -f "$mlx_prebuilt/lib/libjaccl.dylib" ]] && cp "$mlx_prebuilt/lib/libjaccl.dylib" "$prefix/lib/libjaccl.dylib"
            [[ -f "$mlx_prebuilt/lib/mlx.metallib" ]] && cp "$mlx_prebuilt/lib/mlx.metallib" "$prefix/lib/mlx.metallib"
        fi
    fi
fi

printf 'Installed %s\n' "$prefix/bin/ncm"
case ":${PATH}:" in
    *":${prefix}/bin:"*) ;;
    *) printf 'Note: add %s/bin to your PATH to invoke ncm directly.\n' "$prefix" ;;
esac

if "$run_doctor"; then
    "$prefix/bin/ncm" doctor || true
fi
