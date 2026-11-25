#!/usr/bin/env bash
# Build and verify a relocatable local release tarball for ncm-rs.
set -euo pipefail

usage() {
    printf '%s\n' \
        'Usage: tools/package.sh [OPTIONS]' \
        '' \
        'Options:' \
        '  --target linux-cuda|linux-cpu|macos-mlx  Target bundle (default: auto-detect host)' \
        '  --version VER                            Version string without v prefix (default: Cargo.toml)' \
        '  --output-dir DIR                         Output directory (default: dist)' \
        '  --skip-build                             Reuse existing target/release/ncm binary' \
        '  --no-verify                              Skip clean-directory extraction self-check' \
        '  -h, --help                               Show this help'
}

root=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

target=""
version=""
output_dir="$root/dist"
skip_build=false
verify=true

while (( $# > 0 )); do
    case "$1" in
        --target)
            shift
            if (( $# == 0 )); then usage >&2; exit 2; fi
            target="$1"; shift
            ;;
        --version)
            shift
            if (( $# == 0 )); then usage >&2; exit 2; fi
            version="$1"; shift
            ;;
        --output-dir)
            shift
            if (( $# == 0 )); then usage >&2; exit 2; fi
            output_dir="$1"; shift
            ;;
        --skip-build) skip_build=true; shift ;;
        --no-verify) verify=false; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

host_os=$(uname -s)
host_arch=$(uname -m)

if [[ -z "$target" ]]; then
    if [[ "$host_os" == Darwin && "$host_arch" == arm64 ]]; then
        target="macos-mlx"
    elif [[ "$host_os" == Linux ]]; then
        target="linux-cuda"
    else
        printf 'Unsupported host for auto-detected packaging: %s/%s\n' "$host_os" "$host_arch" >&2
        exit 2
    fi
fi

if [[ -z "$version" ]]; then
    version=$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$root/Cargo.toml" | head -n 1)
    if [[ -z "$version" ]]; then
        printf 'Failed to read workspace version from Cargo.toml\n' >&2
        exit 1
    fi
fi

sha256_file() {
    local file="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        (cd "$(dirname -- "$file")" && sha256sum "$(basename -- "$file")")
    elif command -v shasum >/dev/null 2>&1; then
        (cd "$(dirname -- "$file")" && shasum -a 256 "$(basename -- "$file")")
    else
        printf 'Neither sha256sum nor shasum found in PATH\n' >&2
        exit 1
    fi
}

case "$target" in
    linux-cuda)
        if [[ "$host_os" != Linux ]]; then
            printf 'Target linux-cuda requires Linux\n' >&2
            exit 2
        fi
        bundle_name="ncm-v${version}-linux-${host_arch}-cuda"
        if ! "$skip_build"; then
            # shellcheck disable=SC2016
            cargo rustc --release --locked -p ncm-cli --features cuda-cudnn --bin ncm -- \
                -C 'link-arg=-Wl,-rpath,$ORIGIN/../lib:/usr/local/cuda/lib64:/opt/cuda/lib64'
        fi
        ;;
    linux-cpu)
        if [[ "$host_os" != Linux ]]; then
            printf 'Target linux-cpu requires Linux\n' >&2
            exit 2
        fi
        bundle_name="ncm-v${version}-linux-${host_arch}-cpu"
        if ! "$skip_build"; then
            cargo build --release --locked -p ncm-cli --features cpu --bin ncm
        fi
        ;;
    macos-mlx)
        if [[ "$host_os" != Darwin || "$host_arch" != arm64 ]]; then
            printf 'Target macos-mlx requires Apple Silicon macOS (arm64)\n' >&2
            exit 2
        fi
        mlx_prebuilt="${MLX_SYS_PREBUILT:-$root/.reference/mlx-native}"
        if [[ ! -f "$mlx_prebuilt/lib/libmlx.dylib" || ! -f "$mlx_prebuilt/lib/mlx.metallib" ]]; then
            printf 'Missing native MLX at %s (run tools/setup-mlx-native.sh first)\n' "$mlx_prebuilt" >&2
            exit 1
        fi
        bundle_name="ncm-v${version}-macos-arm64-mlx"
        if ! "$skip_build"; then
            MLX_SYS_PREBUILT="$mlx_prebuilt" \
                cargo rustc --release --locked -p ncm-cli --features mlx --bin ncm -- \
                -C 'link-arg=-Wl,-rpath,@executable_path/../lib'
        fi
        ;;
    *)
        printf 'Unsupported target: %s\n' "$target" >&2
        exit 2
        ;;
esac

bin_src="$root/target/release/ncm"
if [[ ! -x "$bin_src" ]]; then
    printf 'Missing built binary: %s\n' "$bin_src" >&2
    exit 1
fi

staging_root=$(mktemp -d "${TMPDIR:-/tmp}/ncm-pkg-XXXXXX")
trap 'rm -rf "$staging_root"' EXIT

bundle_dir="$staging_root/$bundle_name"
mkdir -p "$bundle_dir/bin"
cp "$bin_src" "$bundle_dir/bin/ncm"
chmod 0755 "$bundle_dir/bin/ncm"
cp "$root/LICENSE" "$bundle_dir/LICENSE"
cp "$root/NOTICE" "$bundle_dir/NOTICE"

if [[ "$target" == macos-mlx ]]; then
    mkdir -p "$bundle_dir/lib"
    cp "$mlx_prebuilt/lib/libmlx.dylib" "$bundle_dir/lib/libmlx.dylib"
    if [[ -f "$mlx_prebuilt/lib/libjaccl.dylib" ]]; then
        cp "$mlx_prebuilt/lib/libjaccl.dylib" "$bundle_dir/lib/libjaccl.dylib"
    fi
    cp "$mlx_prebuilt/lib/mlx.metallib" "$bundle_dir/lib/mlx.metallib"
    chmod 0755 "$bundle_dir/lib/"*.dylib
    if [[ -f "$mlx_prebuilt/LICENSE-MLX" ]]; then
        cp "$mlx_prebuilt/LICENSE-MLX" "$bundle_dir/LICENSE-MLX"
    fi

    # Strip any absolute build-host LC_RPATH entries and ensure relocatable @executable_path/@loader_path.
    while IFS= read -r rpath; do
        [[ -z "$rpath" ]] && continue
        if [[ "$rpath" != "@executable_path/../lib" ]]; then
            install_name_tool -delete_rpath "$rpath" "$bundle_dir/bin/ncm" || true
        fi
    done < <(otool -l "$bundle_dir/bin/ncm" | awk '/cmd LC_RPATH/{f=1} f && /path /{print $2; f=0}')
    if ! otool -l "$bundle_dir/bin/ncm" | grep -q '@executable_path/../lib'; then
        install_name_tool -add_rpath '@executable_path/../lib' "$bundle_dir/bin/ncm"
    fi

    if ! otool -l "$bundle_dir/lib/libmlx.dylib" | grep -q '@loader_path'; then
        install_name_tool -add_rpath '@loader_path' "$bundle_dir/lib/libmlx.dylib"
    fi

    codesign --force --sign - "$bundle_dir/lib/libmlx.dylib"
    if [[ -f "$bundle_dir/lib/libjaccl.dylib" ]]; then
        codesign --force --sign - "$bundle_dir/lib/libjaccl.dylib"
    fi
    codesign --force --sign - "$bundle_dir/bin/ncm"

    # Fail closed if any build-host /Users/ path remains in linked deps or rpaths.
    if otool -L "$bundle_dir/bin/ncm" "$bundle_dir/lib/"*.dylib | grep -q '/Users/'; then
        printf 'Error: build-host /Users/ path leaked into dylib references!\n' >&2
        exit 1
    fi
    if otool -l "$bundle_dir/bin/ncm" "$bundle_dir/lib/"*.dylib | grep -A2 'LC_RPATH' | grep -q '/Users/'; then
        printf 'Error: build-host /Users/ path leaked into LC_RPATH!\n' >&2
        exit 1
    fi

    cat > "$bundle_dir/README-INSTALL.txt" <<EOF
ncm v${version} (Apple Silicon macOS / Metal MLX)

Self-contained relocatable bundle (includes libmlx.dylib, libjaccl.dylib, mlx.metallib).
No Python, Xcode, Rust, or DYLD_LIBRARY_PATH required.

Quick install (recommended):
  mkdir -p ~/.local/bin ~/.local/lib
  cp bin/ncm ~/.local/bin/
  cp lib/* ~/.local/lib/
  export PATH="\$HOME/.local/bin:\$PATH"

Verify your GPU and start pulling/running models:
  ncm --version
  ncm doctor --self-test
  ncm pull wmt14
  ncm run wmt14 "Guten Morgen!" --precision fp16

Models default to ~/Library/Application Support/ncm/models (override via NCM_MODELS_DIR or --models-dir).
EOF
elif [[ "$target" == linux-cuda ]]; then
    cat > "$bundle_dir/README-INSTALL.txt" <<EOF
ncm v${version} (Linux ${host_arch} / NVIDIA CUDA + cuDNN)

Runtime requirements (not redistributed in this MIT archive):
  - NVIDIA GPU driver (libcuda.so.1)
  - CUDA Toolkit runtime libraries (libcudart, libcublas, libcublasLt, libcurand)
  - cuDNN 9 shared libraries (libcudnn.so.9) on LD_LIBRARY_PATH, system ldconfig,
    or placed inside ../lib relative to bin/ncm

Quick install:
  mkdir -p ~/.local/bin
  cp bin/ncm ~/.local/bin/
  export PATH="\$HOME/.local/bin:\$PATH"

Verify your GPU and start pulling/running models:
  ncm --version
  ncm doctor --self-test
  ncm pull wmt14
  ncm run wmt14 "Guten Morgen!" --precision fp16

Models default to ~/.local/share/ncm/models (override via NCM_MODELS_DIR or --models-dir).
EOF
else
    cat > "$bundle_dir/README-INSTALL.txt" <<EOF
ncm v${version} (Linux ${host_arch} / CPU)

Quick install:
  mkdir -p ~/.local/bin
  cp bin/ncm ~/.local/bin/
  export PATH="\$HOME/.local/bin:\$PATH"

Verify installation:
  ncm --version
  ncm doctor --self-test
EOF
fi

mkdir -p "$output_dir"
archive="$output_dir/${bundle_name}.tar.gz"
checksum="${archive}.sha256"
rm -f "$archive" "$checksum"

tar -C "$staging_root" -czf "$archive" "$bundle_name"
sha256_file "$archive" > "$checksum"

printf 'Packaged: %s\n' "$archive"
printf 'SHA-256 : %s\n' "$(cat "$checksum")"

if "$verify"; then
    verify_dir="$staging_root/verify"
    mkdir -p "$verify_dir"
    tar -C "$verify_dir" -xzf "$archive"
    extracted="$verify_dir/$bundle_name"
    printf 'Verifying extracted bundle in %s ...\n' "$extracted"
    if [[ "$target" == macos-mlx ]]; then
        (cd "$verify_dir" && env -i HOME="$HOME" PATH=/usr/bin:/bin "$extracted/bin/ncm" --version)
        (cd "$verify_dir" && env -i HOME="$HOME" PATH=/usr/bin:/bin "$extracted/bin/ncm" doctor --self-test)
    else
        (cd "$verify_dir" && "$extracted/bin/ncm" --version)
        (cd "$verify_dir" && "$extracted/bin/ncm" doctor --self-test)
    fi
fi
