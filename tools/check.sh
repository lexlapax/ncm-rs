#!/usr/bin/env bash
# Local validation only. This script never downloads model weights or starts SSH jobs.
set -euo pipefail

usage() {
    printf '%s\n' \
        'Usage: tools/check.sh [--cuda|--mlx]' \
        '' \
        'Default: formatting, Clippy, CPU fixtures, docs and Python report-tool tests.' \
        '--cuda: additionally lint/test CUDA code and run the real-device operator probe.' \
        '--mlx: additionally lint/test/probe MLX on Apple Silicon (native MLX required).'  \
        '' \
        'Requires Rust 1.99.1+, rustfmt, Clippy, and Python 3 (no Torch for these checks).' \
        'CUDA requires Linux, a compatible NVIDIA driver/toolkit and host C++ compiler.' \
        'Set NVCC_CCBIN, CUDA_COMPUTE_CAP and CARGO_BUILD_JOBS for your machine.' \
        'MLX native build/link environment is documented in tools/reference/README.md.'
}

cuda=false
mlx=false
case "${1:-}" in
    '') ;;
    --cuda) cuda=true ;;
    --mlx) mlx=true ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    usage >&2
    exit 2
fi
if "$cuda" && [[ "$(uname -s)" != Linux ]]; then
    printf '%s\n' '--cuda currently supports Linux only; no fallback.' >&2
    exit 2
fi

if "$mlx" && { [[ "$(uname -s)" != Darwin ]] || [[ "$(uname -m)" != arm64 ]]; }; then
    printf '%s\n' '--mlx requires Apple Silicon macOS; no fallback.' >&2
    exit 2
fi

root=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
# Report-tool tests create temporary files here, including in a fresh checkout.
mkdir -p reports
for tool in cargo rustc python3; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'Missing required tool: %s\n' "$tool" >&2
        exit 1
    fi
done

printf '%s\n' '== Local validation =='
rustc --version
cargo --version
python3 --version
bash -n tools/check.sh tools/setup-mlx-native.sh tools/package.sh tools/install.sh
if command -v shellcheck >/dev/null 2>&1; then
    shellcheck tools/check.sh tools/setup-mlx-native.sh tools/package.sh tools/install.sh
fi
cargo fmt --all -- --check
cargo check --workspace --locked
cargo clippy --workspace --all-targets --features ncm-backend-cuda/cpu,ncm-train/cpu --locked -- -D warnings
cargo test --workspace --features ncm-backend-cuda/cpu,ncm-train/cpu --locked
RUSTDOCFLAGS="${RUSTDOCFLAGS:-} -D warnings" \
    cargo doc --workspace --features ncm-backend-cuda/cpu,ncm-train/cpu --no-deps --locked
if [[ "$(uname -s)" == Darwin ]]; then
    python3 tools/bench/run_macos_report_tests.py
elif [[ "$(uname -s)" == Linux ]]; then
    python3 tools/bench/run_linux_report_tests.py
else
    python3 -m unittest discover -s tools/bench -p 'test_*.py' -v
fi
python3 -m unittest discover -s tools/reference -p "test_text_phase4_data.py" -v
python3 -m unittest discover -s tools/reference -p "test_compare_artifact_provenance.py" -v
python3 -m unittest discover -s tools/reference -p "test_fetch_checkpoints.py" -v
python3 -m unittest discover -s tools/reference -p "test_phase5_tinystories_text_reference.py" -v
python3 -m unittest discover -s tools/reference -p "test_lm1b_reference_rope_diagnostic.py" -v
python3 -m unittest discover -s tools/reference -p "test_phase5_text_boundaries.py" -v
python3 -m unittest discover -s tools/reference -p "test_phase5_decode_boundaries.py" -v
python3 -m unittest discover -s tools/reference -p "test_phase5_lm1b_retained_decode.py" -v
python3 -m unittest discover -s tools/reference -p "test_phase5_tinystories_decode_reference.py" -v
python3 -m unittest discover -s tools/reference -p "test_export_local_ncm.py" -v
python3 -m unittest discover -s tools/reference -p "test_quickstart_export.py" -v
python3 -m unittest discover -s tools/reference/lm1b_campaign -p "test_campaign_retest.py" -v
python3 -m unittest discover -s tools -p "test_render_homebrew_formula.py" -v
python3 -m py_compile tools/reference/*.py tools/bench/compare.py tools/render_homebrew_formula.py

if "$cuda"; then
    printf '%s\n' '== Explicit CUDA validation (real GPU required) =='
    cargo clippy --workspace --all-targets --features ncm-backend-cuda/cuda --locked -- -D warnings
    cargo test -p ncm-backend-cuda --features cuda --release --locked
    report_dir="reports/local-check-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    mkdir -p "$report_dir"
    cargo run -p ncm-backend-cuda --features cuda --example probe --release --locked -- \
        cuda tests/fixtures/operators-v1.json "$report_dir/cuda.json"
    printf 'CUDA smoke report: %s/cuda.json\n' "$report_dir"
fi

if "$mlx"; then
    printf '%s\n' '== Explicit MLX validation (real Metal GPU required) =='
    cargo clippy -p ncm-backend-mlx --all-targets --features mlx --release --locked -- -D warnings
    cargo test -p ncm-backend-mlx --features mlx --release --locked
    RUSTDOCFLAGS="${RUSTDOCFLAGS:-} -D warnings" cargo doc -p ncm-backend-mlx --features mlx --no-deps --locked
    report_dir="reports/local-mlx-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    mkdir -p "$report_dir"
    cargo run -p ncm-backend-mlx --features mlx --example mlx_probe --release --locked -- \
        tests/fixtures/operators-v1.json "$report_dir/mlx.json"
fi

printf '%s\n' 'Requested checks passed. No model-quality or unrequested GPU validation is implied.'
