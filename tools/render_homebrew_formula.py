#!/usr/bin/env python3
"""Render a deterministic Homebrew formula for Apple Silicon macOS ncm-rs release bundles."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9.]+)?$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def render_formula(
    version: str,
    sha256: str,
    url: str | None = None,
    homepage: str = "https://github.com/lexlapax/ncm-rs",
) -> str:
    if not VERSION_RE.match(version):
        raise ValueError(f"invalid version {version!r} (expected semver without leading v)")
    if not SHA256_RE.match(sha256):
        raise ValueError(f"invalid sha256 {sha256!r} (expected 64 lowercase hex characters)")
    download_url = (
        url
        if url is not None
        else f"https://github.com/lexlapax/ncm-rs/releases/download/v{version}/ncm-v{version}-macos-arm64-mlx.tar.gz"
    )
    if not download_url.startswith("https://"):
        raise ValueError(f"download URL must use https:// (got {download_url!r})")

    return (
        "class Ncm < Formula\n"
        '  desc "Rust Neural Compression Models (NCM) runtime, server, and LoRA trainer for Apple Silicon"\n'
        f'  homepage "{homepage}"\n'
        f'  url "{download_url}"\n'
        f'  version "{version}"\n'
        f'  sha256 "{sha256}"\n'
        '  license "MIT"\n'
        "\n"
        "  depends_on :macos\n"
        "  depends_on arch: :arm64\n"
        "\n"
        "  def install\n"
        '    bin.install "bin/ncm"\n'
        '    lib.install Dir["lib/*"]\n'
        '    doc.install "README-INSTALL.txt", "LICENSE", "NOTICE"\n'
        '    doc.install "LICENSE-MLX" if File.exist?("LICENSE-MLX")\n'
        "  end\n"
        "\n"
        "  def caveats\n"
        "    <<~EOS\n"
        "      Models are stored by default in:\n"
        "        ~/Library/Application Support/ncm/models\n"
        "      Override with NCM_MODELS_DIR or --models-dir.\n"
        "\n"
        "      Verify Metal/MLX readiness and pull a starter model:\n"
        "        ncm doctor --self-test\n"
        "        ncm pull wmt14\n"
        '        ncm run wmt14 "Guten Morgen!" --precision fp16\n'
        "    EOS\n"
        "  end\n"
        "\n"
        "  test do\n"
        '    assert_match "ncm #{version}", shell_output("#{bin}/ncm --version")\n'
        '    assert_match "\\"ok\\": true", shell_output("#{bin}/ncm doctor --json")\n'
        "  end\n"
        "end\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Release semver without leading v")
    parser.add_argument("--sha256", required=True, help="64-hex SHA-256 of the macOS arm64 tarball")
    parser.add_argument("--url", default=None, help="Override tarball HTTPS download URL")
    parser.add_argument("--output", default=None, help="Write rendered formula to this path")
    args = parser.parse_args(argv)

    rendered = render_formula(version=args.version, sha256=args.sha256, url=args.url)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
