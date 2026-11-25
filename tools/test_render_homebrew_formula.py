#!/usr/bin/env python3
"""Unit tests for tools/render_homebrew_formula.py."""
import tempfile
import unittest
from pathlib import Path

from render_homebrew_formula import main, render_formula


class RenderHomebrewFormulaTests(unittest.TestCase):
    def test_renders_expected_formula_fields(self) -> None:
        sha = "a" * 64
        rendered = render_formula(version="0.7.0", sha256=sha)
        self.assertIn('class Ncm < Formula', rendered)
        self.assertIn('version "0.7.0"', rendered)
        self.assertIn(f'sha256 "{sha}"', rendered)
        self.assertIn(
            'url "https://github.com/lexlapax/ncm-rs/releases/download/v0.7.0/ncm-v0.7.0-macos-arm64-mlx.tar.gz"',
            rendered,
        )
        self.assertIn('bin.install "bin/ncm"', rendered)
        self.assertIn('lib.install Dir["lib/*"]', rendered)
        self.assertIn('depends_on arch: :arm64', rendered)

    def test_rejects_bad_version_sha_and_insecure_url(self) -> None:
        sha = "0123456789abcdef" * 4
        with self.assertRaises(ValueError):
            render_formula(version="v0.7.0", sha256=sha)
        with self.assertRaises(ValueError):
            render_formula(version="0.7.0", sha256="not-a-sha")
        with self.assertRaises(ValueError):
            render_formula(version="0.7.0", sha256=sha, url="http://example.com/ncm.tar.gz")

    def test_cli_writes_output_file(self) -> None:
        sha = "b" * 64
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "ncm.rb"
            rc = main(["--version", "0.7.0", "--sha256", sha, "--output", str(out)])
            self.assertEqual(rc, 0)
            self.assertIn(f'sha256 "{sha}"', out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
