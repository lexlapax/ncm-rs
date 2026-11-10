"""Report reproduction and tamper rejection without MLX, Torch or ignored files."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import phase2_mlx_summary as summary
from phase2_summary import reports


class MlxSummaryTests(unittest.TestCase):
    def setUp(self):
        self.raw = summary.ROOT / summary.BASELINE

    def test_committed_anchor(self):
        self.assertEqual(summary.check_baseline(self.raw), summary.QUALIFICATION_SHA256)

    def test_reject_changed_runtime_baseline_and_sources(self):
        original = reports(self.raw, "codec")
        mutations = [
            lambda r: r["rust"][0].update(runtime="rust-cuda"),
            lambda r: r["python"][0].update(runtime="python-torch"),
            lambda r: r["python"][0].update(baseline_qualification_sha256="wrong"),
            lambda r: r["python"][0]["source_sha256"].update({"tools/reference/mlx_codec.py": "wrong"}),
        ]
        for mutate in mutations:
            runs = copy.deepcopy(original)
            mutate(runs)
            with patch.object(summary, "reports", return_value=runs):
                with self.assertRaises(ValueError):
                    summary.check_baseline(self.raw)

    def test_modified_or_missing_anchor_is_not_replaced_by_local_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor = root / summary.BASELINE
            anchor.mkdir(parents=True)
            names = ["python-mlx-baseline-report.json", "python-mlx-baseline-qualification.json"]
            for name in names:
                shutil.copyfile(self.raw / name, anchor / name)
            # Files at the old ignored location must not rescue corrupt/missing evidence.
            local = root / "reports/phase2-codec/python-mlx-v1"
            local.mkdir(parents=True)
            shutil.copyfile(self.raw / names[0], local / "report.json")
            shutil.copyfile(self.raw / names[1], local.parent / "python-mlx-v1-qualification.json")
            with patch.object(summary, "ROOT", root):
                for name in names:
                    original = (anchor / name).read_bytes()
                    (anchor / name).write_bytes(original + b"\n")
                    with self.assertRaisesRegex(ValueError, "baseline .* changed"):
                        summary.check_baseline(self.raw)
                    (anchor / name).unlink()
                    with self.assertRaises(FileNotFoundError):
                        summary.check_baseline(self.raw)
                    (anchor / name).write_bytes(original)

    def test_clean_checkout_cli_reproduces_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Only versioned inputs, no reports/, artifacts/, venv, or GPU runtime.
            for relative in [
                summary.BASELINE,
                "docs/results/2026-11-10-phase2-oracle/raw",
                "docs/results/2026-11-10-sudoku-artifacts/raw",
                "tests/fixtures",
            ]:
                shutil.copytree(summary.ROOT / relative, root / relative)
            tools = root / "tools/bench"
            tools.mkdir(parents=True)
            for name in ("phase2_mlx_summary.py", "phase2_summary.py"):
                shutil.copyfile(summary.ROOT / "tools/bench" / name, tools / name)
            output = root / "summary.json"
            subprocess.run([sys.executable, str(tools / "phase2_mlx_summary.py"),
                            str(root / summary.BASELINE), str(output)], check=True, cwd=root)
            actual = json.loads(output.read_bytes())
            expected = json.loads((self.raw.parent / "summary.json").read_bytes())
            # Historical metrics stay exact; the summarizer's own provenance changes.
            sources = actual.pop("source_sha256")
            expected.pop("source_sha256")
            self.assertEqual(actual, expected)
            for name, sha in sources.items():
                self.assertEqual(sha, hashlib.sha256((root / name).read_bytes()).hexdigest())
            self.assertFalse((root / "reports").exists())
            result = subprocess.run([sys.executable, str(tools / "phase2_mlx_summary.py"),
                                     str(root / summary.BASELINE), str(output)],
                                    cwd=root, capture_output=True)
            self.assertNotEqual(result.returncode, 0, "must not overwrite a report")


if __name__ == "__main__":
    unittest.main()
