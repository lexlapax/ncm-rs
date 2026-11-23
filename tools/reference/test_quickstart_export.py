"""Offline unit tests for tools/reference/quickstart_export.py."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import quickstart_export as qe


class TestQuickstartExport(unittest.TestCase):
    def test_build_plan_for_text_and_lm1b_v2_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            upstream_dir = root / "upstream"
            repo_root = Path(__file__).resolve().parents[2]

            out, steps = qe.build_plan(
                model="wmt14",
                python_exe="/usr/bin/python3",
                models_dir=models_dir,
                upstream_dir=upstream_dir,
                repo_root=repo_root,
            )
            self.assertEqual(out, models_dir / "wmt14-quickstart")
            self.assertEqual(len(steps), 4)
            self.assertIn("fetch_tokenizers.py", steps[0][1])
            self.assertIn("convert_tokenizers.py", steps[1][1])
            self.assertIn("fetch_checkpoints.py", steps[2][1])
            self.assertIn("wmt14_runtime_artifact.py", steps[3][1])

            # LM1B with optional --reference-rope builds self-contained v2 command
            rope = root / "lm1b-reference-rope-v1.safetensors"
            out_lm, steps_lm = qe.build_plan(
                model="lm1b-small",
                python_exe="/usr/bin/python3",
                models_dir=models_dir,
                upstream_dir=upstream_dir,
                repo_root=repo_root,
                reference_rope=rope,
            )
            self.assertEqual(out_lm, models_dir / "lm1b-small-quickstart")
            self.assertIn("--reference-rope", steps_lm[-1])
            self.assertIn(str(rope), steps_lm[-1])

            # Sudoku skips tokenizer steps
            out_sud, steps_sud = qe.build_plan(
                model="sudoku-easy",
                python_exe="/usr/bin/python3",
                models_dir=models_dir,
                upstream_dir=upstream_dir,
                repo_root=repo_root,
            )
            self.assertEqual(out_sud, models_dir / "sudoku-easy-quickstart")
            self.assertEqual(len(steps_sud), 2)

    def test_dry_run_and_list_do_not_touch_filesystem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "nonexistent-models"
            upstream_dir = Path(tmp) / "nonexistent-upstream"
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = qe.main(["--list"])
            self.assertEqual(rc, 0)
            self.assertIn("wmt14", buf.getvalue())

            buf2 = io.StringIO()
            with redirect_stdout(buf2):
                rc2 = qe.main(
                    [
                        "--model",
                        "xsum",
                        "--models-dir",
                        str(models_dir),
                        "--upstream-dir",
                        str(upstream_dir),
                        "--dry-run",
                    ]
                )
            self.assertEqual(rc2, 0)
            self.assertFalse(models_dir.exists())
            self.assertFalse(upstream_dir.exists())
            self.assertIn("xsum_artifact.py", buf2.getvalue())


if __name__ == "__main__":
    unittest.main()
