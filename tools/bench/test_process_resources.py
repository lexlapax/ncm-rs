import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from run_process_resources import measure, run


class ProcessResourcesTests(unittest.TestCase):
    def jobs(self, root):
        code = "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('{}')"
        return [{"name": name, "argv": [sys.executable, "-c", code, "{report}"], "cwd": str(root)}
                for name in ("previous", "python", "candidate")]

    def test_rotating_fresh_processes_and_units(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp)
            result = run(self.jobs(root), root / "out")
            self.assertEqual(result["status"], "complete")
            self.assertTrue(result["diagnostic_only"])
            self.assertEqual([r["name"] for r in result["runs"]],
                             ["previous", "python", "candidate", "python", "candidate", "previous",
                              "candidate", "previous", "python"])
            for row in result["runs"]:
                self.assertEqual(row["exit_code"], 0)
                self.assertGreater(row["elapsed_ms"], 0)
                self.assertGreater(row["user_cpu_ms"] + row["system_cpu_ms"], 0)
                self.assertGreater(row["wait4_peak_rss_bytes_including_pre_exec"], 1024 * 1024)
                self.assertEqual(len(row["report_sha256"]), 64)
            with self.assertRaises(FileExistsError):
                run(self.jobs(root), root / "out")

    def test_child_high_water_not_previous_child_high_water(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            large = measure([sys.executable, "-c", "x=bytearray(64*1024*1024)"], tmp,
                            root / "large.out", root / "large.err", os.environ.copy())
            small = measure([sys.executable, "-c", "pass"], tmp,
                            root / "small.out", root / "small.err", os.environ.copy())
            field = "wait4_peak_rss_bytes_including_pre_exec"
            self.assertGreater(large[field], small[field] + 32 * 1024 * 1024)

    def test_rejects_invalid_protocol_and_injection(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp)
            for repeats in (0, 1, 2):
                with self.assertRaises(ValueError):
                    run(self.jobs(root), root / "out", repeats)
            jobs = self.jobs(root)
            jobs[0]["name"] = "../unsafe"
            with self.assertRaises(ValueError):
                run(jobs, root / "out")
            jobs = self.jobs(root)
            jobs[0]["argv"][-1] = "no-placeholder"
            with self.assertRaises(ValueError):
                run(jobs, root / "out")
            with patch.dict(os.environ, {"LD_PRELOAD": "untrusted.so"}):
                with self.assertRaises(ValueError):
                    run(self.jobs(root), root / "out")
            self.assertFalse((root / "out").exists())

    def test_failed_child_is_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp)
            jobs = self.jobs(root)
            jobs[0]["argv"][2] = "raise SystemExit(7)"
            with self.assertRaises(RuntimeError):
                run(jobs, root / "out")
            result = json.loads((root / "out/resources.json").read_bytes())
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(len(result["runs"]), 1)
            self.assertEqual(result["runs"][0]["exit_code"], 7)

    def test_missing_report_is_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp)
            jobs = self.jobs(root)
            jobs[0]["argv"][2] = "pass"
            with self.assertRaises(FileNotFoundError):
                run(jobs, root / "out")
            result = json.loads((root / "out/resources.json").read_bytes())
            self.assertEqual(result["status"], "incomplete")
            self.assertNotIn("report_sha256", result["runs"][0])


if __name__ == "__main__":
    unittest.main()
