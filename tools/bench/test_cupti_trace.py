import json
from pathlib import Path
import random
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from run_phase2_linux_comparison import order
from summarize_cupti_trace import summarize
from summarize_tokenizer_control import interval, summarize as control_summary


class CuptiTraceTests(unittest.TestCase):
    def test_counts_not_wall_time(self):
        raw = b'''CONCURRENT_KERNEL [ 1, 5 ] duration 4, "f32"
DRIVER [ 0, 100001 ] duration 100001, "cuMemAllocAsync"
MEMCPY "HTOD" [ 1, 4 ] duration 3, size 32, copyCount 1
MEMORY2 [ 1 ] memoryOperationType ALLOCATION, memoryKind DEVICE, size 32, address 1
'''
        result = summarize(raw)
        self.assertEqual(result["totals"]["CONCURRENT_KERNEL"], {"count": 1, "sum_ns": 4})
        alloc = next(row for row in result["events"] if row["name"] == "cuMemAllocAsync")
        self.assertEqual(alloc["over_100us"], 1)
        self.assertEqual(result["memcpy"]["HTOD"]["bytes"], 32)
        self.assertEqual(result["memory_events"]["ALLOCATION/DEVICE"]["bytes"], 32)
        with self.assertRaises(ValueError): summarize(raw + b"Dropped 2 records")
        with self.assertRaises(ValueError): summarize(b"no events")

    def test_control_report_rejects_mutations(self):
        archive = Path(__file__).resolve().parents[2]/"docs/results/2026-11-10-phase2-cuda-remediation/raw/tiny-control-v1.tar.gz"
        with tempfile.TemporaryDirectory() as tmp, patch("summarize_tokenizer_control.interval", return_value={}):
            root = Path(tmp)
            with tarfile.open(archive) as bundle: bundle.extractall(root, filter="data")
            self.assertTrue(control_summary(root)["native_binaries_identical"])
            path = root/"candidate/tokenizer-tinystories-rust-1.json"
            original = path.read_bytes()
            for key, value in (("runtime", "wrong"), ("warmups", 0), ("manifest_sha256", "0"*64),
                               ("fixture_sha256", "0"*64), ("source_sha256", {})):
                report = json.loads(original); report[key] = value
                path.write_text(json.dumps(report))
                with self.subTest(key=key), self.assertRaises(ValueError): control_summary(root)
            report = json.loads(original); report["measurements"][0]["output_sha256"] = "wrong"
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError): control_summary(root)
            path.write_bytes(original); path.unlink()
            with self.assertRaises(ValueError): control_summary(root)

    def test_process_resampling_preserves_identity(self):
        processes = [[1., 2.], [3., 4.], [8., 9.]]
        self.assertEqual(interval(processes, processes, random.Random(42), repeats=40),
                         {"median_ratio": [1., 1.], "p95_ratio": [1., 1.]})
        with self.assertRaises(ValueError): interval(processes, processes[:2], random.Random(42))

    def test_counterbalanced_process_order(self):
        expected = {"previous", "candidate", "python"}
        rows = [order(i) for i in range(1, 4)]
        for row in rows: self.assertEqual(set(row), expected)
        for column in zip(*rows): self.assertEqual(set(column), expected)
        self.assertEqual(order(4), order(1))


if __name__ == "__main__":
    unittest.main()
