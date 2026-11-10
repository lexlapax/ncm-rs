import copy
import unittest

from compare import summarize


def reports():
    result = []
    for runtime in ("candle-0.11.1", "pytorch-eager-math-sdpa"):
        for _ in range(3):
            result.append({
                "fixture_sha256": "fixture", "upstream_commit": "pin", "device": "cuda",
                "dtype": "float32", "tf32": False, "scope": "operator", "atol": 1e-5,
                "rtol": 1e-4, "passed": True, "warmups": 5, "trials": 30,
                "runtime": runtime, "results": [{"name": "case", "trials_ms": [1.0] * 30,
                    "errors": {"passed": True, "mismatches": 0, "max_abs_error": 0.0}}],
            })
    return result


class ComparisonTests(unittest.TestCase):
    def test_valid_report(self):
        summary = summarize(reports())
        self.assertEqual(summary["runtimes"]["candle-0.11.1"][0]["trials"], 90)
        self.assertIsNone(summary["quality"]["model_quality"])

    def test_rejects_incompatible_and_incomplete_runs(self):
        for key, value in (("fixture_sha256", "changed"), ("tf32", True),
                           ("passed", False), ("trials", 0), ("device", "cpu")):
            data = reports()
            data[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                summarize(data)
        with self.assertRaises(ValueError):
            summarize(reports()[:-1])

    def test_rejects_nan_and_numerical_failure(self):
        for value in (float("nan"), float("inf"), -1.0, 0.0):
            data = reports()
            data[0]["results"][0]["trials_ms"][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize(data)
        data = reports()
        data[0]["results"][0]["errors"]["mismatches"] = 1
        with self.assertRaises(ValueError):
            summarize(data)

    def test_rejects_source_changes_and_duplicate_cases(self):
        data = reports()
        data[0]["compiled_source_sha256"] = {"ops.rs": "new"}
        with self.assertRaises(ValueError):
            summarize(data)
        data = reports()
        data[0]["results"].append(copy.deepcopy(data[0]["results"][0]))
        with self.assertRaises(ValueError):
            summarize(data)


if __name__ == "__main__":
    unittest.main()
