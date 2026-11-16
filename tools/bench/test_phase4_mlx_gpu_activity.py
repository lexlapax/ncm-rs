import unittest
import json
import plistlib
import subprocess
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

from phase4_mlx_gpu_activity import FIELDS, monitor_gpu, parse_snapshot


class GpuActivityParsingTests(unittest.TestCase):
    def test_selects_only_expected_device_wide_fields(self):
        stats = {name: i for i, name in enumerate(FIELDS)}
        data = [{'PerformanceStatistics': {**stats, 'unrelated': 'private'},
                 'AGCInfo': {'fLastSubmissionPID': 427, 'unrelated': 'private'},
                 'unrelated': 'private'}]
        self.assertEqual(parse_snapshot(data), {'statistics': stats, 'last_submission_pid': 427})

    def test_missing_ambiguous_or_malformed_counters_are_rejected(self):
        stats = {name: 0 for name in FIELDS}
        for data in ([], [{}, {}], [{}], [{'PerformanceStatistics': {}}],
                     [{'PerformanceStatistics': {**stats, FIELDS[0]: True}}],
                     [{'PerformanceStatistics': {**stats, FIELDS[0]: -1}}],
                     [{'PerformanceStatistics': stats, 'AGCInfo': {'fLastSubmissionPID': -1}}]):
            with self.subTest(data=data), self.assertRaises(ValueError):
                parse_snapshot(data)

    def test_observer_signals_readiness_and_writes_one_whitelisted_sample(self):
        stats = {name: 1 for name in FIELDS}
        payload = plistlib.dumps([{'PerformanceStatistics': stats, 'AGCInfo': {'fLastSubmissionPID': 9}}])
        stop, ready, errors = threading.Event(), threading.Event(), []

        def fake_run(*args, **kwargs):
            stop.set()
            return subprocess.CompletedProcess(args[0], 0, stdout=payload, stderr=b'')

        with tempfile.TemporaryDirectory(prefix='ncm-gpu-observer-test-') as directory:
            out = Path(directory) / 'activity.jsonl'
            with patch('phase4_mlx_gpu_activity.subprocess.run', side_effect=fake_run):
                monitor_gpu(stop, out, errors, ready)
            sample = json.loads(out.read_text())
        self.assertTrue(ready.is_set())
        self.assertEqual(errors, [])
        self.assertEqual(sample['statistics'], stats)
        self.assertEqual(sample['last_submission_pid'], 9)
        self.assertLessEqual(sample['begin_monotonic_ns'], sample['end_monotonic_ns'])

    def test_observer_failure_signals_readiness_and_records_error(self):
        stop, ready, errors = threading.Event(), threading.Event(), []
        with tempfile.TemporaryDirectory(prefix='ncm-gpu-observer-test-') as directory:
            with patch('phase4_mlx_gpu_activity.subprocess.run', side_effect=OSError('unavailable')):
                monitor_gpu(stop, Path(directory) / 'activity.jsonl', errors, ready)
        self.assertTrue(ready.is_set())
        self.assertEqual(errors, ['unavailable'])


if __name__ == '__main__':
    unittest.main()
