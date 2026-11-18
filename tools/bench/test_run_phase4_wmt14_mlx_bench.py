import json
import tempfile
import unittest
from pathlib import Path

from run_phase4_wmt14_mlx_bench import CELLS, checked_output, checked_rss, ordered_runs, summarize


class Wmt14MlxBenchTests(unittest.TestCase):
    def test_fresh_process_order(self):
        for cell in CELLS:
            runs = list(ordered_runs(cell))
            self.assertEqual(len(set(runs)), 6)
            orders = [[runtime for runtime, _ in runs[i:i + 2]] for i in range(0, len(runs), 2)]
            self.assertEqual(orders[0], orders[2])
            self.assertEqual(orders[0], orders[1][::-1])

    def test_rejects_mismatch_or_incomplete_trials(self):
        rows = [{'codes': [0] * 313, 'raw_ids': [0] * 64}]
        record = {'cfg': 0, 'batch': 1, 'warmups': 5, 'start': 0,
                  'trials_seconds': [.02] * 30, 'load_seconds': 1,
                  'mlx_memory': {'active': 3, 'cache': 4, 'peak': 5}, **rows[0]}
        self.assertEqual(checked_output(record, 'rust', 0, 1, rows)['load_seconds'], 1)
        for update in ({'start': 4}, {'codes': [False] * 313}, {'raw_ids': [1] * 64},
                       {'trials_seconds': [float('nan')] * 30}, {'trials_seconds': [.02] * 29}):
            with self.subTest(fields=list(update)), self.assertRaises(ValueError):
                checked_output(dict(record, **update), 'rust', 0, 1, rows)

    def test_process_rss_requires_macos_units(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = root / 'test.meta.json'
            meta.write_text(json.dumps({'returncode': 0, 'child_peak_rss_platform_units': 128}))
            self.assertEqual(checked_rss(root, 'test', True, 'Darwin'), 128)
            with self.assertRaisesRegex(ValueError, 'macOS process RSS'):
                checked_rss(root, 'test', True, 'Linux')
            meta.write_text(json.dumps({'returncode': 1, 'child_peak_rss_platform_units': 128}))
            with self.assertRaisesRegex(ValueError, 'invalid process status'):
                checked_rss(root, 'test', True, 'Darwin')

    def test_pooled_tail_can_exceed_each_process_median(self):
        records = [{'trials_seconds': [.01] * 25 + [.12] * 5, 'load_seconds': 1,
                    'peak_rss_bytes': 64, 'mlx_memory': {'active': 10, 'cache': 30, 'peak': 20}}] * 3
        summary = summarize(records)
        self.assertEqual(summary['median_ms'], 10)
        self.assertEqual(summary['p95_ms'], 120)
        self.assertEqual(summary['median_peak_rss_bytes'], 64)
        self.assertEqual(summary['median_mlx_memory']['cache'], 30)


if __name__ == '__main__':
    unittest.main()
