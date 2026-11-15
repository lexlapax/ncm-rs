"""Focused negative checks for the final matrix coverage verifier."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

root = Path(__file__).parent
spec = importlib.util.spec_from_file_location('closure_bindings', root / 'phase2-closure-bindings-v1.py')
bindings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bindings)

class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.audit = json.loads((root / 'mlx-acceptance.json').read_bytes())

    def test_original_and_coverage_corruption(self):
        self.assertEqual(bindings.verify_coverage(self.audit, 'apple'), 18)
        for mutation in ('duplicate', 'wrong_comparator', 'missing_control', 'wrong_repeats'):
            with self.subTest(mutation=mutation):
                value = copy.deepcopy(self.audit)
                if mutation == 'duplicate':
                    value['cells'][0] = copy.deepcopy(value['cells'][1])
                elif mutation == 'wrong_comparator':
                    value['cells'][0]['comparator'] = 'other'
                elif mutation == 'missing_control':
                    del value['controls']['qwen3']
                else:
                    value['controls']['codec']['processes_per_runtime'] = 17
                with self.assertRaises(ValueError):
                    bindings.verify_coverage(value, 'apple')

    def test_passing_flag_cannot_hide_a_failed_bound(self):
        for mutation in ('median', 'p95', 'median_interval', 'p95_interval', 'nan', 'memory'):
            with self.subTest(mutation=mutation):
                value = copy.deepcopy(self.audit)
                timing = value['cells'][0]['controlled_followup']
                if mutation == 'median': timing['median_ratio'] = 1.051
                elif mutation == 'p95': timing['p95_ratio'] = 1.101
                elif mutation == 'median_interval': timing['paired_process_percentile_95_interval']['median_ratio'][1] = 1.051
                elif mutation == 'p95_interval': timing['paired_process_percentile_95_interval']['p95_ratio'][1] = 1.101
                elif mutation == 'nan': timing['median_ratio'] = float('nan')
                else: next(c for c in value['cells'] if c['cell'].endswith('/VmHWM'))['peak_observed_ratio'] = 1.051
                with self.assertRaises(ValueError):
                    bindings.verify_coverage(value, 'apple')

if __name__ == '__main__':
    unittest.main()
