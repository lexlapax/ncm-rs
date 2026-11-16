"""Check sparse outlier arithmetic and mismatch rejection with synthetic tensors."""
from copy import deepcopy
import unittest

from validate_phase4_rust_outliers import f32, validate_pair


class RustOutlierValidationTests(unittest.TestCase):
    def setUp(self):
        overlap = {'both_outside': 0, 'rust_only_outside': 0, 'python_only_outside': 0,
                   'neither_outside': 8, 'rust_python_different_bits': 0}
        self.prior = {f'generator.level_{i:02}.logits': {'elements': 8, 'cross_runtime_overlap': deepcopy(overlap)}
                      for i in range(11)}
        self.prior['generator.final_logits'] = {'elements': 8, 'cross_runtime_overlap': deepcopy(overlap)}
        self.actual = deepcopy(self.prior)
        for value in self.actual.values():
            value['cross_runtime_overlap']['rust_only_values'] = []
        self.key = 'generator.final_logits'
        for data in (self.actual, self.prior):
            item = data[self.key]['cross_runtime_overlap']
            item.update(rust_only_outside=1, neither_outside=7, rust_python_different_bits=1)
        cuda, rust, python = 0.0, f32(0.0006), f32(0.0004)
        self.record = dict(flat_offset=1, cuda=cuda, rust_mlx=rust, python_mlx=python,
                           band_limit=0.0005, rust_abs_error=abs(rust-cuda), python_abs_error=abs(python-cuda))
        self.actual[self.key]['cross_runtime_overlap']['rust_only_values'].append(self.record)

    def test_valid_crossing_and_fp32_roundtrip(self):
        records = validate_pair(self.actual, self.prior)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['tensor'], self.key)
        self.assertGreater(records[0]['rust_ratio'], 1)
        self.assertGreater(records[0]['rust_python_abs'], 0)
        self.assertNotEqual(f32(0.0006), 0.0006)

    def test_rejects_tampering_and_invalid_crossings(self):
        for mode in ('count', 'offset', 'nonfinite', 'band', 'rust_inside', 'python_outside', 'inventory'):
            with self.subTest(mode=mode):
                actual = deepcopy(self.actual)
                item = actual[self.key]['cross_runtime_overlap']
                record = item['rust_only_values'][0]
                if mode == 'count': item['neither_outside'] -= 1
                if mode == 'offset': record['flat_offset'] = 8
                if mode == 'nonfinite': record['rust_mlx'] = float('nan')
                if mode == 'band': record['band_limit'] = 0.0004
                if mode == 'rust_inside': record.update(rust_mlx=0.0, rust_abs_error=0.0)
                if mode == 'python_outside': record.update(python_mlx=f32(0.0007), python_abs_error=f32(0.0007))
                if mode == 'inventory': actual.pop('generator.level_00.logits')
                with self.assertRaises(ValueError):
                    validate_pair(actual, self.prior)


if __name__ == '__main__':
    unittest.main()
