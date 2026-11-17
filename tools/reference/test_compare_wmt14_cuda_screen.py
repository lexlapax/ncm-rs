"""Check row attribution and input validation for the WMT14 screen."""
import unittest
from compare_wmt14_cuda_screen import compare


class ScreenComparisonTests(unittest.TestCase):
    def test_attribution_and_shape(self):
        rows = [{'row': i, 'codes': [0] * 313, 'raw_ids': [1] * 64} for i in range(32)]
        other = [{'row': r['row'], 'codes': r['codes'][:], 'raw_ids': r['raw_ids'][:]} for r in rows]
        other[3]['codes'][4] = 1
        other[5]['raw_ids'][2] = 7
        result = compare(rows, other)
        self.assertEqual((result['code_differences'], result['code_rows']), (1, [3]))
        self.assertEqual((result['id_differences'], result['id_rows']), (1, [5]))
        other[0]['raw_ids'].pop()
        with self.assertRaises(ValueError):
            compare(rows, other)


if __name__ == '__main__':
    unittest.main()
