"""Check bounded range coverage and mismatch attribution."""
import unittest
from compare_wmt14_cuda_capture import compare


class CaptureComparisonTests(unittest.TestCase):
    def test_nonzero_start_and_partial_batch(self):
        rows = [{'row': i, 'codes': [0] * 313, 'raw_ids': [1] * 64} for i in range(5, 12)]
        other = [{'row': r['row'], 'codes': r['codes'][:], 'raw_ids': r['raw_ids'][:]} for r in rows]
        other[2]['codes'][0] = 1
        self.assertEqual(compare(rows, other, 5, 7)['code_rows'], [7])
        other[-1]['row'] = 99
        with self.assertRaises(ValueError):
            compare(rows, other, 5, 7)
        with self.assertRaises(ValueError):
            compare(rows, rows, 3000, 7)


if __name__ == '__main__':
    unittest.main()
