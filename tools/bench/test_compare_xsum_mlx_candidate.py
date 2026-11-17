"""Check exact Metal candidate output comparisons and row attribution."""
import unittest
from compare_xsum_mlx_candidate import compare_batch


class CompareTests(unittest.TestCase):
    def test_counts_and_rows_and_rejects_short_data(self):
        reference = {'codes': [[1] * 200, [2] * 200], 'ids': [[3] * 48, [4] * 48]}
        candidate = {'codes': [row[:] for row in reference['codes']], 'ids': [row[:] for row in reference['ids']]}
        candidate['codes'][0][4] = 5
        candidate['ids'][1][8] = 6
        self.assertEqual(compare_batch(reference, candidate, 96),
                         {'rows': 2, 'code_differences': 1, 'id_differences': 1,
                          'code_rows': [96], 'id_rows': [97]})
        candidate['ids'][0].pop()
        with self.assertRaises(ValueError):
            compare_batch(reference, candidate, 96)


if __name__ == '__main__':
    unittest.main()
