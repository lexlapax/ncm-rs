"""Validate Metal benchmark receipts and runtime ordering."""
import unittest

from run_phase4_xsum_mlx_bench import checked_output, order


class XsumMlxBenchTests(unittest.TestCase):
    def result(self):
        return {'batch': 1, 'warmups': 5, 'trials_seconds': [0.1] * 30,
                'codes': [[1] * 200], 'ids': [[2] * 48]}

    def test_validates_shape_counts_values_and_timings(self):
        self.assertEqual(len(checked_output(self.result(), 1)[0]), 200)
        for key, value in [('warmups', 2), ('trials_seconds', [float('nan')] * 30),
                           ('codes', [[4096] * 200]), ('ids', [[2] * 47])]:
            with self.subTest(key=key):
                result = self.result()
                result[key] = value
                with self.assertRaises(ValueError):
                    checked_output(result, 1)

    def test_alternates_runtime_order(self):
        self.assertEqual(order(1, 0), ('rust', 'python'))
        self.assertEqual(order(1, 1), ('python', 'rust'))
        self.assertEqual(order(4, 0), ('python', 'rust'))


if __name__ == '__main__':
    unittest.main()
