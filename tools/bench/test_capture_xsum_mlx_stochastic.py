"""Reject malformed stochastic Metal capture rows before recording them."""
import unittest
from capture_xsum_mlx_stochastic import validate_output


class StochasticCaptureTests(unittest.TestCase):
    def test_valid_rows_and_invalid_schema_or_values(self):
        valid = {'codes': [[0] * 200], 'ids': [[0] * 48]}
        validate_output(valid, 1)
        for bad in ({'codes': [], 'ids': []}, {'codes': [[4096] * 200], 'ids': [[0] * 48]},
                    {'codes': [[0] * 200], 'ids': [[False] * 48]}, {'codes': [[0] * 200]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_output(bad, 1)


if __name__ == '__main__':
    unittest.main()
