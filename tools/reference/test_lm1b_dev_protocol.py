import math
import unittest

from lm1b_dev_protocol import summary


class Lm1bDevProtocolTests(unittest.TestCase):
    def test_summary_is_deterministic_and_rounds_up(self):
        result = summary([1., 2., 3., 4., 5.], 100)
        self.assertEqual(result, summary([1., 2., 3., 4., 5.], 100))
        self.assertEqual(result['mean'], 3.)
        self.assertEqual(result['twice_sd_rounded_up'], 3.17)
        self.assertLessEqual(result['seed_bootstrap_95'][0], 3.)
        self.assertGreaterEqual(result['seed_bootstrap_95'][1], 3.)

    def test_invalid_values_rejected(self):
        for values in ([1.], [1., 2., 3., 4., math.nan]):
            with self.assertRaises(ValueError):
                summary(values, 100)


if __name__ == '__main__':
    unittest.main()
