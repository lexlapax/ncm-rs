import math
import unittest

from lm1b_quality_compare import SEEDS, checked_values, compare


def records(ppl=100., entropy=4.):
    return [{'seed': seed, 'count': 256, 'ppl': ppl, 'entropy': entropy} for seed in SEEDS]


class Lm1bQualityCompareTests(unittest.TestCase):
    def test_equal_metrics_pass_and_are_deterministic(self):
        result = compare(records(), list(reversed(records())))
        self.assertTrue(result['metrics_passed'])
        self.assertEqual(result['difference'], {'ppl': 0.0, 'entropy': 0.0})
        self.assertEqual(result, compare(records(), records()))

    def test_threshold_failures_are_visible(self):
        self.assertFalse(compare(records(), records(ppl=106.))[ 'checks']['ppl_mean'])
        self.assertFalse(compare(records(), records(entropy=4.005))['checks']['entropy_mean'])
        varying = records()
        varying[0]['ppl'] = 120.
        result = compare(records(), varying)
        self.assertTrue(result['checks']['ppl_mean'])
        self.assertFalse(result['checks']['ppl_interval'])

    def test_invalid_or_missing_seed_records_fail(self):
        invalid = records()
        invalid[0]['ppl'] = math.nan
        for value in (invalid, records()[:-1], records() + records()[:1]):
            with self.assertRaises(ValueError):
                checked_values(value)
        duplicate = records()
        duplicate[0]['seed'] = duplicate[1]['seed']
        with self.assertRaises(ValueError):
            checked_values(duplicate)
        bad_count = records()
        bad_count[0]['count'] = True
        with self.assertRaises(ValueError):
            checked_values(bad_count)


if __name__ == '__main__':
    unittest.main()
