"""Check upstream metric defaults and input validation."""
import unittest
from wmt14_score_dev import score


class ScoreTests(unittest.TestCase):
    def test_exact_and_invalid(self):
        values, signatures = score(['one two three four'], ['one two three four'])
        self.assertAlmostEqual(values['bleu'], 100.0)
        self.assertAlmostEqual(values['chrf'], 100.0)
        self.assertEqual(values['length_ratio'], 1.0)
        self.assertIn('tok:13a', signatures['bleu'])
        with self.assertRaises(ValueError):
            score([], [])
        with self.assertRaises(ValueError):
            score(['a'], [''])


if __name__ == '__main__':
    unittest.main()
