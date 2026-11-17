"""Check exact-output and raw-ID gates in held-out scoring."""
import unittest
from unittest.mock import patch
from wmt14_score_heldout import evaluate


class HeldoutScoreTests(unittest.TestCase):
    def test_identity_shortcut_and_id_gate(self):
        hypotheses = ['one two three four', 'five six seven eight']
        result = evaluate(hypotheses, hypotheses[:], hypotheses[:], [])
        self.assertTrue(result['all_passed'])
        self.assertTrue(result['bootstrap_identical_hypotheses_shortcut'])
        self.assertTrue(all(interval == [0.0, 0.0] for interval in result['paired_intervals_95'].values()))
        self.assertFalse(evaluate(hypotheses, hypotheses[:], hypotheses[:], [1])['all_passed'])
        intervals = {key: [0.0, 0.0] for key in ('bleu', 'bleu_lowercase', 'chrf', 'length_ratio')}
        intervals['bleu'] = [-0.5001, 0.0]
        with patch('wmt14_score_heldout.paired', return_value=(intervals, False)):
            self.assertFalse(evaluate(hypotheses, hypotheses[:], hypotheses[:], [])['all_passed'])


if __name__ == '__main__':
    unittest.main()
