"""Same-history diagnostics must reproduce the existing independent selector."""
import unittest
import numpy as np
from xsum_numpy_selection import select
from xsum_top1_row_compare import scores_at


class PenaltyDiagnosticTests(unittest.TestCase):
    def test_scores_reproduce_each_selector_choice(self):
        logits = np.random.default_rng(42).normal(size=(1, 8, 16)).astype(np.float32)
        unigram = np.full(16, 1 / 16, dtype=np.float32)
        selected = select(logits, unigram)[0].tolist()
        for t in range(8):
            self.assertEqual(int(scores_at(logits[0], unigram, selected[:t], t).argmax()), selected[t])
        with self.assertRaises(ValueError):
            scores_at(logits[0], unigram, selected[:2], 3)


if __name__ == '__main__':
    unittest.main()
