"""Same-history diagnostics must reproduce the existing independent selector."""
import unittest
import numpy as np
from xsum_numpy_selection import select
from xsum_top1_row_compare import cuda_identity, scores_at, validate_result


class PenaltyDiagnosticTests(unittest.TestCase):
    def test_new_rows_require_complete_trace_identity(self):
        with self.assertRaises(ValueError):
            cuda_identity(9813, None, None, None)
        with self.assertRaises(ValueError):
            cuda_identity(9813, 'trace', None, 'digest')
        self.assertEqual(cuda_identity(9813, 'trace', 'result', 'digest'), ('trace', 'result', 'digest'))
        self.assertTrue(cuda_identity(4937, None, None, None)[2])

    def test_trace_metadata_binds_requested_row(self):
        result = {'batch': 32, 'row_slot': 21, 'forced_codes': 200, 'selected_ids': [0] * 48}
        validate_result(result, 9813)
        with self.assertRaises(ValueError):
            validate_result(result, 9814)
        with self.assertRaises(ValueError):
            validate_result({**result, 'selected_ids': [0]}, 9813)

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
