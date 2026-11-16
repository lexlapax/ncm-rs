"""Check row alignment and distinct code/output counts in batch diagnostics."""
import unittest
from check_xsum_top1_batch_shape import differences


class BatchDifferenceTests(unittest.TestCase):
    def test_differences_are_attributed_to_corpus_rows(self):
        a = {'codes': [[1, 2], [3, 4]], 'ids': [[5], [6]]}
        b = {'codes': [[1, 9], [3, 4]], 'ids': [[5], [7]]}
        self.assertEqual(differences(a, b, 100), {'code_differences': 1, 'id_differences': 1,
                                                    'code_rows': [100], 'id_rows': [101]})


if __name__ == '__main__':
    unittest.main()
