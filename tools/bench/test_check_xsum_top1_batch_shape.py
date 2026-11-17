"""Check row alignment and distinct code/output counts in batch diagnostics."""
import unittest
from check_xsum_top1_batch_shape import differences, validate_starts


class BatchDifferenceTests(unittest.TestCase):
    def test_batch_starts_are_unique_aligned_and_full(self):
        validate_starts([9792, 9984])
        for starts in ([], [1], [-32], [9792, 9792], [11328]):
            with self.subTest(starts=starts), self.assertRaises(ValueError):
                validate_starts(starts)

    def test_differences_are_attributed_to_corpus_rows(self):
        a = {'codes': [[1, 2], [3, 4]], 'ids': [[5], [6]]}
        b = {'codes': [[1, 9], [3, 4]], 'ids': [[5], [7]]}
        self.assertEqual(differences(a, b, 100), {'code_differences': 1, 'id_differences': 1,
                                                    'code_rows': [100], 'id_rows': [101]})


if __name__ == '__main__':
    unittest.main()
