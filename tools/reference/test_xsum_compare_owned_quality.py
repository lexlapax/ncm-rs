"""Check code and ID disagreements are kept separate and row-aligned."""
import unittest
from xsum_compare_owned_quality import compare_rows


class OwnedQualityCompareTests(unittest.TestCase):
    def test_counts_and_alignment(self):
        a = {'indices': [4, 6], 'codes': [[0, 1], [2, 3]], 'raw_ids': [[0], [1]]}
        b = {'indices': [4, 6], 'codes': [[0, 2], [2, 3]], 'raw_ids': [[0], [2]]}
        self.assertEqual(compare_rows(a, b), {'codes': 1, 'ids': 1, 'code_rows': [4], 'id_rows': [6]})
        b['indices'] = [6, 4]
        with self.assertRaises(ValueError):
            compare_rows(a, b)


if __name__ == '__main__':
    unittest.main()
