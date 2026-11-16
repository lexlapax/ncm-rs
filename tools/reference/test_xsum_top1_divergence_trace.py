"""Validate the fixed-history diagnostic's selected corpus rows."""
import unittest
from xsum_top1_divergence_trace import ROWS, select_rows


class TraceInputTests(unittest.TestCase):
    def test_selection_and_corruption_rejection(self):
        data = {'profile': 'xsum', 'inputs': [[0] * 560] * 11334}
        scored = {'rows': [{} for _ in range(11334)]}
        for index in ROWS:
            scored['rows'][index] = {'index': index, 'codes': [1] * 200, 'raw_ids': [2] * 48}
        prefixes, codes, ids = select_rows(data, scored)
        self.assertEqual([len(row) for row in prefixes], [512, 512])
        self.assertEqual(codes[0], [1] * 200)
        self.assertEqual(ids[1], [2] * 48)
        scored['rows'][ROWS[0]]['codes'][0] = 4096
        with self.assertRaises(ValueError):
            select_rows(data, scored)


if __name__ == '__main__':
    unittest.main()
