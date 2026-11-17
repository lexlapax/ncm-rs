"""Check input validation for the pinned WMT14 top-1 capture."""
import unittest
from wmt14_top1_trace import prefixes


class Top1PrefixTests(unittest.TestCase):
    def test_rejects_invalid_rows_and_ids(self):
        rows = [[1] * 96, [2] * 96] + [[] for _ in range(3001)]
        self.assertEqual(prefixes({'prefixes': rows}), rows[:2])
        rows[0][0] = -1
        with self.assertRaises(ValueError):
            prefixes({'prefixes': rows})
        rows[0] = [1] * 95
        with self.assertRaises(ValueError):
            prefixes({'prefixes': rows})


if __name__ == '__main__':
    unittest.main()
