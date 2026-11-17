"""Validate the fixed WMT14 teacher prefix bounds."""
import unittest
from wmt14_teacher_trace import prefixes


class PrefixTests(unittest.TestCase):
    def test_shape_and_id_bounds(self):
        rows = [[1] * 96, [151679] * 96] + [[] for _ in range(3001)]
        self.assertEqual(prefixes({'prefixes': rows}), rows[:2])
        rows[0][0] = 151680
        with self.assertRaises(ValueError):
            prefixes({'prefixes': rows})
        with self.assertRaises(ValueError):
            prefixes({'prefixes': rows[:-1]})


if __name__ == '__main__':
    unittest.main()
