"""Validate source prefixes for the explicit WMT14 CFG diagnostic."""
import unittest
from wmt14_cfg_trace import prefixes


class CfgPrefixTests(unittest.TestCase):
    def test_rejects_invalid_prefix_values_and_shape(self):
        rows = [[1] * 96, [2] * 96] + [[] for _ in range(3001)]
        self.assertEqual(prefixes({'prefixes': rows}), rows[:2])
        rows[1][0] = 151680
        with self.assertRaises(ValueError):
            prefixes({'prefixes': rows})
        with self.assertRaises(ValueError):
            prefixes({'prefixes': rows[:-1]})


if __name__ == '__main__':
    unittest.main()
