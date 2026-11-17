"""Check the diagnostic selects the original contiguous four-row batch."""
import unittest
from wmt14_cfg_row_trace import prefixes


class RowTraceTests(unittest.TestCase):
    def test_selected_rows_and_bounds(self):
        rows = [[i % 100] * 96 for i in range(3003)]
        self.assertEqual(prefixes({'prefixes': rows}), rows[2180:2184])
        rows[2181][0] = 151680
        with self.assertRaises(ValueError):
            prefixes({'prefixes': rows})


if __name__ == '__main__':
    unittest.main()
