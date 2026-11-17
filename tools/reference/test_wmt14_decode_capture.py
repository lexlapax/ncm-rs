"""Validate bounded WMT14 capture rows before decoding."""
import unittest
from wmt14_decode_capture import validate_outputs


class DecodeCaptureTests(unittest.TestCase):
    def test_schema_and_bounds(self):
        rows = [{'row': i, 'codes': [0] * 313, 'raw_ids': [1] * 64} for i in range(4, 7)]
        validate_outputs(rows, 4, 3)
        rows[-1]['raw_ids'][0] = 151680
        with self.assertRaises(ValueError):
            validate_outputs(rows, 4, 3)
        with self.assertRaises(ValueError):
            validate_outputs([], 3003, 1)


if __name__ == '__main__':
    unittest.main()
