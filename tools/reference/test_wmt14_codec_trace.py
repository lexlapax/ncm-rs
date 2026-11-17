"""Check WMT14 codec reference input bounds and padding."""
import unittest
from wmt14_codec_trace import codec_inputs


class Wmt14CodecTraceTests(unittest.TestCase):
    def test_uses_first_two_rows_with_exact_padding_and_truncation(self):
        rows = [[1, 2], list(range(70))] + [[] for _ in range(3001)]
        result = codec_inputs(rows)
        self.assertEqual(result[0], [1, 2] + [151643] * 62)
        self.assertEqual(result[1], list(range(64)))
        with self.assertRaises(ValueError):
            codec_inputs(rows[:-1])
        rows[0] = [151680]
        with self.assertRaises(ValueError):
            codec_inputs(rows)


if __name__ == '__main__':
    unittest.main()
