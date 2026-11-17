import unittest

from lm1b_small_codec_trace import codec_inputs


class Lm1bSmallCodecTraceTests(unittest.TestCase):
    def test_input_rows_are_exact_and_reject_bad_values(self):
        rows = [[0] * 128 for _ in range(32)]
        rows[1][0] = 30527
        self.assertEqual(codec_inputs(rows), rows[:2])
        bad = [row.copy() for row in rows]
        bad[10][0] = 30528
        with self.assertRaises(ValueError):
            codec_inputs(bad)
        with self.assertRaises(ValueError):
            codec_inputs(rows[:-1])
        rows[0][0] = True
        with self.assertRaises(ValueError):
            codec_inputs(rows)


if __name__ == '__main__':
    unittest.main()
