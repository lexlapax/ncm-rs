"""Reject malformed macOS RSS output and normalize nested token rows."""
import unittest
from summarize_xsum_mlx_screen import flatten, maximum_rss


class MetalScreenSummaryTests(unittest.TestCase):
    def test_rss_parser_requires_one_positive_measurement(self):
        self.assertEqual(maximum_rss('  12345  maximum resident set size\n'), 12345)
        for text in ('', '0 maximum resident set size\n', '1 maximum resident set size\n2 maximum resident set size\n'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                maximum_rss(text)

    def test_flatten_preserves_batched_token_order(self):
        self.assertEqual(flatten([[1, 2], [3]]), [1, 2, 3])
        self.assertEqual(flatten([1, 2, 3]), [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
