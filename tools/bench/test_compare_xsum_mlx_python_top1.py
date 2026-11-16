"""Keep raw differences distinct from tokens reaching the text decoder."""
import unittest
from compare_xsum_mlx_python_top1 import decoded_tokens


class DecodeBoundaryTests(unittest.TestCase):
    def test_eos_and_pad_filtering(self):
        self.assertEqual(decoded_tokens([1, 50257, 2, 50256, 3]), [1, 2])
        self.assertEqual(decoded_tokens([1, 50256, 2]), decoded_tokens([1, 50256, 3]))
        self.assertNotEqual(decoded_tokens([1, 2]), decoded_tokens([1, 3]))


if __name__ == '__main__':
    unittest.main()
