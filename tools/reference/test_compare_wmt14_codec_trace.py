"""Regression tests for the frozen WMT14 comparison bands."""
import unittest
import torch
from compare_wmt14_codec_trace import compare_tensor


class CompareTests(unittest.TestCase):
    def test_float_band_and_integer_disagreement(self):
        ref = torch.tensor([0.0, 1.0])
        self.assertEqual(compare_tensor('decode.logits', ref, torch.tensor([0.0004, 1.0]))['mismatches'], 0)
        self.assertEqual(compare_tensor('decode.logits', ref, torch.tensor([0.0006, 1.0]))['mismatches'], 1)
        self.assertEqual(compare_tensor('decode.ids', torch.tensor([1, 2]), torch.tensor([1, 3]))['mismatches'], 1)
        with self.assertRaises(ValueError):
            compare_tensor('x', ref, torch.tensor([float('nan'), 1.0]))


if __name__ == '__main__':
    unittest.main()
