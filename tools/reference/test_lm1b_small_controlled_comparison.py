"""Regression checks for LM1B controlled-trace numerical policy."""
import unittest

import torch

from compare_lm1b_small_controlled_trace import compare_tensor


class Lm1bControlledComparisonTests(unittest.TestCase):
    def test_reconstruction_uses_component_band_not_accumulated_band(self):
        expected = torch.tensor([0.0], dtype=torch.float32)
        candidate = torch.tensor([0.0002], dtype=torch.float32)
        reconstruction = compare_tensor('reconstruction.level_00', expected, candidate)
        logits = compare_tensor('generator.level_01.logits', expected, candidate)
        self.assertEqual(reconstruction['policy'], 'component')
        self.assertEqual(reconstruction['mismatches'], 1)
        self.assertEqual(logits['policy'], 'accumulated')
        self.assertEqual(logits['mismatches'], 0)

    def test_equal_float_band_does_not_hide_argmax_change(self):
        expected = torch.tensor([[1.0, 0.9999]], dtype=torch.float32)
        candidate = torch.tensor([[0.9999, 1.0]], dtype=torch.float32)
        result = compare_tensor('generator.level_01.logits', expected, candidate)
        self.assertEqual(result['mismatches'], 0)
        self.assertEqual(result['argmax_mismatches'], 1)

    def test_code_differences_are_exact(self):
        expected = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        candidate = torch.tensor([[1, 4, 3]], dtype=torch.int64)
        self.assertEqual(compare_tensor('generator.generated_codes', expected, candidate)['mismatches'], 1)


if __name__ == '__main__':
    unittest.main()
