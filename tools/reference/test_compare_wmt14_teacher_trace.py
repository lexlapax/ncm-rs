"""Ensure teacher comparisons retain the existing accumulated and exact gates."""
import unittest
import torch
from compare_wmt14_teacher_trace import compare_tensor


class TeacherComparisonTests(unittest.TestCase):
    def test_head_band_and_exact_ids(self):
        ref = torch.tensor([0.0, 1.0])
        self.assertEqual(compare_tensor('generator.level_00.logits', ref, torch.tensor([0.0004, 1.0]))['mismatches'], 0)
        self.assertEqual(compare_tensor('generator.level_00.logits', ref, torch.tensor([0.0006, 1.0]))['mismatches'], 1)
        self.assertEqual(compare_tensor('generator.greedy_ids', torch.tensor([1, 2]), torch.tensor([1, 3]))['mismatches'], 1)
        close = compare_tensor('generator.level_00.logits', torch.tensor([[1.0, 1.00001]]), torch.tensor([[1.00001, 1.0]]))
        self.assertEqual(close['mismatches'], 0)
        self.assertEqual(close['argmax_mismatches'], 1)


if __name__ == '__main__':
    unittest.main()
