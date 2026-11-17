"""Reject mismatched historical outputs in the CFG diagnostic."""
import unittest
import torch
from wmt14_cfg_row_compare import analyze


class Tensors:
    def __init__(self, codes):
        self.codes = codes

    def get_tensor(self, name):
        if name == 'generator.codes':
            return self.codes
        if name == 'generator.greedy_ids':
            return torch.zeros((4, 64), dtype=torch.int64)
        raise AssertionError(name)


class RowComparisonTests(unittest.TestCase):
    def test_rejects_changed_previous_history(self):
        a = torch.zeros((4, 313), dtype=torch.int64)
        b = a.clone()
        b[1, 0] = 1
        rows_a = [{'codes': row, 'raw_ids': [0] * 64} for row in a.tolist()]
        rows_b = [{'codes': row, 'raw_ids': [0] * 64} for row in b.tolist()]
        with self.assertRaisesRegex(ValueError, 'earlier histories'):
            analyze(Tensors(a), Tensors(b), rows_a, rows_b)


if __name__ == '__main__':
    unittest.main()
