import math
import unittest

import torch

from lm1b_quality_probe import entropy, valid_targets


class Lm1bQualityProbeTests(unittest.TestCase):
    def test_reference_eos_expression_including_padding_and_later_tokens(self):
        ids = torch.tensor([[7, 8, 9, 9], [7, 9, 9, 8], [7, 8, 8, 8]])
        self.assertEqual(valid_targets(ids, 9).tolist(), [[True, True, False], [True, False, True], [True, True, True]])

    def test_raw_id_entropy(self):
        self.assertAlmostEqual(entropy([1, 1, 2, 2]), math.log(2), places=6)
        self.assertEqual(entropy([101, 101, 101]), 0.0)


if __name__ == '__main__':
    unittest.main()
