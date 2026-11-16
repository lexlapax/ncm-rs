"""Check LM1B inventory accounting and malformed checkpoint rejection."""
import unittest

import torch
from lm1b_inventory import describe


class Lm1bInventoryTests(unittest.TestCase):
    def test_inventory_preserves_shapes_and_detects_shared_storage(self):
        shared = torch.arange(4, dtype=torch.float32)
        result = describe({'model': {'a': shared, 'b': shared.view(2, 2)},
                           'inference': {'ids': torch.tensor([1, 2], dtype=torch.int64)}})
        self.assertEqual(result['model']['count'], 2)
        self.assertEqual(result['model']['bytes'], 32)
        self.assertEqual(result['model']['shared_storage'], [['a', 'b']])
        self.assertEqual(result['model']['tensors']['b']['shape'], [2, 2])
        self.assertEqual(result['inference']['dtypes'], {'torch.int64': 1})

    def test_rejects_non_dictionary_metadata_and_nonstring_keys(self):
        for value in ({'model': 1}, {1: torch.ones(1)}, {'model': {'not_tensor': 1}}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                describe(value)


if __name__ == '__main__':
    unittest.main()
