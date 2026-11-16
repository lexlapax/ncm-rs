import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import wmt14_inventory as subject


class Wmt14InventoryTests(unittest.TestCase):
    def test_verification_precedes_checkpoint_load(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(subject.torch, 'load') as load:
            with self.assertRaises((OSError, ValueError)):
                subject.inventory(Path(directory))
            load.assert_not_called()

    def test_rejects_non_tensor_checkpoint_entries(self):
        with mock.patch.object(subject, 'verify_input'), mock.patch.object(
            subject.torch, 'load', return_value={'model': {'invalid': 1}}
        ):
            with self.assertRaisesRegex(ValueError, 'non-tensor'):
                subject.inventory(Path('.'))

    def test_inventory_records_shapes_and_training_names(self):
        state = {'model': {'weight': torch.ones(2, 3), 'vq.codebook_hit_0': torch.zeros(4, dtype=torch.int64)}}
        with mock.patch.object(subject, 'verify_input'), mock.patch.object(subject.torch, 'load', return_value=state):
            result = subject.inventory(Path('.'))
        row = result['tokenizer.pt']
        self.assertEqual(row['count'], 2)
        self.assertEqual(row['bytes'], 56)
        self.assertEqual(row['tensors']['weight']['shape'], [2, 3])
        self.assertEqual(row['training_name_pattern_matches'], ['vq.codebook_hit_0'])


if __name__ == '__main__':
    unittest.main()
