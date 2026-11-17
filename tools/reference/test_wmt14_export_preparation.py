"""Exercise experimental WMT14 export preservation and overwrite rejection."""
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file
from wmt14_export_preparation import export, save_checked


class Wmt14ExportPreparationTests(unittest.TestCase):
    def test_roundtrip_preserves_bits_and_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.safetensors'
            tensors = {'float': torch.tensor([-0.0, 1.5]), 'integer': torch.tensor([0, -2], dtype=torch.int64)}
            inventory = save_checked(path, tensors)
            self.assertEqual([item['name'] for item in inventory], ['float', 'integer'])
            self.assertEqual([item['dtype'] for item in inventory], ['F32', 'I64'])
            loaded = load_file(path)
            self.assertTrue(torch.equal(loaded['float'].view(torch.int32), tensors['float'].view(torch.int32)))
            self.assertTrue(torch.equal(loaded['integer'], tensors['integer']))

    def test_existing_output_is_rejected_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaises(FileExistsError):
                export(path / 'missing-source', path / 'missing-tokenizer', path)


if __name__ == '__main__':
    unittest.main()
