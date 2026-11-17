import tempfile
import unittest
from pathlib import Path

import torch

from lm1b_export_preparation import checkpoint_parts, export
from wmt14_export_preparation import save_checked


def fixture():
    model = {f'model.{i}': torch.tensor([float(i)]) for i in range(213)}
    codec = {f'codec.{i}': torch.tensor([i], dtype=torch.int64) for i in range(136)}
    level = torch.zeros(8192)
    token = torch.zeros(30528)
    level[2] = 1
    token[3] = 1
    return {'model': model, 'inference': {'level0_unigram': level, 'token_unigram': token}}, {'model': codec}


class Lm1bExportPreparationTests(unittest.TestCase):
    def test_preserves_all_parts_and_roundtrips(self):
        generator, tokenizer = fixture()
        parts = checkpoint_parts(generator, tokenizer, 'small')
        self.assertEqual([len(parts[x]) for x in ('generator.safetensors', 'codec.safetensors', 'inference.safetensors')], [213, 136, 2])
        with tempfile.TemporaryDirectory() as directory:
            for name, tensors in parts.items():
                self.assertEqual(len(save_checked(Path(directory) / name, tensors)), len(tensors))

    def test_invalid_metadata_and_distributions_fail(self):
        generator, tokenizer = fixture()
        generator['inference']['level0_unigram'][1] = float('nan')
        with self.assertRaises(ValueError):
            checkpoint_parts(generator, tokenizer, 'small')
        generator, tokenizer = fixture()
        generator['extra'] = {}
        with self.assertRaises(ValueError):
            checkpoint_parts(generator, tokenizer, 'small')
        generator, tokenizer = fixture()
        with self.assertRaises(ValueError):
            checkpoint_parts(generator, tokenizer, 'medium')

    def test_existing_output_is_rejected_before_input_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaises(FileExistsError):
                export(path / 'missing', path / 'missing-tokenizer', 'small', path)


if __name__ == '__main__':
    unittest.main()
