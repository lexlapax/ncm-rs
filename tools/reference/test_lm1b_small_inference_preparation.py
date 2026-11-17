import tempfile
import unittest
from pathlib import Path

import torch

from lm1b_small_inference_preparation import bundle_states, export
from test_lm1b_export_preparation import fixture


class Lm1bSmallInferencePreparationTests(unittest.TestCase):
    def make_fixture(self):
        generator, tokenizer = fixture()
        tokenizer['model'] = {f'codec.{i}': torch.tensor([i]) for i in range(100)}
        tokenizer['model'].update({f'vq.{kind}_{i}': torch.tensor([i])
                                   for kind in ('ema_cluster_size', 'ema_w', 'codebook_hit') for i in range(12)})
        return generator, tokenizer

    def test_split_preserves_all_state(self):
        generator, tokenizer = self.make_fixture()
        bundles = bundle_states(generator, tokenizer)
        self.assertEqual({k: len(v) for k, v in bundles.items()},
                         {'generator.safetensors': 213, 'codec.safetensors': 100,
                          'training.safetensors': 36, 'inference.safetensors': 2})
        self.assertEqual(set(bundles['codec.safetensors']) | set(bundles['training.safetensors']), set(tokenizer['model']))
        self.assertFalse(set(bundles['codec.safetensors']) & set(bundles['training.safetensors']))

    def test_bad_training_names_and_outside_output_fail(self):
        generator, tokenizer = self.make_fixture()
        tokenizer['model']['vq.ema_w_12'] = tokenizer['model'].pop('vq.ema_w_0')
        with self.assertRaises(ValueError):
            bundle_states(generator, tokenizer)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                export(Path(directory) / 'missing', Path(directory) / 'missing-tokenizer', Path(directory) / 'outside')


if __name__ == '__main__':
    unittest.main()
