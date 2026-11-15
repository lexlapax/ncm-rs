"""Focused checks for TinyStories checkpoint metadata and source authentication."""
import hashlib
import pathlib
import tempfile
import unittest

import torch

from tinystories_artifact import checkpoint_parts, verify_input


def checkpoints():
    generator = {'model': {f'weight.{i}': torch.tensor([float(i)]) for i in range(197)},
                 'inference': {'level0_unigram': torch.full((4096,), 1 / 4096)}}
    codec = {f'codec.{i}': torch.tensor([float(i)]) for i in range(70)}
    for label in ('ema_cluster_size', 'ema_w', 'codebook_hit'):
        for i in range(12):
            codec[f'vq.{label}_{i}'] = torch.tensor([i])
    return generator, {'model': codec}


class TinyStoriesArtifactTests(unittest.TestCase):
    def test_preserves_inference_and_separates_training(self):
        generator, tokenizer = checkpoints()
        result = checkpoint_parts(generator, tokenizer)
        self.assertEqual({name: len(value) for name, value in result.items()},
                         {'generator.safetensors': 197, 'codec.safetensors': 70,
                          'training.safetensors': 36, 'inference.safetensors': 1})
        self.assertIs(result['inference.safetensors']['level0_unigram'], generator['inference']['level0_unigram'])

    def test_rejects_missing_unigram_bad_distribution_and_aliases(self):
        generator, tokenizer = checkpoints()
        generator['inference'] = {}
        with self.assertRaises(ValueError): checkpoint_parts(generator, tokenizer)
        generator, tokenizer = checkpoints()
        generator['inference']['level0_unigram'][0] = float('nan')
        with self.assertRaises(ValueError): checkpoint_parts(generator, tokenizer)
        generator, tokenizer = checkpoints()
        generator['model']['weight.1'] = generator['model']['weight.0']
        with self.assertRaisesRegex(ValueError, 'shared'): checkpoint_parts(generator, tokenizer)

    def test_rejects_changed_or_symlinked_input(self):
        with tempfile.TemporaryDirectory(dir='reports') as directory:
            path = pathlib.Path(directory) / 'input'
            path.write_bytes(b'original')
            expected = hashlib.sha256(b'original').hexdigest()
            verify_input(path, 8, expected)
            path.write_bytes(b'modified')
            with self.assertRaises(ValueError): verify_input(path, 8, expected)
            link = pathlib.Path(directory) / 'link'
            link.symlink_to(path)
            with self.assertRaises(ValueError): verify_input(link, 8, expected)


if __name__ == '__main__':
    unittest.main()
