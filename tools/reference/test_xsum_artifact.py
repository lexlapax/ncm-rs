"""Validate XSum's pinned inference inventory and training-state separation."""
import unittest
import tempfile

import torch
from safetensors.torch import save_file

from xsum_artifact import checkpoint_parts, reference_rope


def checkpoints():
    generator = {'model': {f'weight.{i}': torch.tensor([float(i)]) for i in range(194)},
                 'inference': {'token_unigram': torch.full((50304,), 1 / 50304)}}
    codec = {f'codec.{i}': torch.tensor([float(i)]) for i in range(53)}
    for label in ('ema_cluster_size', 'ema_w', 'codebook_hit'):
        for i in range(9):
            codec[f'vq.{label}_{i}'] = torch.tensor([i])
    return generator, {'model': codec}


class XSumArtifactTests(unittest.TestCase):
    def test_pinned_rotary_reference(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        reference = root / 'reports/phase4-xsum-rope-reference-v5'
        if not reference.is_dir():
            self.skipTest('pinned rotary reference has not been generated')
        tables = reference_rope(reference)
        self.assertEqual(set(tables), {'rope_cos', 'rope_sin'})
        self.assertEqual(tuple(tables['rope_cos'].shape), (512, 32))

    def test_rejects_noncanonical_rotary_reference(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        (root / 'reports').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root / 'reports') as directory:
            save_file({'cos': torch.zeros(512, 32), 'sin': torch.zeros(512, 32)},
                      str(Path(directory) / 'rope.safetensors'))
            with self.assertRaises(ValueError):
                reference_rope(Path(directory))

    def test_preserves_allowance_and_training_state(self):
        generator, tokenizer = checkpoints()
        result = checkpoint_parts(generator, tokenizer)
        self.assertEqual({name: len(value) for name, value in result.items()},
                         {'generator.safetensors': 194, 'codec.safetensors': 53,
                          'training.safetensors': 27, 'inference.safetensors': 1})
        self.assertIs(result['inference.safetensors']['token_unigram'], generator['inference']['token_unigram'])

    def test_rejects_missing_invalid_or_aliased_state(self):
        for mode in ('missing', 'nonfinite', 'negative', 'wrong_shape', 'alias', 'training_class'):
            with self.subTest(mode=mode):
                generator, tokenizer = checkpoints()
                if mode == 'missing': generator['inference'] = {}
                if mode == 'nonfinite': generator['inference']['token_unigram'][0] = float('nan')
                if mode == 'negative': generator['inference']['token_unigram'][0] = -1
                if mode == 'wrong_shape': generator['inference']['token_unigram'] = torch.ones(4)
                if mode == 'alias': generator['model']['weight.1'] = generator['model']['weight.0']
                if mode == 'training_class': tokenizer['model']['renamed'] = tokenizer['model'].pop('vq.ema_w_0')
                with self.assertRaises(ValueError):
                    checkpoint_parts(generator, tokenizer)


if __name__ == '__main__':
    unittest.main()
