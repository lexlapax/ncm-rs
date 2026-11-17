"""Verify WMT14 generation artifact keeps every checkpoint tensor."""
import tempfile
import unittest
from pathlib import Path

from test_wmt14_artifact import fixtures
from wmt14_runtime_artifact import bundle_states, export


class Wmt14RuntimeArtifactTests(unittest.TestCase):
    def test_bundle_split_preserves_every_tensor(self):
        generator, tokenizer = fixtures()
        bundles = bundle_states(generator, tokenizer)
        self.assertEqual({name: len(value) for name, value in bundles.items()},
                         {'generator.safetensors': 186, 'codec.safetensors': 63, 'training.safetensors': 33})
        self.assertEqual(set(bundles['codec.safetensors']) | set(bundles['training.safetensors']),
                         set(tokenizer['model']))
        self.assertFalse(set(bundles['codec.safetensors']) & set(bundles['training.safetensors']))
        self.assertIs(bundles['training.safetensors']['vq.ema_w_10'], tokenizer['model']['vq.ema_w_10'])

    def test_unexpected_state_and_existing_or_outside_output_fail(self):
        generator, tokenizer = fixtures()
        tokenizer['model']['vq.ema_w_11'] = tokenizer['model'].pop('vq.ema_w_0')
        with self.assertRaises(ValueError):
            bundle_states(generator, tokenizer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaises(ValueError):
                export(path / 'missing', path / 'missing-tokenizer', path)


if __name__ == '__main__':
    unittest.main()
