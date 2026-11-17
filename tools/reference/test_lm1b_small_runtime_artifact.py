import tempfile
import unittest
from pathlib import Path

from lm1b_small_runtime_artifact import bundle_states, export
from test_lm1b_small_inference_preparation import Lm1bSmallInferencePreparationTests


class Lm1bSmallRuntimeArtifactTests(unittest.TestCase):
    def test_all_tensor_groups_preserved(self):
        generator, tokenizer = Lm1bSmallInferencePreparationTests().make_fixture()
        bundles = bundle_states(generator, tokenizer)
        self.assertEqual({name: len(values) for name, values in bundles.items()},
                         {'generator.safetensors': 213, 'codec.safetensors': 100,
                          'training.safetensors': 36, 'inference.safetensors': 2})
        self.assertEqual(set(bundles['codec.safetensors']) | set(bundles['training.safetensors']), set(tokenizer['model']))
        self.assertFalse(set(bundles['codec.safetensors']) & set(bundles['training.safetensors']))

    def test_outside_output_and_unexpected_training_state_rejected(self):
        generator, tokenizer = Lm1bSmallInferencePreparationTests().make_fixture()
        tokenizer['model']['vq.ema_w_12'] = tokenizer['model'].pop('vq.ema_w_0')
        with self.assertRaises(ValueError):
            bundle_states(generator, tokenizer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaises(ValueError):
                export(path / 'source', path / 'tokenizer', path / 'output')


if __name__ == '__main__':
    unittest.main()
