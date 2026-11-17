import unittest

import torch

from lm1b_training_state_probe import summarize, training_names


class Lm1bTrainingStateProbeTests(unittest.TestCase):
    def test_training_inventory_is_exact(self):
        names = [f'vq.{kind}_{i}' for kind in ('ema_cluster_size', 'ema_w', 'codebook_hit') for i in range(12)]
        self.assertEqual(len(training_names(names + ['vq.embeddings.0.weight'])), 36)
        with self.assertRaises(ValueError):
            training_names(names[:-1])
        with self.assertRaises(ValueError):
            training_names(names + ['vq.ema_w_12'])

    def test_summary_captures_shape_dtype_and_bytes(self):
        a = summarize([torch.tensor([1.0, -0.0])])[0]
        b = summarize([torch.tensor([1.0, 0.0])])[0]
        self.assertEqual(a['shape'], [2])
        self.assertEqual(a['dtype'], 'torch.float32')
        self.assertNotEqual(a['sha256'], b['sha256'])


if __name__ == '__main__':
    unittest.main()
