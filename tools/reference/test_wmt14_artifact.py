"""Check structural preflight without claiming WMT14 inference support."""
import unittest

import torch
from wmt14_artifact import checkpoint_parts


def fixtures():
    generator = {'model': {f'g.{i}': torch.tensor([float(i)]) for i in range(186)}}
    model = {f'c.{i}': torch.tensor([float(i)]) for i in range(63)}
    for kind in ('ema_cluster_size', 'ema_w', 'codebook_hit'):
        for index in range(11):
            model[f'vq.{kind}_{index}'] = torch.tensor([index])
    return generator, {'model': model}


class Wmt14CheckpointTests(unittest.TestCase):
    def test_preserves_every_tensor(self):
        generator, tokenizer = fixtures()
        result = checkpoint_parts(generator, tokenizer)
        self.assertEqual({name: len(value) for name, value in result.items()},
                         {'generator': 186, 'codec_other': 63, 'ema_and_hits': 33})
        self.assertIs(result['ema_and_hits']['vq.ema_w_10'], tokenizer['model']['vq.ema_w_10'])
        self.assertEqual(set(result['codec_other']) | set(result['ema_and_hits']), set(tokenizer['model']))

    def test_rejects_missing_renamed_and_shared_tensors(self):
        for mode in ('missing', 'renamed', 'alias', 'extra_metadata'):
            with self.subTest(mode=mode):
                generator, tokenizer = fixtures()
                if mode == 'missing':
                    del tokenizer['model']['vq.ema_w_0']
                elif mode == 'renamed':
                    tokenizer['model']['vq.ema_w_11'] = tokenizer['model'].pop('vq.ema_w_0')
                elif mode == 'alias':
                    generator['model']['g.1'] = generator['model']['g.0']
                else:
                    generator['extra'] = {}
                with self.assertRaises(ValueError):
                    checkpoint_parts(generator, tokenizer)


if __name__ == '__main__':
    unittest.main()
