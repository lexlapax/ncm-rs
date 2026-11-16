"""Checks for the frozen XSum quality comparison rule."""
import copy
import unittest

from xsum_quality_compare import MODEL_REV, compare


class QualityComparison(unittest.TestCase):
    def setUp(self):
        self.python = {'schema_version': 2, 'scope': 'pinned-upstream-cuda XSum outputs',
                       'model_revision': MODEL_REV, 'scorer_sha256': 'test', 'start': 256, 'stop': 257,
                       'versions': {'test': '1'},
                       'shard_manifests': [{'origin': 'pinned-upstream-cuda', 'shard': i,
                                            'gpu_identity_and_driver': 'test'} for i in range(2)],
                       'metrics': {'rouge1': 1.0, 'rouge2': 2.0, 'rougeL': 3.0, 'mean_words': 4.0,
                                   'generation_perplexity': 5.0, 'generation_ppl_token_count': 6},
                       'rows': [{'index': 256, 'codes': [0] * 200, 'raw_ids': [1] * 48,
                                 'hypothesis': 'test', 'rouge': {'rouge1': 1.0}}]}
        self.rust = copy.deepcopy(self.python)
        self.rust['scope'] = 'rust-strict-cudnn XSum outputs'
        for manifest in self.rust['shard_manifests']: manifest['origin'] = 'rust-strict-cudnn'

    def test_exact_and_changed_outputs(self):
        self.assertTrue(compare(self.python, self.rust, 'test')['quality_rule_passed'])
        self.rust['rows'][0]['codes'][0] = 2
        result = compare(self.python, self.rust, 'test')
        self.assertTrue(result['quality_rule_passed'])
        self.assertEqual(result['code_differences'], 1)
        self.rust['rows'][0]['raw_ids'][0] = 2
        self.assertFalse(compare(self.python, self.rust, 'test')['quality_rule_passed'])

    def test_rejects_mismatched_evaluator_or_rows(self):
        self.rust['versions'] = {'test': 'changed'}
        with self.assertRaises(ValueError): compare(self.python, self.rust, 'test')
        self.rust['versions'] = self.python['versions']
        self.rust['rows'][0]['index'] = 257
        with self.assertRaises(ValueError): compare(self.python, self.rust, 'test')


if __name__ == '__main__': unittest.main()
