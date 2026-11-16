"""Validate unigram inventory shape and detection without checkpoint files."""
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import lm1b_unigram_inspect as subject


def checkpoint():
    return {'model': {}, 'inference': {'level0_unigram': torch.zeros(8192),
                                       'token_unigram': torch.ones(30528)}}


class Lm1bUnigramTests(unittest.TestCase):
    def test_reports_values_and_does_not_hide_negatives(self):
        obj = checkpoint()
        obj['inference']['level0_unigram'][0] = -1
        with patch.object(subject, 'verify_input'), patch.object(subject.torch, 'load', return_value=obj):
            result = subject.inspect(Path('.'))
        for row in result.values():
            self.assertEqual(row['level0_unigram']['negative_count'], 1)
            self.assertEqual(row['level0_unigram']['zero_count'], 8191)
            self.assertEqual(row['token_unigram']['sum_f64'], 30528.0)

    def test_rejects_wrong_shape(self):
        obj = checkpoint()
        obj['inference']['token_unigram'] = torch.ones(2)
        with patch.object(subject, 'verify_input'), patch.object(subject.torch, 'load', return_value=obj):
            with self.assertRaises(ValueError):
                subject.inspect(Path('.'))


if __name__ == '__main__':
    unittest.main()
