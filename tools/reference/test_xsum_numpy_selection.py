"""Compare the NumPy XSum penalty implementation with the pinned upstream function."""
import ast
import hashlib
import unittest
from pathlib import Path

import numpy as np
import torch
from xsum_numpy_selection import select


class XsumSelectionTest(unittest.TestCase):
    def test_matches_upstream_with_repeats_and_ties(self):
        path = Path(__file__).resolve().parents[2] / '.reference/upstream-ncm/scripts/eval_shard.py'
        if not path.exists():
            self.skipTest('pinned upstream checkout unavailable')
        raw = path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), 'e1eb550123ec76601d1af9c59c2bb7a81d5ca683a662948fcb4d8164dd510c4a')
        tree = ast.parse(raw)
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'pick_tokens']
        self.assertEqual(len(funcs), 1)
        rng = np.random.default_rng(79)
        logits = rng.normal(size=(4, 16, 13)).astype(np.float32)
        logits[0, 0, :] = 0
        logits[1, :, 2] = 4
        allowance = rng.random(13).astype(np.float32)
        allowance /= allowance.sum()
        scope = {'torch': torch, 'freq': 2.0, 'mu': 1.0, 'ADJ': [5.0, 4, 0.5],
                 'ALLOW': torch.from_numpy(allowance).double()}
        exec(compile(ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[])), str(path), 'exec'), scope)
        expected = scope['pick_tokens'](torch.from_numpy(logits)).numpy()
        np.testing.assert_array_equal(select(logits, allowance), expected)
        with self.assertRaises(ValueError):
            select(np.full((1, 1, 13), np.nan, dtype=np.float32), allowance)


if __name__ == '__main__':
    unittest.main()
