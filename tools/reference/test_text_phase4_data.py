"""Focused safety checks for pinned text-data conversion."""
import hashlib
import pickle
import tempfile
import unittest
from pathlib import Path

from text_phase4_data import read


class RestrictedDataTests(unittest.TestCase):
    def check(self, raw):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'input.pickle'
            path.write_bytes(raw)
            return read(path, len(raw), hashlib.sha256(raw).hexdigest())

    def test_basic_lists_and_strings(self):
        self.assertEqual(self.check(pickle.dumps(['one', 'two'], protocol=4)), ['one', 'two'])
        self.assertEqual(self.check(pickle.dumps([[1, 256], [2, 257]], protocol=4)), [[1, 256], [2, 257]])

    def test_forbids_global_and_reduce_opcodes(self):
        with self.assertRaises(ValueError):
            self.check(b'cbuiltins\nlist\n)R.')

    def test_forbids_trailing_bytes(self):
        with self.assertRaises(ValueError):
            self.check(pickle.dumps(['one'], protocol=4) + b'trailing')


if __name__ == '__main__':
    unittest.main()
