"""Tests for the bounded Phase 3 evaluation-data conversion."""
import hashlib
import pickle
import tempfile
import unittest
from pathlib import Path
from sudoku_phase3_data import read

class Phase3DataTests(unittest.TestCase):
    def test_hash_and_non_executable_tokens(self):
        grid = [x for row in range(9) for x in ([1] * 9 + ([10] if row < 8 else []))]
        board = [11] + grid + [11]
        raw = pickle.dumps([board[:] for _ in range(2000)], protocol=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "boards"
            path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            self.assertEqual(len(read(path, digest, len(raw), 91)), 2000)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                read(path, "0" * 64, len(raw), 91)
            path.write_bytes(pickle.dumps([set([1])], protocol=4))
            unsafe = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "opcode"):
                read(path, hashlib.sha256(unsafe).hexdigest(), len(unsafe), 91)

if __name__ == "__main__": unittest.main()
