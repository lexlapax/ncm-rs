"""Fail-closed output preflight for retained LM1B decode references."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import phase5_lm1b_retained_decode as tool


class OutputPreflightTests(unittest.TestCase):
    def test_existing_and_symlinked_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / 'existing'
            existing.write_bytes(b'unchanged')
            live = root / 'live'
            live.symlink_to(existing)
            dangling = root / 'dangling'
            dangling.symlink_to(root / 'missing')
            for output in (existing, live, dangling):
                with patch('sys.argv', ['decode', 'small', str(output)]):
                    with self.assertRaisesRegex(ValueError, 'output must be new'):
                        tool.main()
            self.assertEqual(existing.read_bytes(), b'unchanged')
            self.assertFalse((root / 'missing').exists())


if __name__ == '__main__':
    unittest.main()
