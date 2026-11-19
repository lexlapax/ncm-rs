"""Negative controls for TinyStories reference input preflight."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import phase5_tinystories_text_reference as tool


class ReferencePreflightTests(unittest.TestCase):
    def test_checked_rejects_wrong_hash_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / 'input'
            original.write_bytes(b'abc')
            digest = hashlib.sha256(b'abc').hexdigest()
            self.assertEqual(tool.checked(original, digest), b'abc')
            with self.assertRaisesRegex(ValueError, 'hash'):
                tool.checked(original, '0' * 64)
            link = root / 'link'
            link.symlink_to(original)
            with self.assertRaisesRegex(ValueError, 'regular'):
                tool.checked(link, digest)

    def test_preflight_rejects_wrong_revision_dirty_checkout_and_version(self):
        args = (Path('/upstream'), Path('/payload'), Path('/manifest'))
        with patch.object(tool.subprocess, 'check_output', return_value='wrong\n'):
            with self.assertRaisesRegex(ValueError, 'revision'):
                tool.preflight(*args)
        with patch.object(tool.subprocess, 'check_output', side_effect=[tool.UPSTREAM, ' M file\n']):
            with self.assertRaisesRegex(ValueError, 'not clean'):
                tool.preflight(*args)
        with patch.object(tool.subprocess, 'check_output', side_effect=[tool.UPSTREAM, '']), \
                patch.object(tool, 'checked', return_value=b'asset'), \
                patch.object(tool.importlib.metadata, 'version', return_value='wrong'):
            with self.assertRaisesRegex(ValueError, 'version'):
                tool.preflight(*args)


if __name__ == '__main__':
    unittest.main()
