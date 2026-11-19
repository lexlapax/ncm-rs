"""Offline output-safety checks for the LM1B rotary reference exporter."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lm1b_reference_rope_diagnostic as tool


class RotaryOutputTests(unittest.TestCase):
    def test_exclusive_write_and_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'rope.safetensors'
            tool.write_exclusive(output, b'first')
            with self.assertRaises(FileExistsError):
                tool.write_exclusive(output, b'second')
            self.assertEqual(output.read_bytes(), b'first')

    def test_exclusive_write_rejects_live_and_dangling_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'target'
            target.write_bytes(b'original')
            for name, destination in (('live', target), ('dangling', root / 'missing')):
                output = root / name
                output.symlink_to(destination)
                with self.assertRaises(FileExistsError):
                    tool.write_exclusive(output, b'replacement')
                self.assertTrue(output.is_symlink())
            self.assertEqual(target.read_bytes(), b'original')
            self.assertFalse((root / 'missing').exists())

    def test_main_rejects_dangling_output_before_upstream_or_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'link'
            output.symlink_to(root / 'missing')
            with patch('sys.argv', ['rope', '--upstream', str(root / 'missing-upstream'),
                                    '--output', str(output)]):
                with self.assertRaisesRegex(ValueError, 'output already exists'):
                    tool.main()
            self.assertFalse((root / 'missing').exists())


if __name__ == '__main__':
    unittest.main()
