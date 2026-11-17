"""Check capture admission fails before starting a child process."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import capture_wmt14_cuda_full as capture


class FullCaptureTests(unittest.TestCase):
    def test_existing_output_never_spawns(self):
        with tempfile.TemporaryDirectory(dir=capture.ROOT / 'reports') as directory:
            args = ['capture', '--mode', 'base', '--head', 'x', '--binary-sha256', 'x',
                    '--protocol-sha256', 'x', '--rows', '35', '--output', directory]
            with patch.object(sys, 'argv', args), patch.object(capture.subprocess, 'run') as child:
                with self.assertRaises(ValueError):
                    capture.main()
                child.assert_not_called()

    def test_manifest_creation_does_not_overwrite(self):
        with tempfile.TemporaryDirectory(dir=capture.ROOT / 'reports') as directory:
            target = Path(directory) / 'receipt.json'
            capture.checked_json(target, {'value': 1})
            with self.assertRaises(FileExistsError):
                capture.checked_json(target, {'value': 2})
            self.assertIn('1', target.read_text())


if __name__ == '__main__':
    unittest.main()
