"""Check candidate capture admission and receipt handling."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import capture_wmt14_reference_candidate as capture


class CandidateCaptureTests(unittest.TestCase):
    def test_existing_output_does_not_spawn(self):
        with tempfile.TemporaryDirectory(dir=capture.ROOT / 'reports') as directory:
            args = ['capture', '--mode', 'base', '--head', 'x', '--binary-sha256', 'x',
                    '--protocol-sha256', 'x', '--rows', '35', '--output', directory]
            with patch.object(sys, 'argv', args), patch.object(capture.subprocess, 'run') as child:
                with self.assertRaises(ValueError):
                    capture.main()
                child.assert_not_called()

    def test_receipt_does_not_overwrite(self):
        with tempfile.TemporaryDirectory(dir=capture.ROOT / 'reports') as directory:
            path = Path(directory) / 'receipt.json'
            capture.write_json(path, {'value': 1})
            with self.assertRaises(FileExistsError):
                capture.write_json(path, {'value': 2})


if __name__ == '__main__':
    unittest.main()
