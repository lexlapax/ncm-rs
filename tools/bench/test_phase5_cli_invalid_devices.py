"""Negative controls for the Phase 5 invalid-device check."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import phase5_cli_invalid_devices as tool


class InvalidDeviceTests(unittest.TestCase):
    def test_rejection_requires_exit_one_empty_stdout_and_expected_error(self):
        good = subprocess.CompletedProcess([], 1, b'', b'ncm: unavailable\n')
        bad = [subprocess.CompletedProcess([], 0, b'', b'unavailable'),
               subprocess.CompletedProcess([], 1, b'output', b'unavailable'),
               subprocess.CompletedProcess([], 1, b'', b'wrong error')]
        with patch.object(tool.subprocess, 'run', return_value=good) as call:
            result = tool.check_case(Path('/cli'), Path('/artifact'), Path('/input'), b'unavailable')
        self.assertEqual(result['exit'], 1)
        self.assertEqual(call.call_args.args[0][3], tool.INVALID_DEVICE)
        for response in bad:
            with self.subTest(response=response), patch.object(tool.subprocess, 'run', return_value=response):
                with self.assertRaisesRegex(ValueError, 'unexpected rejection'):
                    tool.check_case(Path('/cli'), Path('/artifact'), Path('/input'), b'unavailable')

    def test_dangling_output_rejected_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'link'
            output.symlink_to(root / 'missing')
            argv = ['invalid-devices', '--binary', 'cli', '--binary-sha256', 'a',
                    '--artifacts', 'artifacts', '--inputs', 'inputs', '--xsum-input', 'xsum',
                    '--wmt14-input', 'wmt', '--tiny-input', 'tiny', '--lm1b-input', 'lm1b',
                    '--output', str(output)]
            with patch('sys.argv', argv), patch.object(tool, 'digest') as digest:
                with self.assertRaisesRegex(ValueError, 'output already exists'):
                    tool.main()
                digest.assert_not_called()
            self.assertFalse((root / 'missing').exists())


if __name__ == '__main__':
    unittest.main()
