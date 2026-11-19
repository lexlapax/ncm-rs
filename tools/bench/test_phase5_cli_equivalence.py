"""Negative controls for the Phase 5 fixed-request comparator."""
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import phase5_cli_equivalence as runner


class EquivalenceTests(unittest.TestCase):
    def make_output(self, codes=200, ids=48):
        return json.dumps({'codes': [[0] * codes], 'ids': [[1] * ids]}, separators=(',', ':')).encode() + b'\n'

    def test_shape_validation_rejects_missing_empty_and_bad_values(self):
        good = self.make_output()
        runner.validate_output(good, 1, 200, 48)
        for bad in (b'{}', self.make_output(codes=0), b'{"codes":[[true]],"ids":[[1]]}', b'[]'):
            with self.subTest(bad=bad), self.assertRaises((ValueError, json.JSONDecodeError)):
                runner.validate_output(bad, 1, 200, 48)
        with self.assertRaises(ValueError):
            runner.validate_output(good, 1, 200, 48, seeded=True)

    def test_three_independent_commands_compare_bytes(self):
        good = self.make_output()
        captures = [subprocess.CompletedProcess([], 0, good, b'') for _ in range(3)]
        with tempfile.TemporaryDirectory() as directory, patch.object(runner.subprocess, 'run', side_effect=captures) as call:
            result = runner.compare_case('xsum', Path('/old'), Path('/new'), Path('/artifact'),
                                         'cuda-cudnn', Path('/input'), Path('/rope'), Path(directory), 1)
            self.assertEqual(call.call_count, 3)
            self.assertEqual([cmd[1] for cmd in result['commands']], ['sample-xsum', 'sample-xsum', 'sample'])
            self.assertEqual((Path(directory) / 'xsum-unified.stdout').read_bytes(), good)

    def test_identical_binary_identity_does_not_create_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / 'first', root / 'second'
            first.write_bytes(b'same')
            second.write_bytes(b'same')
            digest = hashlib.sha256(b'same').hexdigest()
            argv = ['phase5_cli_equivalence', '--baseline', str(first), '--candidate', str(second),
                    '--baseline-sha256', digest, '--candidate-sha256', digest,
                    '--artifacts', str(root), '--inputs', str(root), '--xsum-input', str(root / 'xsum'),
                    '--wmt14-input', str(root / 'wmt'), '--rope', str(root / 'rope'),
                    '--platform', 'cuda', '--output', str(root / 'output')]
            with patch('sys.argv', argv), self.assertRaisesRegex(ValueError, 'distinct binaries'):
                runner.main()
            self.assertFalse((root / 'output').exists())

    def test_output_symlinks_are_rejected_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'target'
            target.mkdir()
            live, dangling = root / 'live', root / 'dangling'
            live.symlink_to(target, target_is_directory=True)
            dangling.symlink_to(root / 'missing', target_is_directory=True)
            for output in (live, dangling):
                argv = ['phase5_cli_equivalence', '--baseline', 'old', '--candidate', 'new',
                        '--baseline-sha256', 'a', '--candidate-sha256', 'b', '--artifacts', 'artifacts',
                        '--inputs', 'inputs', '--xsum-input', 'xsum', '--wmt14-input', 'wmt',
                        '--rope', 'rope', '--platform', 'cuda', '--output', str(output)]
                with patch('sys.argv', argv), patch.object(runner, 'digest') as digest:
                    with self.assertRaisesRegex(ValueError, 'output directory already exists'):
                        runner.main()
                    digest.assert_not_called()
            self.assertFalse((root / 'missing').exists())
            self.assertEqual(list(target.iterdir()), [])

    def test_differences_and_process_failures_do_not_pass(self):
        good = self.make_output()
        different = good.replace(b'[[1,', b'[[2,', 1)
        for outcomes in ([subprocess.CompletedProcess([], 0, good, b'')] * 2
                         + [subprocess.CompletedProcess([], 0, different, b'')],
                         [subprocess.CompletedProcess([], 7, b'', b'failure')]):
            with tempfile.TemporaryDirectory() as directory, patch.object(runner.subprocess, 'run', side_effect=outcomes):
                with self.assertRaises((ValueError, RuntimeError)):
                    runner.compare_case('xsum', Path('/old'), Path('/new'), Path('/artifact'),
                                        'cuda-cudnn', Path('/input'), Path('/rope'), Path(directory), 1)
                self.assertFalse((Path(directory) / 'result.json').exists())


if __name__ == '__main__':
    unittest.main()
