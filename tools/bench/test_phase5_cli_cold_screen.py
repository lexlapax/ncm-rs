"""Offline checks for the Phase 5 cold-request screen."""
import hashlib
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import phase5_cli_cold_screen as tool


class ColdScreenTests(unittest.TestCase):
    def test_summary_excludes_warmups_and_requires_full_sample_count(self):
        rows = [{'condition': 'baseline', 'warmup': False, 'elapsed_seconds': float(i + 1),
                 'max_rss_kib': 100 + i} for i in range(90)]
        rows += [{'condition': 'baseline', 'warmup': True, 'elapsed_seconds': 9999., 'max_rss_kib': 9999}]
        result = tool.summarize(rows, 'baseline')
        self.assertEqual(result['samples'], 90)
        self.assertEqual(result['median_seconds'], 45.5)
        self.assertEqual(result['p95_seconds_nearest_rank'], 86.)
        self.assertEqual(result['max_rss_kib'], 189)
        with self.assertRaisesRegex(ValueError, 'missing'):
            tool.summarize(rows[:-2], 'baseline')

    def test_execute_validates_output_and_rss(self):
        raw = b'{"ids":[]}'
        expected = hashlib.sha256(raw).hexdigest()
        def run(argv, **kwargs):
            self.assertTrue(kwargs['start_new_session'])
            Path(argv[4]).write_text('1234\n', encoding='utf-8')
            kwargs['stdout'].write(raw)
            return Mock(wait=Mock(return_value=0))
        with tempfile.TemporaryDirectory() as directory, patch.object(tool.subprocess, 'Popen', side_effect=run):
            result = tool.execute(Path(directory), 'candidate', Path('/bin/ncm'), 'sample',
                                  Path('/artifact'), Path('/input'), 1, 0, True, expected)
            self.assertEqual(result['max_rss_kib'], 1234)
            self.assertTrue(result['warmup'])
            self.assertEqual(result['output_sha256'], expected)

    def test_tinystories_uses_matching_seed_and_cudnn_device(self):
        raw = b'{}'
        expected = hashlib.sha256(raw).hexdigest()
        calls = []
        def run(argv, **kwargs):
            calls.append(argv)
            Path(argv[4]).write_text('1234\n', encoding='utf-8')
            kwargs['stdout'].write(raw)
            return Mock(wait=Mock(return_value=0))
        with tempfile.TemporaryDirectory() as directory, patch.object(tool.subprocess, 'Popen', side_effect=run):
            tool.execute(Path(directory), 'baseline', Path('/old'), 'sample-tinystories',
                         Path('/artifact'), Path('/input'), 1, 0, True, expected, 'tinystories')
            tool.execute(Path(directory), 'candidate', Path('/new'), 'sample',
                         Path('/artifact'), Path('/input'), 1, 0, True, expected, 'tinystories')
        self.assertEqual(calls[0][5:], ['/old', 'sample-tinystories', '/artifact', 'cuda-cudnn', '/input', '42'])
        self.assertEqual(calls[1][5:], ['/new', 'sample', '/artifact', 'cuda-cudnn', '/input', '--seed', '42'])

    def test_top1_text_profiles_use_correct_device_without_sampling_flags(self):
        raw = b'{}'
        expected = hashlib.sha256(raw).hexdigest()
        calls = []
        def run(argv, **kwargs):
            calls.append(argv)
            Path(argv[4]).write_text('1234\n', encoding='utf-8')
            kwargs['stdout'].write(raw)
            return Mock(wait=Mock(return_value=0))
        with tempfile.TemporaryDirectory() as directory, patch.object(tool.subprocess, 'Popen', side_effect=run):
            for profile, alias in (('xsum', 'sample-xsum'), ('wmt14', 'sample-wmt14')):
                tool.execute(Path(directory), profile, Path('/old'), alias, Path('/artifact'),
                             Path('/input'), 1, 0, True, expected, profile)
        self.assertEqual(calls[0][5:], ['/old', 'sample-xsum', '/artifact', 'cuda-cudnn', '/input'])
        self.assertEqual(calls[1][5:], ['/old', 'sample-wmt14', '/artifact', 'cuda', '/input'])

    def test_lm1b_uses_matching_seed_and_explicit_rotary_path(self):
        raw = b'{}'
        expected = hashlib.sha256(raw).hexdigest()
        calls = []
        def run(argv, **kwargs):
            calls.append(argv)
            Path(argv[4]).write_text('1234\n', encoding='utf-8')
            kwargs['stdout'].write(raw)
            return Mock(wait=Mock(return_value=0))
        with tempfile.TemporaryDirectory() as directory, patch.object(tool.subprocess, 'Popen', side_effect=run):
            for condition, binary, command in (('baseline', '/old', 'sample-lm1b'), ('candidate', '/new', 'sample')):
                tool.execute(Path(directory), condition, Path(binary), command, Path('/artifact'),
                             Path('/input'), 1, 0, True, expected, 'lm1b', Path('/rope'))
        self.assertEqual(calls[0][5:], ['/old', 'sample-lm1b', '/artifact', 'cuda-cudnn', '/input', '42', '/rope'])
        self.assertEqual(calls[1][5:], ['/new', 'sample', '/artifact', 'cuda-cudnn', '/input', '--seed', '42', '--reference-rope', '/rope'])

    def test_dangling_output_symlink_rejected_before_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'link'
            output.symlink_to(root / 'missing', target_is_directory=True)
            argv = ['phase5_cli_cold_screen', '--baseline', 'old', '--candidate', 'new',
                    '--artifact', 'artifact', '--input', 'input', '--output', str(output),
                    '--baseline-sha256', 'a', '--candidate-sha256', 'b', '--manifest-sha256', 'c',
                    '--input-sha256', 'd', '--expected-output-sha256', 'e']
            with patch('sys.argv', argv), self.assertRaisesRegex(ValueError, 'output already exists'):
                tool.main()
            self.assertFalse((root / 'missing').exists())

    def test_lm1b_requires_both_rotary_arguments_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = ['phase5_cli_cold_screen', '--baseline', 'old', '--candidate', 'new',
                    '--artifact', 'artifact', '--input', 'input', '--output', str(Path(directory) / 'out'),
                    '--baseline-sha256', 'a', '--candidate-sha256', 'b', '--manifest-sha256', 'c',
                    '--input-sha256', 'd', '--expected-output-sha256', 'e', '--profile', 'lm1b']
            for extra in ([], ['--rope', 'rope'], ['--rope-sha256', 'digest']):
                with patch('sys.argv', argv + extra), patch.object(tool, 'digest') as digest:
                    with self.assertRaisesRegex(ValueError, 'requires both'):
                        tool.main()
                    digest.assert_not_called()

    def test_execute_rejects_failure_and_mismatched_output(self):
        def run(argv, **kwargs):
            Path(argv[4]).write_text('1234\n', encoding='utf-8')
            kwargs['stdout'].write(b'wrong')
            return Mock(wait=Mock(return_value=0))
        with tempfile.TemporaryDirectory() as directory, patch.object(tool.subprocess, 'Popen', side_effect=run):
            with self.assertRaisesRegex(ValueError, 'output hash'):
                tool.execute(Path(directory), 'baseline', Path('/bin/ncm'), 'generate',
                             Path('/artifact'), Path('/input'), 1, 0, False, '0' * 64)
        with tempfile.TemporaryDirectory() as directory, patch.object(
                tool.subprocess, 'Popen', return_value=Mock(wait=Mock(return_value=2))):
            with self.assertRaisesRegex(RuntimeError, 'exit 2'):
                tool.execute(Path(directory), 'baseline', Path('/bin/ncm'), 'generate',
                             Path('/artifact'), Path('/input'), 1, 0, False, '0' * 64)

    def test_timeout_kills_only_the_spawned_process_group(self):
        process = Mock(pid=123, wait=Mock(side_effect=[subprocess.TimeoutExpired('time', 900), -9]))
        with tempfile.TemporaryDirectory() as directory, patch.object(
                tool.subprocess, 'Popen', return_value=process), patch.object(tool.os, 'killpg') as kill:
            with self.assertRaises(subprocess.TimeoutExpired):
                tool.execute(Path(directory), 'baseline', Path('/bin/ncm'), 'generate',
                             Path('/artifact'), Path('/input'), 1, 0, False, '0' * 64)
        kill.assert_called_once_with(123, signal.SIGKILL)
        self.assertEqual(process.wait.call_count, 2)


if __name__ == '__main__':
    unittest.main()
