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

    def test_artifact_comparison_pairs_same_binary_and_distinct_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary, inp = root / 'ncm', root / 'input'
            binary.write_bytes(b'binary'); inp.write_bytes(b'input')
            original, alternate = root / 'original', root / 'alternate'
            original.mkdir(); alternate.mkdir()
            (original / 'manifest.json').write_bytes(b'original')
            (alternate / 'manifest.json').write_bytes(b'alternate')
            sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            argv = ['screen', '--baseline', str(binary), '--candidate', str(binary),
                    '--baseline-sha256', sha(binary), '--candidate-sha256', sha(binary),
                    '--baseline-artifact', str(original), '--baseline-manifest-sha256', sha(original / 'manifest.json'),
                    '--artifact', str(alternate), '--manifest-sha256', sha(alternate / 'manifest.json'),
                    '--input', str(inp), '--input-sha256', sha(inp), '--expected-output-sha256', 'unused',
                    '--output', str(root / 'out')]
            monitor = Mock(poll=Mock(return_value=None), wait=Mock(return_value=0), returncode=0)
            def start_monitor(_argv, **kwargs):
                kwargs['stdout'].write(b'telemetry\n'); kwargs['stdout'].flush()
                return monitor
            def execute(_output, condition, _binary, command, artifact, *_args):
                calls.append((condition, command, artifact))
                return {'condition': condition}
            calls = []
            with patch('sys.argv', argv), patch.object(tool.shutil, 'which', return_value='/usr/bin/true'), \
                 patch.object(tool.subprocess, 'Popen', side_effect=start_monitor), \
                 patch.object(tool, 'execute', side_effect=execute), patch.object(tool, 'summarize', return_value={}), \
                 patch.object(tool, 'ROUNDS', 1), patch.object(tool, 'WARMUPS', 0), patch.object(tool, 'TRIALS', 1):
                tool.main()
            self.assertEqual(set(calls), {('baseline', 'sample', original), ('candidate', 'sample', alternate)})
            self.assertIn('"comparison": "artifact"', (root / 'out' / 'result.json').read_text())

    def test_artifact_comparison_rejects_partial_and_non_sudoku_options(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = ['screen', '--baseline', 'old', '--candidate', 'new', '--artifact', 'artifact',
                    '--input', 'input', '--output', str(Path(directory) / 'out'),
                    '--baseline-sha256', 'a', '--candidate-sha256', 'b', '--manifest-sha256', 'c',
                    '--input-sha256', 'd', '--expected-output-sha256', 'e', '--baseline-artifact', 'other']
            with patch('sys.argv', argv), self.assertRaisesRegex(ValueError, 'supplied together'):
                tool.main()
            with patch('sys.argv', argv + ['--baseline-manifest-sha256', 'f', '--profile', 'xsum']), \
                 self.assertRaisesRegex(ValueError, 'only for Sudoku'):
                tool.main()

    def test_artifact_comparison_rejects_same_identity_or_different_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = ['screen', '--baseline', 'ncm', '--baseline-sha256', 'binary',
                    '--candidate', 'ncm', '--candidate-sha256', 'binary', '--artifact', 'new',
                    '--manifest-sha256', 'newhash', '--input', 'input', '--output', str(root / 'out'),
                    '--input-sha256', 'input', '--expected-output-sha256', 'output']
            for extra in (
                ['--baseline-artifact', 'new', '--baseline-manifest-sha256', 'oldhash'],
                ['--baseline-artifact', 'old', '--baseline-manifest-sha256', 'newhash'],
                ['--baseline-artifact', 'old', '--baseline-manifest-sha256', 'oldhash', '--candidate', 'other'],
                ['--baseline-artifact', 'old', '--baseline-manifest-sha256', 'oldhash', '--candidate-sha256', 'other'],
            ):
                with patch('sys.argv', base + extra), patch.object(tool, 'digest') as digest:
                    with self.assertRaisesRegex(ValueError, 'same binary and distinct'):
                        tool.main()
                    digest.assert_not_called()

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
