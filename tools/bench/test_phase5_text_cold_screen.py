"""Offline controls for cold text/decode measurements."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import phase5_text_cold_screen as tool


class TextColdScreenTests(unittest.TestCase):
    def test_rotation_and_arguments(self):
        for group in range(1, 4):
            for ordinal in range(35):
                self.assertEqual(set(tool.order(group, ordinal)), set(tool.CONDITIONS))
        profile = {'name': 'xsum', 'device': 'cuda-cudnn'}
        args = [Path('/old'), Path('/new'), Path('/art'), Path('/ids'), Path('/text')]
        self.assertEqual(tool.command('old_id', profile, *args), ['/old', 'sample-xsum', '/art', 'cuda-cudnn', '/ids'])
        self.assertEqual(tool.command('new_text_decode', profile, *args), ['/new', 'sample', '/art', 'cuda-cudnn', '/text', '--text', '--decode'])
        with self.assertRaises(ValueError):
            tool.command('unknown', profile, *args)

    def test_summary_requires_all_samples_and_excludes_warmups(self):
        rows = [{'group': group, 'ordinal': ordinal, 'condition': condition, 'warmup': ordinal < 5,
                 'elapsed_seconds': float(ordinal + 1), 'max_rss_kib': 100 + ordinal}
                for group in range(1, 4) for ordinal in range(35) for condition in tool.CONDITIONS]
        result = tool.summarize(rows)
        self.assertEqual(result['new_text']['samples'], 90)
        self.assertEqual(result['new_text']['median_seconds'], 20.5)
        self.assertEqual(result['new_text']['p95_seconds'], 34.)
        rows.pop()
        with self.assertRaisesRegex(ValueError, 'missing'):
            tool.summarize(rows)

    def test_execute_checks_output_and_rss(self):
        raw = b'expected'
        def launch(argv, **kwargs):
            self.assertTrue(kwargs['start_new_session'])
            Path(argv[4]).write_text('1234\n')
            kwargs['stdout'].write(raw)
            return Mock(wait=Mock(return_value=0))
        with tempfile.TemporaryDirectory() as directory, patch.object(tool.subprocess, 'Popen', side_effect=launch):
            result = tool.execute(Path(directory), 'old_id', ['/bin/ncm'], 1, 0, hashlib.sha256(raw).hexdigest())
            self.assertEqual(result['max_rss_kib'], 1234)
            self.assertTrue(result['warmup'])
        with tempfile.TemporaryDirectory() as directory, patch.object(tool.subprocess, 'Popen', side_effect=launch):
            with self.assertRaisesRegex(ValueError, 'output differs'):
                tool.execute(Path(directory), 'old_id', ['/bin/ncm'], 1, 0, '0' * 64)


if __name__ == '__main__':
    unittest.main()
