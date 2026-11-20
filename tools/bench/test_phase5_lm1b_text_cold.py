"""Offline controls for the LM1B cold CLI comparison."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import phase5_lm1b_text_cold as tool


class Lm1bTextColdTests(unittest.TestCase):
    def test_explicit_baseline_and_default_candidate_commands(self):
        args = [Path('/old'), Path('/new'), Path('/artifact'), Path('/rope'), Path('/ids'), Path('/text')]
        self.assertEqual(tool.command('old_id', *args), ['/old', 'sample-lm1b', '/artifact', 'cuda-cudnn', '/ids', '42', '/rope'])
        self.assertEqual(tool.command('new_id', *args), ['/new', 'sample', '/artifact', 'cuda-cudnn', '/ids', '--seed', '42'])
        self.assertEqual(tool.command('new_text_decode', *args), ['/new', 'sample', '/artifact', 'cuda-cudnn', '/text', '--seed', '42', '--text', '--decode'])
        with self.assertRaises(ValueError):
            tool.command('invalid', *args)

    def test_existing_output_rejected_before_hashing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); link = root / 'link'; link.symlink_to(root / 'missing')
            argv = ['screen', '--size', 'small', '--baseline', 'old', '--candidate', 'new',
                    '--artifact-root', 'artifacts', '--output', str(link)]
            with patch('sys.argv', argv), patch.object(tool, 'digest') as digest:
                with self.assertRaisesRegex(ValueError, 'output exists'):
                    tool.main()
                digest.assert_not_called()


if __name__ == '__main__':
    unittest.main()
