"""Output preflight for the TinyStories synthetic decoder reference."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import phase5_tinystories_decode_reference as tool


class DecodePreflightTests(unittest.TestCase):
    def test_existing_and_symlinked_outputs_rejected_before_input_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / 'existing'
            existing.write_bytes(b'unchanged')
            live, dangling = root / 'live', root / 'dangling'
            live.symlink_to(existing)
            dangling.symlink_to(root / 'missing')
            for output in (existing, live, dangling):
                argv = ['tiny-decode', '--upstream', str(root / 'no-upstream'),
                        '--tokenizer-asset', str(root / 'no-asset'), '--output', str(output)]
                with patch('sys.argv', argv), self.assertRaisesRegex(ValueError, 'output already exists'):
                    tool.main()
            self.assertEqual(existing.read_bytes(), b'unchanged')
            self.assertFalse((root / 'missing').exists())


if __name__ == '__main__':
    unittest.main()
