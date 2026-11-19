"""Output preflight checks for synthetic tokenizer boundary references."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import phase5_text_boundaries as tool


class BoundaryPreflightTests(unittest.TestCase):
    def test_existing_and_symlinked_outputs_fail_before_loading_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / 'existing'
            existing.write_bytes(b'unchanged')
            live = root / 'live'
            live.symlink_to(existing)
            dangling = root / 'dangling'
            dangling.symlink_to(root / 'missing')
            for output in (existing, live, dangling):
                with patch('sys.argv', ['boundaries', 'xsum', str(output)]):
                    with self.assertRaisesRegex(ValueError, 'output already exists'):
                        tool.main()
            self.assertEqual(existing.read_bytes(), b'unchanged')
            self.assertFalse((root / 'missing').exists())


if __name__ == '__main__':
    unittest.main()
