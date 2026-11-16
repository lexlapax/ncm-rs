"""Verify that native-library provenance rejects missing or ambiguous links."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import xsum_linked_libraries as links


class LinkedLibraryTests(unittest.TestCase):
    def test_requires_each_unique_resolved_library(self):
        Path('reports').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir='reports') as directory:
            root = Path(directory).resolve()
            lines = []
            for name in sorted(links.REQUIRED):
                path = root / name
                path.write_bytes(name.encode())
                lines.append(f'{name} => {path} (0x0)')
            with patch.object(links.subprocess, 'check_output', return_value='\n'.join(lines)):
                self.assertEqual(set(links.linked_libraries(root / 'binary', root)), links.REQUIRED)
            for bad in (lines[:-1], lines + [lines[0]], lines[:-1] + [f'{sorted(links.REQUIRED)[-1]} => not found']):
                with self.subTest(bad=bad), patch.object(links.subprocess, 'check_output', return_value='\n'.join(bad)):
                    with self.assertRaises(ValueError):
                        links.linked_libraries(root / 'binary', root)


if __name__ == '__main__':
    unittest.main()
