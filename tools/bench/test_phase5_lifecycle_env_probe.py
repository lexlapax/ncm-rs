"""Privacy and preflight checks for the fake lifecycle diagnostic."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import phase5_lifecycle_env_probe as probe


class LifecycleEnvProbeTests(unittest.TestCase):
    def test_categories_never_serialize_unexpected_keys_or_values(self):
        expected = {b'LANG': b'C', b'PATH': b'expected-secret'}
        actual = {b'LANG': b'C', b'PATH': b'other-secret', b'UNEXPECTED_SECRET': b'hidden'}
        result = probe.categories(actual, expected, actual)
        self.assertEqual(result['missing_count'], 0)
        self.assertEqual(result['extra_count'], 1)
        self.assertTrue(result['same_as_parent_proc'])
        self.assertFalse(result['known_key_values_match']['PATH'])
        encoded = json.dumps(result)
        for value in ('expected-secret', 'other-secret', 'UNEXPECTED_SECRET', 'hidden'):
            self.assertNotIn(value, encoded)

    def test_existing_symlink_rejected_before_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'link'
            output.symlink_to(root / 'missing')
            with patch('sys.argv', ['probe', '--output', str(output)]), patch.object(probe.subprocess, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'new output'):
                    probe.main()
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
