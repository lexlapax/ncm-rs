"""Tests for the offline, separately versioned rotary reference."""
import json
import tempfile
import unittest
from pathlib import Path

import xsum_rope_reference as reference


class RotaryReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = reference.ROOT / '.reference/upstream-ncm'
        if not cls.upstream.is_dir():
            raise unittest.SkipTest('pinned upstream checkout is not available')
        (reference.ROOT / 'reports').mkdir(exist_ok=True)

    def test_pinned_generation_and_existing_output_rejection(self):
        with tempfile.TemporaryDirectory(dir=reference.ROOT / 'reports') as parent:
            output = Path(parent) / 'reference'
            manifest = reference.generate(self.upstream, output)
            self.assertEqual(manifest['profile'], 'xsum-rope-reference-v1')
            self.assertEqual(manifest['tensor_sha256'], reference.EXPECTED)
            self.assertEqual(json.loads((output / 'manifest.json').read_text()), manifest)
            self.assertEqual(reference.digest((output / 'rope.safetensors').read_bytes()), manifest['payload_sha256'])
            with self.assertRaises(FileExistsError):
                reference.generate(self.upstream, output)
            self.assertEqual(json.loads((output / 'manifest.json').read_text()), manifest)

    def test_rejects_changed_upstream_source(self):
        with tempfile.TemporaryDirectory(dir=reference.ROOT / 'reports') as parent:
            root = Path(parent)
            source = root / 'upstream/models/common.py'
            source.parent.mkdir(parents=True)
            source.write_text('# altered source\n')
            output = root / 'output'
            with self.assertRaisesRegex(ValueError, 'upstream rotary implementation differs'):
                reference.generate(source.parents[1], output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
