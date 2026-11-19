"""Negative controls for provenance-only artifact comparison."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import compare_artifact_provenance as tool


def write_artifact(root, provenance, payload=b'weights', tensors=None):
    root.mkdir()
    raw = json.dumps(provenance, sort_keys=True).encode()
    (root / 'provenance.json').write_bytes(raw)
    (root / 'codec.safetensors').write_bytes(payload)
    files = {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
             for name, data in [('provenance.json', raw), ('codec.safetensors', payload)]}
    manifest = {'profile': 'test', 'files': files, 'tensors': tensors or {'codec.safetensors': []}}
    data = json.dumps(manifest, sort_keys=True).encode()
    (root / 'manifest.json').write_bytes(data)
    return hashlib.sha256(data).hexdigest()


class CompareTests(unittest.TestCase):
    def test_accepts_only_requested_provenance_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = root / 'old', root / 'new'
            a = write_artifact(old, {'converter_sha256': 'before', 'python': '3.14.9'})
            b = write_artifact(new, {'converter_sha256': 'after', 'python': '3.14.9'})
            result = tool.compare(old, new, a, b, ['converter_sha256'])
            self.assertEqual(result['file_count'], 2)
            self.assertEqual(set(result['provenance_changes']), {'converter_sha256'})
            with self.assertRaisesRegex(ValueError, 'allowed keys'):
                tool.compare(old, new, a, b, [])
            with self.assertRaisesRegex(ValueError, 'manifest hash'):
                tool.compare(old, new, '0' * 64, b, ['converter_sha256'])
            (new / 'codec.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'payload verification'):
                tool.compare(old, new, a, b, ['converter_sha256'])

    def test_rejects_changed_payload_even_with_rehashed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = root / 'old', root / 'new'
            a = write_artifact(old, {'converter': 1})
            b = write_artifact(new, {'converter': 2}, payload=b'changed')
            with self.assertRaisesRegex(ValueError, 'non-provenance payload'):
                tool.compare(old, new, a, b, ['converter'])

    def test_rejects_changed_tensor_inventory_and_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = root / 'old', root / 'new'
            a = write_artifact(old, {'converter': 1})
            b = write_artifact(new, {'converter': 2}, tensors={'codec.safetensors': [1]})
            with self.assertRaisesRegex(ValueError, 'tensor inventory'):
                tool.compare(old, new, a, b, ['converter'])
            (new / 'codec.safetensors').unlink()
            (new / 'codec.safetensors').symlink_to(old / 'codec.safetensors')
            with self.assertRaisesRegex(ValueError, 'regular file'):
                tool.compare(old, new, a, b, ['converter'])

    def test_direct_byte_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / 'left', Path(directory) / 'right'
            left.write_bytes(b'a' * (1024 * 1024 + 1))
            right.write_bytes(left.read_bytes())
            self.assertTrue(tool.equal_bytes(left, right))
            right.write_bytes(b'a' * (1024 * 1024) + b'b')
            self.assertFalse(tool.equal_bytes(left, right))

    def test_json_comparison_distinguishes_types_and_rejects_nonfinite(self):
        self.assertFalse(tool.same_json({'x': 1}, {'x': True}))
        self.assertFalse(tool.same_json({'x': 1}, {'x': 1.0}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.json'
            path.write_text('{"x":NaN}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                tool.read_json(path, 1024)

    def test_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text('{"files":{},"files":{}}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
                tool.read_json(path, 1024)


if __name__ == '__main__':
    unittest.main()
