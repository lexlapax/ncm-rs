"""Offline failure and budget tests for the checkpoint downloader."""
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fetch_checkpoints as tool


class Response(io.BytesIO):
    def __init__(self, data, url='https://example.test/object'):
        super().__init__(data)
        self.url = url

    def geturl(self):
        return self.url


def record(name='config.yaml', data=b'abc'):
    return {'remote': 'test/' + name, 'output': name, 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


class FetchTests(unittest.TestCase):
    def test_success_verifies_and_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'out'
            with patch.object(tool.urllib.request, 'urlopen', return_value=Response(b'abc')) as request:
                tool.fetch(output, [record()], 3)
            self.assertEqual(request.call_count, 1)
            tool.verify(output, [record()])
            self.assertEqual((output / 'config.yaml').read_bytes(), b'abc')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                tool.fetch(output, [record()], 3)

    def test_budget_and_disk_rejected_before_network_or_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'out'
            with patch.object(tool.urllib.request, 'urlopen') as request:
                for budget in (None, 2, -1, True):
                    with self.assertRaisesRegex(ValueError, 'budget'):
                        tool.fetch(output, [record()], budget)
                with patch.object(tool.shutil, 'disk_usage') as disk:
                    disk.return_value = type('Disk', (), {'free': 2})()
                    with self.assertRaisesRegex(ValueError, 'insufficient'):
                        tool.fetch(output, [record()], 3)
                request.assert_not_called()
            self.assertFalse(output.exists())

    def test_short_long_wrong_hash_and_insecure_redirect_fail(self):
        for data, url in [(b'ab', 'https://example.test/a'), (b'abcd', 'https://example.test/a'),
                          (b'xyz', 'https://example.test/a'), (b'abc', 'http://example.test/a')]:
            with self.subTest(data=data, url=url), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'out'
                with self.assertRaises(ValueError):
                    tool.write_verified(Response(data, url), path, 3, record()['sha256'])
                if path.exists():
                    self.assertLessEqual(path.stat().st_size, 3)

    def test_verify_rejects_modified_and_symlinked_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.yaml').write_bytes(b'abc')
            tool.verify(root, [record()])
            (root / 'config.yaml').write_bytes(b'xyz')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                tool.verify(root, [record()])
            (root / 'config.yaml').unlink()
            (root / 'other').write_bytes(b'abc')
            (root / 'config.yaml').symlink_to(root / 'other')
            with self.assertRaisesRegex(ValueError, 'symlink'):
                tool.verify(root, [record()])

    def test_failed_fetch_retains_partial_directory_for_inspection(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'out'
            records = [record('first', b'abc'), record('second', b'def')]
            with patch.object(tool.urllib.request, 'urlopen', side_effect=[Response(b'abc'), Response(b'wrong')]):
                with self.assertRaises(ValueError):
                    tool.fetch(output, records, 6)
            self.assertEqual((output / 'first').read_bytes(), b'abc')
            self.assertTrue((output / 'second').exists())
            with self.assertRaises(ValueError):
                tool.verify(output, records)

    def test_plan_does_not_use_network_or_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'out'
            args = ['fetch_checkpoints', '--profile', 'sudoku-easy', '--output', str(output), '--plan']
            with patch('sys.argv', args), patch.object(tool, 'specs', return_value=[record()]), \
                    patch.object(tool.urllib.request, 'urlopen') as request, patch('sys.stdout', new_callable=io.StringIO):
                tool.main()
                request.assert_not_called()
            self.assertFalse(output.exists())

    def test_fetch_fails_if_parent_is_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'real').mkdir()
            (root / 'link').symlink_to(root / 'real', target_is_directory=True)
            with patch.object(tool.urllib.request, 'urlopen') as request:
                with self.assertRaisesRegex(ValueError, 'parent'):
                    tool.fetch(root / 'link' / 'out', [record()], 3)
                request.assert_not_called()


if __name__ == '__main__':
    unittest.main()
