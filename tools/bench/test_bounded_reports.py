"""Report receipts with fake CPU producers, never model inference."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from bounded_child import fingerprint_report, run_bounded


@unittest.skipUnless(sys.platform == 'linux', 'Linux collector')
class ReportReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.report = self.root / 'report.json'
        self.capture = self.root / 'capture'
        self.receipt = self.root / 'receipt.json'

    def run_child(self, code, **kwargs):
        options = dict(cwd=self.root, env={'PATH': os.defpath, 'PYTHONNOUSERSITE': '1'},
                       capture_path=self.capture, receipt_path=self.receipt,
                       timeout_seconds=3, output_limit_bytes=4096, report_path=self.report)
        options.update(kwargs)
        return run_bounded([sys.executable, '-c', code, str(self.report)], **options)

    def test_both_outputs_bound_to_receipt_without_quality_claim(self):
        result = self.run_child('import pathlib,sys; print("trace"); pathlib.Path(sys.argv[1]).write_text("{}")')
        self.assertEqual(result['status'], 'complete')
        bound = result['report_capture']
        self.assertEqual(bound['sha256'], hashlib.sha256(b'{}').hexdigest())
        self.assertEqual(bound['size_bytes'], 2)
        self.assertEqual(bound['inode'], self.report.stat().st_ino)
        self.assertTrue(bound['post_exit_hash_verified'])
        self.assertFalse(bound['producer_identity_verified'])
        self.assertFalse(result['workload_output_qualified'])
        self.assertFalse(result['executable_identity_verified'])
        self.assertEqual(result['capture_sha256'], hashlib.sha256(b'trace\n').hexdigest())
        self.assertEqual(json.loads(self.receipt.read_text()), result)

    def test_missing_report_fails_even_when_child_exits_zero(self):
        result = self.run_child('pass')
        self.assertEqual(result['returncode'], 0)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('report_error:FileNotFoundError', result['issues'])
        self.assertFalse(result['report_capture']['post_exit_hash_verified'])

    def test_report_limit_and_empty_report(self):
        for content in ('', '12345'):
            with self.subTest(content=content):
                result = self.run_child(f'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text({content!r})',
                                        report_limit_bytes=4)
                self.assertEqual(result['status'], 'failed')
                self.assertIn('report_error:ValueError', result['issues'])
                for path in (self.report, self.capture, self.receipt):
                    path.unlink()

    def test_preexisting_report_or_broken_symlink_prevents_spawn(self):
        for kind in ('file', 'broken_symlink'):
            if kind == 'file':
                self.report.write_text('preserve')
            else:
                self.report.symlink_to(self.root / 'missing')
            with self.subTest(kind=kind), patch('bounded_child.subprocess.Popen') as child:
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    self.run_child('pass')
                child.assert_not_called()
                self.assertFalse(self.capture.exists())
                self.assertFalse(self.receipt.exists())
            self.report.unlink()

    def test_path_argument_alias_parent_and_limit_guards(self):
        for options in ({'report_path': Path('relative.json')},
                        {'report_path': self.capture},
                        {'report_path': self.root.parent / 'elsewhere.json'},
                        {'report_limit_bytes': True}):
            with self.subTest(options=options), patch('bounded_child.subprocess.Popen') as child:
                with self.assertRaises(ValueError):
                    self.run_child('pass', **options)
                child.assert_not_called()
                self.assertFalse(self.capture.exists())
        # Alias checks also apply when the aliased path really is an argv argument.
        self.report = self.capture
        with self.assertRaisesRegex(ValueError, 'separate files'):
            self.run_child('pass')

    def test_child_created_symlink_fifo_directory_rejected_without_blocking(self):
        target = self.root / 'target'
        target.write_text('do not read through symlink')
        for statement in (f'os.symlink({str(target)!r}, sys.argv[1])',
                          'os.mkfifo(sys.argv[1])', 'os.mkdir(sys.argv[1])'):
            with self.subTest(statement=statement):
                result = self.run_child('import os,sys; ' + statement)
                self.assertEqual(result['status'], 'failed')
                self.assertTrue(any(i.startswith('report_error:') for i in result['issues']))
                if self.report.is_dir():
                    self.report.rmdir()
                else:
                    self.report.unlink()
                self.capture.unlink()
                self.receipt.unlink()
        self.assertEqual(target.read_text(), 'do not read through symlink')

    def test_failed_child_report_is_not_promoted(self):
        result = self.run_child('import pathlib,sys; pathlib.Path(sys.argv[1]).write_text("{}"); sys.exit(2)')
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['report_capture']['post_exit_hash_verified'])
        self.assertEqual(self.report.read_text(), '{}')

    def test_visible_replacement_while_hashing_is_rejected(self):
        self.report.write_text('{}')
        actual = os.stat

        def replace(path, **kwargs):
            self.report.unlink()
            self.report.write_text('different')
            return actual(path, **kwargs)

        with patch('bounded_child.os.stat', side_effect=replace):
            with self.assertRaisesRegex(ValueError, 'changed'):
                fingerprint_report(self.report, 16)


if __name__ == '__main__':
    unittest.main()
