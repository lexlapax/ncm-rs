"""Offline controls for the LM1B cohort CLI comparison."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import phase5_lm1b_cli_cohort as tool


class CohortTests(unittest.TestCase):
    def test_request_seed_and_bounds(self):
        self.assertEqual(tool.request_seed(2000, 0), 2000 << 32)
        self.assertEqual(tool.request_seed(2079, 31), (2079 << 32) | 31)
        for seed, batch in [(1999, 0), (2080, 0), (2000, 32), (True, 0)]:
            with self.assertRaises(ValueError):
                tool.request_seed(seed, batch)

    def test_comparison_checks_bytes_and_cohort_rows(self):
        capture = {'codes': [[i] for i in range(256)], 'raw_ids': [[i + 1] for i in range(256)]}
        value = {'codes': capture['codes'][8:16], 'ids': capture['raw_ids'][8:16]}
        raw = json.dumps(value).encode()
        tool.compare(raw, raw, capture, 1)
        with self.assertRaisesRegex(ValueError, 'bytes differ'):
            tool.compare(raw, raw + b' ', capture, 1)
        with self.assertRaisesRegex(ValueError, 'differs from cohort'):
            tool.compare(raw, raw, capture, 0)

    def test_invalid_capture_metadata_and_rows(self):
        capture = {'size': 'small', 'seed': 2000, 'count': 256, 'batch_size': 8,
                   'artifact_manifest_sha256': 'manifest', 'codes': [[0] * 641 for _ in range(256)],
                   'raw_ids': [[0] * 128 for _ in range(256)]}
        tool.validate_capture(capture, 'small', 2000, 'manifest')
        with self.assertRaisesRegex(ValueError, 'metadata'):
            tool.validate_capture(capture, 'large', 2000, 'manifest')
        capture['raw_ids'][1][0] = True
        with self.assertRaisesRegex(ValueError, 'rows'):
            tool.validate_capture(capture, 'small', 2000, 'manifest')

    def test_existing_output_rejected_before_input_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'dangling'
            output.symlink_to(root / 'missing')
            argv = ['cohort', '--size', 'small', '--baseline', 'old', '--candidate', 'new',
                    '--artifacts', 'artifacts', '--rope', 'rope', '--output', str(output)]
            with patch('sys.argv', argv), patch.object(tool, 'checked') as checked:
                with self.assertRaisesRegex(ValueError, 'output exists'):
                    tool.main()
                checked.assert_not_called()


if __name__ == '__main__':
    unittest.main()
