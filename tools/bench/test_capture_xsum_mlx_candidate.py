"""Check resumable Metal capture receipts and qualification binding."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from capture_xsum_mlx_candidate import (ARTIFACT, ARTIFACT_SHA, BINARY, INPUT_SHA, REFERENCE_SHA,
                                        ROOT, canonical, checked_qualification, publish, read_batch,
                                        validate_batch_sequence)


class CaptureTests(unittest.TestCase):
    def test_resume_rejects_gaps_and_extra_batches(self):
        starts = [0, 32]
        validate_batch_sequence([Path('batch-00000.json')], starts)
        for names in (['batch-00032.json'], ['batch-00000.json', 'batch-00032.json', 'batch-00064.json']):
            with self.subTest(names=names), self.assertRaises(ValueError):
                validate_batch_sequence([Path(name) for name in names], starts)

    def test_publish_is_non_overwriting_and_batch_receipt_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'batch.json'
            record = {'start': 32, 'count': 1, 'result': {'codes': [[1] * 200], 'ids': [[2] * 48]}}
            envelope = {'record': record, 'sha256': hashlib.sha256(canonical(record)).hexdigest()}
            publish(path, canonical(envelope) + b'\n')
            self.assertEqual(read_batch(path, 32, 1), record['result'])
            with self.assertRaises(FileExistsError):
                publish(path, b'changed')
            with self.assertRaises(ValueError):
                read_batch(path, 64, 1)
            envelope['record']['result']['ids'][0][0] = 9
            path.write_text(json.dumps(envelope))
            with self.assertRaises(ValueError):
                read_batch(path, 32, 1)

    def test_qualification_binds_binary_and_inputs(self):
        sources = {str(BINARY): 'binary', str(ARTIFACT / 'manifest.json'): ARTIFACT_SHA,
                   str(ROOT / 'reports/phase4-xsum-mlx-cli-inputs-v1/prefixes.json'): INPUT_SHA,
                   str(ROOT / 'reports/phase4-xsum-mlx-cli-inputs-v1/reference.json'): REFERENCE_SHA}
        value = {'shapes_ok': True, 'code_differences': 0, 'id_differences': 0, 'source_sha256': sources}
        checked_qualification(value, 'binary')
        for key in sources:
            with self.subTest(key=key):
                changed = dict(value, source_sha256={**sources, key: 'different'})
                with self.assertRaises(ValueError):
                    checked_qualification(changed, 'binary')
        with self.assertRaises(ValueError):
            checked_qualification(dict(value, id_differences=1), 'binary')


if __name__ == '__main__':
    unittest.main()
