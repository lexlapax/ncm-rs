"""Metal scorer refuses missing, altered or misidentified capture receipts."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import xsum_mlx_stochastic_score as subject


class MetalScorerCaptureTests(unittest.TestCase):
    def test_validate_complete_capture_and_detect_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {'head': 'e14ae02a517b0b3d1000d942f94d72bc408d851a',
                        'data_sha256': subject.DATA_SHA, 'artifact_sha256': subject.ARTIFACT_SHA,
                        'binary_sha256': subject.BINARY_SHA, 'protocol_sha256': subject.PROTOCOL_SHA,
                        'script_sha256': subject.CAPTURE_SHA, 'seed': 4, 'batches': 64,
                        'batch_size': 32, 'start': 512,
                        'sampling': {'temperature': 1.0, 'top_k': 0, 'top_p': 1.0, 'cfg': 0.0}}
            (root / 'manifest.json').write_bytes(subject.canonical(manifest))
            (root / 'complete.json').write_bytes(subject.canonical({'seed': 4, 'batches': 64,
                                                                       'binary_sha256': subject.BINARY_SHA}))
            data = {'inputs': [[0] * 560] * 2560}
            for batch in range(64):
                prefixes = [[0] * 512] * 32
                (root / f'input-{batch:02}.json').write_bytes(subject.canonical(prefixes) + b'\n')
                record = {'batch': batch, 'start': 512 + 32 * batch, 'batch_seed': 4000 + batch,
                          'result': {'codes': [[0] * 200] * 32, 'ids': [[0] * 48] * 32}}
                value = {'record': record, 'sha256': hashlib.sha256(subject.canonical(record)).hexdigest()}
                (root / f'batch-{batch:02}.json').write_bytes(subject.canonical(value))
            found, rows, files = subject.read_metal(root, data)
            self.assertEqual(found['seed'], 4)
            self.assertEqual(len(rows), 2048)
            self.assertEqual(len(files), 130)
            path = root / 'batch-00.json'
            changed = json.loads(path.read_bytes())
            changed['record']['batch_seed'] += 1
            changed['sha256'] = hashlib.sha256(subject.canonical(changed['record'])).hexdigest()
            path.write_bytes(subject.canonical(changed))
            with self.assertRaisesRegex(ValueError, 'batch identity'):
                subject.read_metal(root, data)
            (root / 'input-00.json').write_text('[]\n')
            with self.assertRaisesRegex(ValueError, 'prefixes differ'):
                subject.read_metal(root, data)


if __name__ == '__main__':
    unittest.main()
