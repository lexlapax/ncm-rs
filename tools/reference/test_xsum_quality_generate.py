"""CPU-only integrity checks for resumable XSum quality batches."""
import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
import xsum_quality_generate as quality


class QualityBatches(unittest.TestCase):
    def test_rng_state_roundtrips(self):
        state = torch.tensor([0, 1, 127, 128, 255], dtype=torch.uint8)
        self.assertTrue(torch.equal(state, quality.restore_state(quality.encoded_state(state))))

    def test_shards_preserve_official_order_and_complete_batch_boundaries(self):
        a, short = quality.shard_batches(0, 256)
        b, short_b = quality.shard_batches(1, 256)
        self.assertEqual(len(short), 4)
        self.assertEqual(len(short_b), 4)
        self.assertEqual(a[0], list(range(0, 64, 2)))
        self.assertEqual(b[0], list(range(1, 64, 2)))
        self.assertEqual(short[-1][1][-1], 254)
        self.assertEqual(short_b[-1][1][-1], 255)
        self.assertEqual(len(quality.shard_batches(0, 11334)[1]), len(a))
        with self.assertRaises(ValueError): quality.shard_batches(2, 256)
        with self.assertRaises(ValueError): quality.shard_batches(0, 0)

    def test_batch_envelope_detects_tampering_and_symlinks(self):
        state = base64.b64encode(b'cpu-state').decode()
        record = {'ordinal': 0, 'indices': [0], 'codes': [[0]*200], 'raw_ids': [[1]*48],
                  'cpu_rng': state, 'cuda_rng': state}
        envelope = {'record': record, 'sha256': hashlib.sha256(quality.canonical(record)).hexdigest()}
        with tempfile.TemporaryDirectory(dir='reports') as directory:
            root = Path(directory)
            path = root / 'batch.json'
            path.write_text(json.dumps(envelope))
            self.assertEqual(quality.read_record(path, [0], 0), record)
            with self.assertRaises(ValueError): quality.read_record(path, [2], 0)
            link = root / 'link.json'; link.symlink_to(path.name)
            with self.assertRaises(ValueError): quality.read_record(link, [0], 0)
            envelope['record']['raw_ids'][0][0] = 2
            path.write_text(json.dumps(envelope))
            with self.assertRaises(ValueError): quality.read_record(path, [0], 0)

    def test_rejects_bad_token_bounds_and_rng(self):
        state = base64.b64encode(b'x').decode()
        value = {'ordinal': 0, 'indices': [0], 'codes': [[0]*200], 'raw_ids': [[0]*48],
                 'cpu_rng': state, 'cuda_rng': state}
        quality.validate_record(value, [0], 0)
        value['codes'][0][0] = 4096
        with self.assertRaises(ValueError): quality.validate_record(value, [0], 0)
        value['codes'][0][0] = 0
        value['cuda_rng'] = ''
        with self.assertRaises(ValueError): quality.validate_record(value, [0], 0)


if __name__ == '__main__':
    unittest.main()
