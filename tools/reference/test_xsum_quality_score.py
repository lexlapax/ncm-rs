"""CPU-only checks for the XSum scorer's input validation."""
import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import xsum_quality_generate as generate
import xsum_quality_score as score


class QualityScoreChecks(unittest.TestCase):
    def test_decode_stops_at_eos_and_removes_padding(self):
        class Tokenizer:
            def decode(self, ids, skip_special_tokens):
                return ids, skip_special_tokens
        self.assertEqual(score.decode_ids(Tokenizer(), [1, 50257, 2, 50256, 3]), ([1, 2], True))

    def test_shards_validate_coverage_integrity_and_environment(self):
        with tempfile.TemporaryDirectory(dir='reports') as temp:
            directories = [Path(temp) / '0', Path(temp) / '1']
            state = base64.b64encode(b'x').decode()
            for shard, directory in enumerate(directories):
                directory.mkdir()
                manifest = {'schema_version': 1, 'origin': 'pinned-upstream-cuda',
                            'shard': shard, 'shards': 2, 'upstream': score.UPSTREAM,
                            'data_sha256': score.DATA_SHA, 'evaluator_sha256': score.EVALUATOR_SHA,
                            'sources': {k: v[1] for k, v in score.INPUTS.items()},
                            'batch_size': 32, 'temperature': 1.0, 'top_k': 1,
                            'top_p': 0.0, 'cfg': 0.0, 'tf32': False, 'sdpa': 'math',
                            'between_batch_gc': True, 'seed': 42, 'gpu_identity_and_driver': 'test',
                            'generator_sha256': score.digest_file(score.ROOT / 'tools/reference/xsum_quality_generate.py')}
                (directory / 'manifest.json').write_text(json.dumps(manifest))
                indices = list(range(shard, 64, 2))
                record = {'ordinal': 0, 'indices': indices, 'codes': [[0] * 200 for _ in indices],
                          'raw_ids': [[1] * 48 for _ in indices], 'cpu_rng': state, 'cuda_rng': state}
                envelope = {'record': record, 'sha256': hashlib.sha256(generate.canonical(record)).hexdigest()}
                (directory / 'batch-00000.json').write_text(json.dumps(envelope))
            rows, _ = score.checked_rows(directories, 64)
            self.assertEqual([row['index'] for row in rows], list(range(64)))
            batch = directories[0] / 'batch-00000.json'
            original = batch.read_text()
            altered = json.loads(original); altered['record']['raw_ids'][0][0] = 2
            batch.write_text(json.dumps(altered))
            with self.assertRaises(ValueError): score.checked_rows(directories, 64)
            batch.write_text(original)
            meta = directories[1] / 'manifest.json'
            manifest = json.loads(meta.read_text()); manifest['seed'] = 43
            meta.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): score.checked_rows(directories, 64)
            batch.unlink()
            with self.assertRaises(ValueError): score.checked_rows(directories, 64)


if __name__ == '__main__':
    unittest.main()
