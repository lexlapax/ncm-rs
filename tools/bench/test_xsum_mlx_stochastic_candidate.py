"""Test stochastic replay receipts, resume ordering and exact comparisons."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from capture_xsum_mlx_quality import canonical
from capture_xsum_mlx_stochastic_candidate import read_batch, validate_sequence
from compare_xsum_mlx_stochastic_candidate import compare_seed


def write_batch(directory, seed, batch, changed=False):
    result = {'codes': [[1] * 200 for _ in range(32)], 'ids': [[2] * 48 for _ in range(32)]}
    if changed:
        result['codes'][3][7] = 4
        result['ids'][5][8] = 6
    record = {'batch': batch, 'start': 512 + batch * 32, 'batch_seed': seed * 1000 + batch, 'result': result}
    envelope = {'record': record, 'sha256': hashlib.sha256(canonical(record)).hexdigest()}
    (directory / f'batch-{batch:02}.json').write_bytes(canonical(envelope) + b'\n')


class StochasticCaptureTests(unittest.TestCase):
    def test_receipts_bind_seed_batch_and_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_batch(path, 4, 0)
            self.assertEqual(len(read_batch(path / 'batch-00.json', 4, 0)['codes']), 32)
            with self.assertRaises(ValueError):
                read_batch(path / 'batch-00.json', 5, 0)
            envelope = json.loads((path / 'batch-00.json').read_bytes())
            envelope['record']['result']['ids'][0][0] = 9
            (path / 'batch-00.json').write_text(json.dumps(envelope))
            with self.assertRaises(ValueError):
                read_batch(path / 'batch-00.json', 4, 0)

    def test_resume_sequence_rejects_gaps_and_extra_files(self):
        validate_sequence([Path('batch-00.json'), Path('batch-01.json')], 2)
        for names, count in ((['batch-01.json'], 2), (['batch-00.json', 'batch-01.json'], 1)):
            with self.subTest(names=names), self.assertRaises(ValueError):
                validate_sequence([Path(name) for name in names], count)

    def test_full_seed_comparison_attributes_exact_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline, candidate = Path(directory) / 'a', Path(directory) / 'b'
            baseline.mkdir()
            candidate.mkdir()
            for batch in range(64):
                write_batch(baseline, 4, batch)
                write_batch(candidate, 4, batch, changed=batch == 2)
            result = compare_seed(baseline, candidate, 4)
            self.assertEqual(result, {'rows': 2048, 'code_differences': 1, 'id_differences': 1,
                                      'code_rows': [579], 'id_rows': [581]})


if __name__ == '__main__':
    unittest.main()
