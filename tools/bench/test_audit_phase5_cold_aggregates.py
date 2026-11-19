"""Negative controls for the Phase 5 compact cold-screen audit."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import audit_phase5_cold_aggregates as audit


class ColdAggregateAuditTests(unittest.TestCase):
    def make_records(self, root):
        rows = []
        output = b'fixed output\n'
        digest = hashlib.sha256(output).hexdigest()
        for group in range(1, 4):
            for ordinal in range(35):
                for condition in ('baseline', 'candidate'):
                    (root / f'{group}-{ordinal:02d}-{condition}.stdout').write_bytes(output)
                    rows.append({'group': group, 'ordinal': ordinal, 'condition': condition,
                                 'warmup': ordinal < 5, 'elapsed_seconds': float(ordinal + 1),
                                 'max_rss_kib': 100 + ordinal, 'output_sha256': digest})
        return rows

    def test_calculates_measured_samples_and_detects_tampered_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.make_records(root)
            result = audit.calculate(rows, root)
            self.assertEqual(result['baseline']['median_seconds'], 20.5)
            self.assertEqual(result['baseline']['p95_seconds_nearest_rank'], 34.0)
            self.assertEqual(result['baseline']['max_rss_kib'], 134)
            (root / '1-05-baseline.stdout').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'stdout hash differs'):
                audit.calculate(rows, root)

    def test_rejects_duplicate_warmup_and_invalid_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.make_records(root)
            mutations = []
            duplicate = copy.deepcopy(rows)
            duplicate[1] = copy.deepcopy(duplicate[0])
            mutations.append(duplicate)
            for key, value in (('warmup', True), ('elapsed_seconds', float('nan')),
                               ('max_rss_kib', True), ('group', True)):
                changed = copy.deepcopy(rows)
                changed[10][key] = value
                mutations.append(changed)
            for changed in mutations:
                with self.subTest(row=changed[10]), self.assertRaises(ValueError):
                    audit.calculate(changed, root)

    def test_rejects_altered_aggregate_and_raw_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.make_records(root)
            values = audit.calculate(rows, root)
            gpu = b'gpu samples\n'
            (root / 'gpu.csv').write_bytes(gpu)
            data = {'gpu_csv_sha256': audit.sha(gpu), 'records': rows, 'summary': values,
                    'watched_sha256': {'/baseline-ncm': 'old', '/candidate/ncm': 'new'}}
            raw = json.dumps(data).encode()
            (root / 'result.json').write_bytes(raw)
            entry = {'raw_result_sha256': audit.sha(raw), 'gpu_csv_sha256': audit.sha(gpu), **values}
            audit.audit_one('test', entry, root, 'old', 'new')
            changed = copy.deepcopy(entry)
            changed['candidate']['median_seconds'] += 1
            with self.assertRaisesRegex(ValueError, 'aggregate differs'):
                audit.audit_one('test', changed, root, 'old', 'new')
            (root / 'result.json').write_bytes(raw + b' ')
            with self.assertRaisesRegex(ValueError, 'raw result or GPU CSV hash differs'):
                audit.audit_one('test', entry, root, 'old', 'new')

    def test_rejects_symlinked_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'target'
            target.write_bytes(b'file')
            link = root / 'link'
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, 'not a regular file'):
                audit.read_regular(link)


if __name__ == '__main__':
    unittest.main()
