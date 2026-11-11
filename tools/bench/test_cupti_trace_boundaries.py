import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from cupti_trace_boundaries import main, validate_boundaries


def driver(index, api, *, process=1, thread=2):
    start = index * 10
    return (f'DRIVER [ {start}, {start + 2} ] duration 2, "{api}", '
            f'cbid 1, processId {process}, threadId {thread}, correlationId {index}')


def fixture(layout='historical-block4'):
    apis = []

    def timed():
        apis.extend(['cuEventRecord', 'cuEventRecord', 'cuEventElapsedTime_v2'])

    if layout == 'historical-block4':
        for _ in range(36):
            timed()
    else:
        timed()  # Cold reconstruction.
        for _ in range(3):
            for _ in range(4 * 36):
                timed()
            for _ in range(35):
                apis.extend(['cuEventRecord'] * 30)
                apis.extend(['cuEventElapsedTime_v2'] * 15)
    return [driver(i, api) for i, api in enumerate(apis, 1)]


def raw(lines):
    return ('\n'.join(lines) + '\n').encode()


def archived_fixture_results():
    root = Path(__file__).resolve().parents[2] / 'docs/results/2026-11-10-phase2-cuda-remediation'
    rows = []
    with tarfile.open(root / 'raw/cupti-profiles.tar.gz') as archive:
        for suffix in ('', '-b', '-retained'):
            name = f'codec-diagnostic-block4{suffix}-cupti'
            data = archive.extractfile(f'{name}.txt').read()
            expected = json.loads((root / f'profiles/{name}.json').read_text())
            result = validate_boundaries(data, 'historical-block4')
            try:
                validate_boundaries(data, 'codec-bench-b1')
            except ValueError:
                b1_rejected = True
            else:
                b1_rejected = False
            rows.append({'name': name, 'sha256': result['trace_sha256'],
                         'expected_sha256': expected['trace_sha256'],
                         'windows': result['validated_prefix_windows'],
                         'queries': result['total_elapsed_queries'], 'b1_rejected': b1_rejected})
    return rows


class BoundaryTests(unittest.TestCase):
    def test_historical_labels_and_fingerprint(self):
        data = raw(fixture())
        result = validate_boundaries(data, 'historical-block4')
        self.assertEqual(result['trace_sha256'], hashlib.sha256(data).hexdigest())
        self.assertEqual(result['validated_prefix_windows'], 36)
        work = [w['assumed_work'] for w in result['windows']]
        self.assertEqual(work[0], dict(batch=4, scope='resident_decoder_block', phase='first', index=0))
        self.assertEqual([w['phase'] for w in work[1:6]], ['warmup'] * 5)
        self.assertEqual([w['index'] for w in work[6:]], list(range(30)))
        self.assertIn('not kernel attribution', result['scope'])

    def test_full_source_schedule_includes_stage_queries(self):
        result = validate_boundaries(raw(fixture('codec-bench-b1')), 'codec-bench-b1')
        self.assertEqual(result['total_elapsed_queries'], 2008)
        self.assertEqual(result['validated_prefix_windows'], 145)
        windows = result['windows']
        self.assertEqual(windows[0]['assumed_work']['phase'], 'cold')
        block = [w for w in windows if w['assumed_work']['scope'] == 'resident_decoder_block']
        self.assertEqual([w['ordinal'] for w in block], list(range(110, 146)))
        self.assertEqual(block[0]['assumed_work']['phase'], 'first')
        self.assertEqual([w['ordinal'] for w in block if w['assumed_work']['phase'] == 'trial'],
                         list(range(116, 146)))

    def test_shuffled_driver_print_order(self):
        lines = fixture()
        expected = validate_boundaries(raw(lines), 'historical-block4')['windows']
        random.Random(42).shuffle(lines)
        # Other activity is intentionally not interpreted by this boundary helper.
        lines.insert(12, 'CONCURRENT_KERNEL malformed payload is not validated here')
        result = validate_boundaries(raw(lines), 'historical-block4')
        self.assertEqual(result['windows'], expected)

    def test_layout_rejection_and_no_events(self):
        for data, layout in ((b'', 'historical-block4'), (raw(fixture()), 'unknown'),
                             (raw(fixture()), 'codec-bench-b1')):
            with self.subTest(layout=layout), self.assertRaises(ValueError):
                validate_boundaries(data, layout)

    def test_dropped_and_malformed_records(self):
        for suffix in (b'Dropped 0 records', b'dropped 2 records', b'DRIVER [ 12,'):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                validate_boundaries(raw(fixture()) + suffix, 'historical-block4')

    def test_invalid_duration_and_reverse_interval(self):
        for replacement in ('duration 3', 'duration -2'):
            data = raw(fixture()).replace(b'duration 2', replacement.encode(), 1)
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                validate_boundaries(data, 'historical-block4')
        with self.assertRaises(ValueError):
            validate_boundaries(raw(fixture()).replace(b'[ 10, 12 ]', b'[ 12, 10 ]', 1),
                                'historical-block4')

    def test_duplicate_correlation(self):
        lines = fixture() + [driver(1, 'cuMemAllocAsync')]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_boundaries(raw(lines), 'historical-block4')

    def test_multiple_threads_or_processes(self):
        for kwargs in ({'thread': 3}, {'process': 2}):
            lines = fixture() + [driver(500, 'cuMemAllocAsync', **kwargs)]
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, 'process/thread'):
                validate_boundaries(raw(lines), 'historical-block4')

    def test_missing_extra_and_truncated_queries(self):
        lines = fixture()
        for changed in (lines[:-1], lines[1:], lines + [driver(500, 'cuEventRecord')]):
            with self.subTest(length=len(changed)), self.assertRaises(ValueError):
                validate_boundaries(raw(changed), 'historical-block4')

    def test_ambiguous_boundary_intervals(self):
        for replacement in ('[ 10, 12 ]', '[ 11, 13 ]'):
            data = raw(fixture()).replace(b'[ 20, 22 ]', replacement.encode(), 1)
            with self.subTest(replacement=replacement), self.assertRaisesRegex(ValueError, 'ambiguous'):
                validate_boundaries(data, 'historical-block4')

    def test_correct_totals_but_wrong_pairing(self):
        lines = fixture()
        # Exchange a query and the next start event: totals still match.
        lines[2] = driver(3, 'cuEventRecord')
        lines[3] = driver(4, 'cuEventElapsedTime_v2')
        with self.assertRaisesRegex(ValueError, 'exactly two'):
            validate_boundaries(raw(lines), 'historical-block4')

    def test_trailing_event_and_unsupported_api(self):
        lines = fixture()
        lines[-2] = driver(107, 'cuEventElapsedTime_v2')
        lines[-1] = driver(108, 'cuEventRecord')
        with self.assertRaisesRegex(ValueError, 'after final'):
            validate_boundaries(raw(lines), 'historical-block4')
        for old, new in ((b'cuEventRecord', b'cuEventRecordWithFlags'),
                         (b'cuEventElapsedTime_v2', b'cuEventElapsedTime_unknown')):
            with self.subTest(new=new), self.assertRaisesRegex(ValueError, 'unsupported'):
                validate_boundaries(raw(fixture()).replace(old, new, 1), 'historical-block4')

    def test_archived_b4_fixtures_are_not_current_b1_evidence(self):
        # Large archived traces inflate the parent allocator/RSS. Keep that work
        # out of the suite process: later wait4 tests inherit its pre-exec RSS floor.
        child = subprocess.run(
            [sys.executable, '-c', 'import json; from test_cupti_trace_boundaries import '
             'archived_fixture_results; print(json.dumps(archived_fixture_results()))'],
            cwd=Path(__file__).resolve().parent, check=True, capture_output=True, text=True,
            timeout=30,
        )
        rows = json.loads(child.stdout)
        self.assertEqual(len(rows), 3)
        for row in rows:
            with self.subTest(name=row['name']):
                self.assertEqual(row['sha256'], row['expected_sha256'])
                self.assertEqual(row['windows'], 36)
                self.assertEqual(row['queries'], 36)
                self.assertTrue(row['b1_rejected'])

    def test_cli_exclusive_output_and_invalid_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace, output = Path(tmp) / 'trace.txt', Path(tmp) / 'output.json'
            trace.write_bytes(raw(fixture()))
            args = ['boundaries', str(trace), str(output), '--layout', 'historical-block4']
            with patch('sys.argv', args):
                main()
                self.assertEqual(json.loads(output.read_text())['validated_prefix_windows'], 36)
                with self.assertRaises(FileExistsError):
                    main()
                output.unlink()
                trace.write_text('truncated')
                with self.assertRaises(ValueError):
                    main()
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
