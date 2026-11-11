"""Offline tests only: fake CPU children or mocked child calls, never GPU work."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import readonly_trace_sequence as sequence


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'sequence'
        self.jobs = [dict(condition=condition,
                          argv=[sys.executable, '-c', 'print("fake")'],
                          cwd=str(self.root),
                          env={'PATH': os.defpath, 'PYTHONNOUSERSITE': '1'})
                     for condition in sequence.ORDER]
        disk = patch.object(sequence.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 1024 ** 3))
        self.disk = disk.start()
        self.addCleanup(disk.stop)

    def test_fixed_order_real_fake_children(self):
        result = sequence.run_sequence(self.jobs, self.output)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual([j['condition'] for j in result['jobs']], list(sequence.ORDER))
        self.assertEqual(result['captured_bytes'], 12 * len(b'fake\n'))
        self.assertEqual(len({j['receipt']['pid'] for j in result['jobs']}), 12)
        for path in self.output.glob('*.capture'):
            self.assertEqual(path.read_bytes(), b'fake\n')
        plan = json.loads((self.output / 'sequence-plan.json').read_text())
        self.assertFalse(plan['launch_readiness_verified'])
        self.assertNotIn('PYTHONNOUSERSITE', (self.output / 'sequence-plan.json').read_text())
        for field in ('launch_provenance_verified', 'benchmark_outputs_qualified',
                      'whole_run_supervision_provided'):
            self.assertFalse(result[field])
        self.assertEqual(plan['sequence_seconds'], 600)
        self.assertEqual(plan['child_seconds'], 120)

    def test_sequence_reports_get_distinct_paths_and_receipt_hashes(self):
        for ordinal, job in enumerate(self.jobs, 1):
            report = self.output / f'{ordinal:02d}-{job["condition"]}.report.json'
            job['report_path'] = str(report)
            job['argv'] = [sys.executable, '-c',
                           'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text("{}"); print("trace")',
                           str(report)]
        result = sequence.run_sequence(self.jobs, self.output)
        self.assertEqual(result['status'], 'complete')
        reports = [j['receipt']['report_capture'] for j in result['jobs']]
        self.assertEqual(len({r['path'] for r in reports}), 12)
        self.assertTrue(all(r['post_exit_hash_verified'] for r in reports))
        self.assertTrue(all(not r['producer_identity_verified'] for r in reports))

    def test_duplicate_report_destination_rejected_before_sequence(self):
        report = str(self.output / '01-AP.report.json')
        for job in self.jobs:
            job['report_path'] = report
            job['argv'].append(report)
        with patch.object(sequence, 'run_bounded') as child:
            with self.assertRaisesRegex(ValueError, 'unique sequence-owned'):
                sequence.run_sequence(self.jobs, self.output)
            child.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_validate_all_before_any_launch_or_output(self):
        for modification in ('order', 'count', 'argv', 'env', 'cwd', 'extra'):
            jobs = sequence.validate_jobs(self.jobs)
            if modification == 'order':
                jobs[-1]['condition'] = 'CT'
            elif modification == 'count':
                jobs.pop()
            elif modification == 'argv':
                jobs[-1]['argv'] = ['embedded\0nul']
            elif modification == 'env':
                jobs[-1]['env'] = {'BAD=KEY': 'value'}
            elif modification == 'cwd':
                jobs[-1]['cwd'] = self.root / 'missing'
            else:
                jobs[-1]['extra'] = True
            with self.subTest(modification=modification), patch.object(sequence, 'run_bounded') as child:
                with self.assertRaises((ValueError, FileNotFoundError)):
                    sequence.run_sequence(jobs, self.output)
                child.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_preflight_space_and_exclusive_directory(self):
        self.disk.return_value.free = sequence.MIN_FREE_BYTES - 1
        with self.assertRaisesRegex(ValueError, '12 GiB'):
            sequence.run_sequence(self.jobs, self.output)
        self.assertFalse(self.output.exists())
        self.disk.return_value.free = sequence.MIN_FREE_BYTES
        self.output.mkdir()
        marker = self.output / 'preserve'
        marker.write_text('untouched')
        with patch.object(sequence, 'run_bounded') as child:
            with self.assertRaises(FileExistsError):
                sequence.run_sequence(self.jobs, self.output)
            child.assert_not_called()
        self.assertEqual(marker.read_text(), 'untouched')

    def test_failure_stops_without_retry(self):
        self.jobs[1]['argv'] = [sys.executable, '-c', 'raise SystemExit(7)']
        result = sequence.run_sequence(self.jobs, self.output)
        self.assertEqual(result['stop_reason'], 'child_failed')
        self.assertEqual(len(result['jobs']), 2)
        self.assertEqual(result['jobs'][-1]['receipt']['returncode'], 7)
        self.assertFalse((self.output / '03-AT.capture').exists())

    def test_aggregate_remaining_cap_and_exhaustion(self):
        # Small byte budgets test arithmetic without allocating large files.
        with patch.object(sequence, 'CAPTURE_BYTES', 4), patch.object(sequence, 'TOTAL_BYTES', 6):
            with patch.object(sequence, 'run_bounded', side_effect=[
                    {'status': 'complete', 'captured_bytes': 4},
                    {'status': 'complete', 'captured_bytes': 2}]) as child:
                result = sequence.run_sequence(self.jobs, self.output)
        self.assertEqual(result['stop_reason'], 'aggregate_output_limit')
        self.assertEqual(result['captured_bytes'], 6)
        self.assertEqual([c.kwargs['output_limit_bytes'] for c in child.call_args_list], [4, 2])
        self.assertEqual(len({c.kwargs['sequence_deadline'] for c in child.call_args_list}), 1)

    def test_streaming_overflow_stops_sequence(self):
        with patch.object(sequence, 'CAPTURE_BYTES', 4):
            result = sequence.run_sequence(self.jobs, self.output)
        self.assertEqual(result['stop_reason'], 'child_failed')
        self.assertEqual(result['captured_bytes'], 4)
        self.assertEqual(len(result['jobs']), 1)
        self.assertIn('output_limit', result['jobs'][0]['receipt']['issues'])

    def test_deadline_consumed_during_setup_prevents_launch(self):
        with patch.object(sequence.time, 'monotonic', side_effect=[0, 601, 601]), \
                patch.object(sequence, 'run_bounded') as child:
            result = sequence.run_sequence(self.jobs, self.output)
        child.assert_not_called()
        self.assertEqual(result['stop_reason'], 'sequence_deadline')
        self.assertTrue((self.output / 'sequence-plan.json').is_file())

    def test_free_space_rechecked_between_children(self):
        self.disk.side_effect = [SimpleNamespace(free=sequence.MIN_FREE_BYTES),
                                SimpleNamespace(free=sequence.MIN_FREE_BYTES),
                                SimpleNamespace(free=sequence.MIN_FREE_BYTES - 5)]
        with patch.object(sequence, 'run_bounded', return_value={
                'status': 'complete', 'captured_bytes': 4}) as child:
            result = sequence.run_sequence(self.jobs, self.output)
        self.assertEqual(child.call_count, 1)
        self.assertEqual(result['stop_reason'], 'free_space')

    def test_unhandled_interrupt_preserves_partial_and_does_not_retry(self):
        with patch.object(sequence, 'run_bounded', side_effect=KeyboardInterrupt) as child:
            with self.assertRaises(KeyboardInterrupt):
                sequence.run_sequence(self.jobs, self.output)
        self.assertEqual(child.call_count, 1)
        self.assertTrue((self.output / 'sequence-plan.json').is_file())
        self.assertFalse((self.output / 'sequence-result.json').exists())


if __name__ == '__main__':
    unittest.main()
