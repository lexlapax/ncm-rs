"""Real CPU-only children; frozen GPU commands/identity observation are substituted."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import clock_codec_sequence as sequence
from run_phase2_linux_comparison import plan_jobs

FAKE = '''import json,os,sys,time
from pathlib import Path
mode,report=sys.argv[1:]
if mode=='timeout':time.sleep(30)
if mode=='failure':raise SystemExit(7)
if mode!='missing':Path(report).write_text(json.dumps({'fake':True,'env':dict(os.environ)}))
if mode=='overflow':sys.stdout.write('x'*8192);sys.stdout.flush();time.sleep(30)
time.sleep(.1)
'''


class Observer:
    def __init__(self, failed=False, after_finish=None):
        self.pids = set()
        self.failed, self.after_finish = failed, after_finish

    def observe(self, pid):
        if os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
            return
        if os.sched_getaffinity(pid) != {2}:
            return  # The taskset launcher has not yet applied its requested mask.
        self.pids.add(pid)

    def finish(self, receipt, path):
        if receipt['pid'] is not None:
            if Path(f'/proc/{receipt["pid"]}').exists():
                raise AssertionError('child not reaped before observation finish')
        result = {'status': 'failed' if self.failed or not self.pids else 'complete',
                  'fake_observer': True, 'pids': sorted(self.pids)}
        with path.open('x') as file:
            json.dump(result, file)
        if self.after_finish:
            self.after_finish()
        return result


@unittest.skipUnless(sys.platform == 'linux' and shutil.which('taskset')
                     and 2 in os.sched_getaffinity(0), 'Linux with CPU2 and taskset required')
class ClockSequenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.previous, self.binaries = self.root / 'previous', self.root / 'binaries'
        self.previous.mkdir(); self.binaries.mkdir()
        self.output = self.root / 'collection'
        self.script = self.root / 'fake.py'
        self.script.write_text(FAKE)
        self.env = {'PATH': os.defpath, 'HOME': str(self.root), 'LANG': 'C', 'LC_ALL': 'C',
                    'PYTHONNOUSERSITE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
        self.taskset = str(Path(shutil.which('taskset')).resolve())
        self.jobs = plan_jobs(self.previous, self.binaries, self.output, root=self.root,
                              repeats=6, workloads=('codec',), cpu=2, taskset=self.taskset)
        for job in self.jobs:
            job['argv'] = [self.taskset, '-c', '2', sys.executable, '-S', str(self.script), 'ok', job['report']]
        self.observers = []

    def factory(self, ordinal, job, env):
        self.assertEqual(env, self.env)
        observer = Observer()
        self.observers.append(observer)
        return observer

    def run_sequence(self, **changes):
        options = dict(root=self.root, env=self.env, work_deadline=time.monotonic() + 30,
                       cancel_check=lambda: False, process_observer_factory=self.factory,
                       taskset=self.taskset)
        with patch.object(sequence, 'plan_jobs', return_value=self.jobs):
            return sequence.run_sequence(self.previous, self.binaries, self.output, **(options | changes))

    def test_all_eighteen_real_fake_children_bound_reports_and_deferred_python_copies(self):
        result = self.run_sequence()
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(result['jobs']), 18)
        self.assertEqual(len(self.observers), 18)
        self.assertFalse(result['benchmark_outputs_qualified'])
        self.assertFalse(result['whole_run_supervision_provided'])
        self.assertEqual(self.output.stat().st_mode & 0o777, 0o700)
        for job, observed in zip(self.jobs, result['jobs']):
            self.assertEqual(observed['receipt']['argv'], job['argv'])
            self.assertTrue(observed['receipt']['report_capture']['post_exit_hash_verified'])
            self.assertTrue(observed['receipt']['process_observer_enabled'])
            self.assertEqual(json.loads(Path(job['report']).read_bytes())['env'], self.env)
            for destination in job['python_copies']:
                self.assertFalse(Path(destination).exists())
        self.assertEqual(result, json.loads((self.output / 'sequence-result.json').read_bytes()))
        with self.assertRaisesRegex(ValueError, 'fresh output'):
            self.run_sequence()

    def test_failure_missing_report_overflow_and_deadline_stop_without_retry(self):
        for mode in ('failure', 'missing', 'overflow', 'timeout'):
            with self.subTest(mode=mode):
                self.output = self.root / mode
                # Rebuild canonical paths for this isolated attempt.
                self.jobs = plan_jobs(self.previous, self.binaries, self.output, root=self.root,
                                      repeats=6, workloads=('codec',), cpu=2, taskset=self.taskset)
                for job in self.jobs:
                    job['argv'] = [self.taskset, '-c', '2', sys.executable, '-S', str(self.script), mode, job['report']]
                with patch.object(sequence, 'CAPTURE_BYTES', 1024):
                    result = self.run_sequence(work_deadline=time.monotonic() + (.3 if mode == 'timeout' else 30))
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(len(result['jobs']), 1)
                self.assertEqual(result['stop_reason'], 'child_failed')
                receipt = result['jobs'][0]['receipt']
                self.assertTrue(receipt['direct_child_reaped'])
                self.assertLessEqual(receipt['captured_bytes'], 1024)
                self.assertFalse(Path(self.jobs[1]['report']).exists())

    def test_late_observation_failure_stops_before_second_job(self):
        def factory(*_):
            return Observer(failed=True)
        result = self.run_sequence(process_observer_factory=factory)
        self.assertEqual(result['stop_reason'], 'process_observation_failed')
        self.assertEqual(len(result['jobs']), 1)
        self.assertEqual(result['jobs'][0]['receipt']['status'], 'complete')
        self.assertFalse(Path(self.jobs[1]['report']).exists())

    def test_cancellation_after_receipt_stops_submission(self):
        cancelled = False
        def finish():
            nonlocal cancelled
            cancelled = True
        result = self.run_sequence(cancel_check=lambda: cancelled,
                                   process_observer_factory=lambda *_: Observer(after_finish=finish))
        self.assertEqual(result['stop_reason'], 'external_cancelled')
        self.assertEqual(len(result['jobs']), 1)
        self.assertTrue(result['external_cancelled'])
        self.assertEqual(result['jobs'][0]['receipt']['status'], 'complete')

    def test_live_cancellation_kills_current_child_and_stops_later_jobs(self):
        result = self.run_sequence(cancel_check=lambda: Path(self.jobs[0]['report']).exists())
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(result['jobs']), 1)
        self.assertTrue(result['external_cancelled'])
        self.assertIn('external_cancelled', result['jobs'][0]['receipt']['issues'])
        self.assertTrue(result['jobs'][0]['receipt']['direct_child_reaped'])
        self.assertFalse(Path(self.jobs[1]['report']).exists())

    def test_check_exception_is_retained_without_spawning(self):
        def failure():
            raise OSError('synthetic observer channel error')
        result = self.run_sequence(cancel_check=failure)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['stop_reason'], 'sequence_error:OSError')
        self.assertTrue(result['external_cancelled'])
        self.assertIsNone(result['jobs'][0]['receipt'])
        self.assertFalse(Path(self.jobs[0]['report']).exists())

    def test_order_and_free_space_guards_precede_output_creation(self):
        self.jobs[-1]['repeat'] = 5
        with self.assertRaisesRegex(ValueError, 'fixed codec order'):
            self.run_sequence()
        self.assertFalse(self.output.exists())
        self.jobs[-1]['repeat'] = 6
        with patch.object(sequence.shutil, 'disk_usage', return_value=type('Space', (), {'free': 0})()):
            with self.assertRaisesRegex(ValueError, 'free space'):
                self.run_sequence()
        self.assertFalse(self.output.exists())

    def test_invalid_late_observer_or_executable_rejected_before_any_outputs(self):
        for failure in ('observer', 'executable'):
            with self.subTest(failure=failure):
                if failure == 'executable':
                    self.jobs[-1]['argv'][3] = str(self.root / 'absent')
                factory = lambda ordinal, *_: None if failure == 'observer' and ordinal == 18 else Observer()
                with self.assertRaises(ValueError):
                    self.run_sequence(process_observer_factory=factory)
                self.assertFalse(self.output.exists())

    def test_invalid_environment_deadline_or_missing_monitor_fails_before_outputs(self):
        for change in ({'env': self.env | {'LD_PRELOAD': '/untrusted'}},
                       {'env': self.env | {'HOME': 'relative'}}, {'cancel_check': None},
                       {'process_observer_factory': None}, {'work_deadline': time.monotonic() - 1},
                       {'work_deadline': float('nan')}, {'work_deadline': time.monotonic() + 601}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_sequence(**change)
            self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
