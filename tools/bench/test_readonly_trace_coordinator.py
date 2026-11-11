"""Integrated fake CPU collection under the real dedicated subreaper; no GPU query."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import readonly_trace_coordinator as coordinator
import readonly_trace_telemetry as telemetry
from readonly_trace_sequence import ORDER
from supervised_worker import supervise
from test_readonly_trace_telemetry import ENV, UUID, SOURCE, fixture_commands

HERE = Path(__file__).resolve().parent


def observer_driver(directory, mode):
    directory = Path(directory)
    telemetry.OBSERVER_CPU = min(os.sched_getaffinity(0))
    telemetry.SECONDS = 5 if mode == 'lifecycle' else 1
    telemetry.STARTUP_SECONDS = 2
    telemetry.CPU_TIMEOUT_SECONDS = 15
    if mode == 'overflow':
        print('x' * (coordinator.OUTPUT_BYTES + 1), flush=True)
        time.sleep(30)
        return
    if mode == 'no_ready':
        time.sleep(30)
        return
    if mode == 'bad_ready':
        directory.mkdir()
        (directory / 'ready.json').write_text('{}')
        time.sleep(30)
        return
    cpu, gpu = fixture_commands(directory, 'gpu_late_failure' if mode == 'observer_failure' else 'complete',
                                telemetry.OBSERVER_CPU, seconds=telemetry.SECONDS)
    result = telemetry.collect(cpu, gpu, cwd=directory.parent, env=ENV, output_dir=directory,
                               expected_cpu_source=SOURCE, gpu_uuid=UUID)
    if mode == 'missing_result':
        (directory / 'telemetry.json').unlink()
    raise SystemExit(0 if result['status'] == 'complete' else 1)


def collection_driver(root, mode):
    root = Path(root)
    affinity = os.sched_getaffinity(0)
    coordinator.OBSERVER_CPU = telemetry.OBSERVER_CPU = min(affinity)
    telemetry.SECONDS = 5 if mode == 'lifecycle' else 1
    coordinator.STARTUP_SECONDS = .25 if mode == 'no_ready' else 3
    coordinator.OBSERVER_SECONDS = .35 if mode == 'deadline' else 10
    if mode in ('validation_expired', 'clock_expired'):
        actual = coordinator.validate_ready
        def expired(*args):
            actual(*args)
            if mode == 'clock_expired':
                original_stamp = coordinator.diagnostic_clock.stamp
                def expire_clock():
                    value = original_stamp()
                    coordinator.STARTUP_SECONDS = 0
                    return value
                coordinator.diagnostic_clock.stamp = expire_clock
            else:
                coordinator.STARTUP_SECONDS = 0
        coordinator.validate_ready = expired
    directory = root / 'collection'
    telemetry_dir = directory / 'telemetry'
    argv = [sys.executable, '-c', 'from test_readonly_trace_coordinator import observer_driver; '
            f'observer_driver({str(telemetry_dir)!r},{mode!r})', str(telemetry_dir)]
    code = ('from pathlib import Path; import time; '
            f'Path({str(directory / "sequence_started")!r}).write_text("started"); '
            f'time.sleep({10 if mode in ("observer_failure", "early_observer", "interrupt", "crash", "deadline") else .12 if mode == "lifecycle" else 0}); '
            f'raise SystemExit({2 if mode == "job_failure" else 0})')
    jobs = [dict(condition=c, argv=[sys.executable, '-c', code], cwd=str(root), env=dict(ENV)) for c in ORDER]
    factory = None
    if mode == 'lifecycle':
        from observe_readonly_process import ProcessObservation
        from readonly_trace_preflight import file_identity
        image = file_identity(Path(sys.executable).resolve())
        injection = file_identity(Path('/usr/bin/true').resolve())
        for job in jobs:
            job['env'].update(LANG='C', LC_ALL='C')
        # All fake children are plain Python, including T-labelled slots. This
        # tests clocks, not injection conditions, CPU2 placement or codec output.
        def factory(_ordinal, job):
            return ProcessObservation(workload=image, workload_argv=job['argv'], expected_env=job['env'],
                                      cpu=coordinator.OBSERVER_CPU, injection=injection, traced=False)
    if mode == 'sequence_exception':
        # Setup failure after ready must clean observers, not leave them running.
        sequence_patch = patch.object(coordinator, 'run_sequence', side_effect=OSError('fake disk failure'))
    else:
        sequence_patch = patch('readonly_trace_sequence.shutil.disk_usage',
                               return_value=SimpleNamespace(free=100 * 1024 ** 3))
    with sequence_patch:
        result = coordinator.coordinate(jobs, argv, cwd=HERE, env=ENV, output_dir=directory,
                                        expected_cpu_source=SOURCE, gpu_uuid=UUID,
                                        process_observer_factory=factory)
    assert os.sched_getaffinity(0) == affinity
    assert not Path(f'/proc/self/task/{os.getpid()}/children').read_text().strip(), 'unreaped coordinator child'
    raise SystemExit(0 if result['status'] == 'complete' else 1)


def supervised_driver(root, mode):
    root = Path(root)
    argv = [sys.executable, '-c', 'from test_readonly_trace_coordinator import collection_driver; '
            f'collection_driver({str(root)!r},{mode!r})']
    result = supervise(argv, cwd=HERE, env=ENV, output_dir=root / 'supervision',
                       timeout_seconds=15, parent_pid=os.getppid())
    assert not Path(f'/proc/self/task/{os.getpid()}/children').read_text().strip()
    print(json.dumps(result))


@unittest.skipUnless(sys.platform == 'linux', 'Linux collection supervision')
class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def launch(self, mode):
        root = self.root / mode
        root.mkdir()
        argv = [sys.executable, '-c', 'from test_readonly_trace_coordinator import supervised_driver; '
                f'supervised_driver({str(root)!r},{mode!r})']
        child = subprocess.Popen(argv, cwd=HERE, env=ENV, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
        return root, child

    def finish(self, child):
        try:
            stdout, stderr = child.communicate(timeout=20)
            self.assertEqual(child.returncode, 0, stderr)
            return json.loads(stdout)
        finally:
            if child.poll() is None:
                child.terminate()
                child.communicate(timeout=10)

    def run_fixture(self, mode):
        root, child = self.launch(mode)
        result = self.finish(child)
        collection = json.loads((root / 'collection/collection.json').read_text())
        return root, result, collection

    def test_success_composes_all_twelve_jobs_and_full_fake_sampler(self):
        root, supervisor, collection = self.run_fixture('complete')
        self.assertEqual(supervisor['status'], 'complete')
        self.assertEqual(collection['status'], 'complete')
        self.assertGreaterEqual(collection['elapsed_seconds_including_cleanup'], 1)
        sequence = json.loads((root / 'collection/sequence/sequence-result.json').read_text())
        self.assertEqual([j['condition'] for j in sequence['jobs']], list(ORDER))
        for field in ('launch_provenance_verified', 'telemetry_contents_qualified',
                      'sequence_coverage_verified', 'benchmark_outputs_qualified'):
            self.assertFalse(collection[field])

    def test_job_failure_preserves_sampler_lifetime_and_stops_later_jobs(self):
        root, supervisor, collection = self.run_fixture('job_failure')
        self.assertEqual(supervisor['status'], 'failed')
        self.assertEqual(collection['sequence_status'], 'failed')
        self.assertEqual(collection['telemetry_status'], 'complete')
        self.assertGreaterEqual(collection['elapsed_seconds_including_cleanup'], 1)
        self.assertFalse((root / 'collection/sequence/02-CP.capture').exists())
        self.assertTrue(supervisor['cleanup']['complete'])

    def test_observer_failure_or_exit_cancels_running_job(self):
        for mode in ('observer_failure', 'early_observer'):
            with self.subTest(mode=mode):
                root, supervisor, collection = self.run_fixture(mode)
                self.assertEqual(collection['status'], 'failed')
                self.assertEqual(supervisor['status'], 'failed')
                sequence = json.loads((root / 'collection/sequence/sequence-result.json').read_text())
                self.assertEqual(len(sequence['jobs']), 1)
                self.assertIn('external_cancelled', sequence['jobs'][0]['receipt']['issues'])
                self.assertFalse((root / 'collection/sequence/02-CP.capture').exists())
                self.assertTrue(supervisor['cleanup']['complete'])

    def test_startup_bad_readiness_deadline_and_sequence_setup_failure(self):
        for mode in ('no_ready', 'bad_ready', 'deadline', 'sequence_exception', 'overflow',
                     'missing_result', 'validation_expired', 'clock_expired'):
            with self.subTest(mode=mode):
                root, supervisor, collection = self.run_fixture(mode)
                self.assertEqual(collection['status'], 'failed')
                self.assertEqual(supervisor['status'], 'failed')
                self.assertTrue(supervisor['cleanup']['complete'])
                if mode != 'missing_result':
                    self.assertFalse((root / 'collection/sequence/02-CP.capture').exists())
                if mode in ('no_ready', 'bad_ready', 'validation_expired', 'clock_expired'):
                    self.assertFalse((root / 'collection/sequence').exists())

    def test_interrupt_and_coordinator_sigkill_cleanup_under_surviving_supervisor(self):
        for mode, sig in (('interrupt', signal.SIGTERM), ('crash', signal.SIGKILL)):
            with self.subTest(mode=mode):
                root, child = self.launch(mode)
                end = time.monotonic() + 5
                try:
                    while not (root / 'collection/sequence_started').exists():
                        self.assertIsNone(child.poll())
                        self.assertLess(time.monotonic(), end)
                        time.sleep(.01)
                    ready = json.loads((root / 'collection/telemetry/ready.json').read_text())
                    coordinator_pid = int(Path(f'/proc/{ready["worker_pid"]}/stat').read_text().split(') ')[1].split()[1])
                    os.kill(coordinator_pid, sig)
                    supervisor = self.finish(child)
                    self.assertEqual(supervisor['status'], 'failed')
                    self.assertTrue(supervisor['cleanup']['complete'])
                    self.assertFalse(Path(f'/proc/{coordinator_pid}').exists())
                    self.assertFalse(Path(f'/proc/{ready["worker_pid"]}').exists())
                    self.assertFalse(Path(f'/proc/{ready["gpu_pid"]}').exists())
                    if mode == 'interrupt':
                        collection = json.loads((root / 'collection/collection.json').read_text())
                        self.assertEqual(collection['interrupt_signal'], signal.SIGTERM)
                    else:
                        self.assertTrue(supervisor['orphan_observed'])
                finally:
                    if child.poll() is None:
                        child.terminate()
                        child.communicate(timeout=10)

    def test_invalid_jobs_and_output_paths_do_not_spawn_observers(self):
        jobs = [dict(condition=c, argv=[sys.executable, '-c', 'pass'], cwd=str(self.root), env=dict(ENV)) for c in ORDER]
        for bad in (jobs[:-1], [dict(j, report_path='/bad') for j in jobs]):
            with patch.object(coordinator.subprocess, 'Popen') as launch, self.assertRaises(ValueError):
                coordinator.coordinate(bad, ['/fake', str(self.root / 'out/telemetry')], cwd=self.root,
                                       env=ENV, output_dir=self.root / 'out', expected_cpu_source=SOURCE, gpu_uuid=UUID)
            launch.assert_not_called()
            self.assertFalse((self.root / 'out').exists())

    def test_ready_rejects_identity_affinity_source_gpu_and_timezone_tampering(self):
        record = dict(schema_version=1, kind='telemetry_ready', worker_pid=123, gpu_pid=124,
                      observer_affinity=[0], local_time_with_offset='2026-11-11T12:00:00+00:00',
                      cpu_metadata=dict(kind='metadata', schema_version=1, source_sha256=SOURCE,
                                        target_cpu=2, observer_affinity=[0], seconds=900, interval_seconds=1),
                      first_gpu_row=['2026/11/11 12:00:00.000', UUID, '0 %', '1 MiB', '40', '20 W', '200 MHz', '400 MHz'])
        coordinator.validate_ready(record, 123, SOURCE, UUID)
        bad_records = [dict(record, worker_pid=125), dict(record, schema_version=True),
                       dict(record, observer_affinity=[2]), dict(record, gpu_pid=123),
                       dict(record, local_time_with_offset='2026-11-11T12:00:00'),
                       dict(record, cpu_metadata=dict(record['cpu_metadata'], source_sha256='b' * 64)),
                       dict(record, first_gpu_row=[record['first_gpu_row'][0], 'GPU-wrong', *record['first_gpu_row'][2:]])]
        for bad in bad_records:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                coordinator.validate_ready(bad, 123, SOURCE, UUID)

    def test_lost_ownership_never_signals_and_read_limits_reject(self):
        child = SimpleNamespace(pid=123, poll=Mock())
        with patch.object(coordinator.os, 'waitid', side_effect=ChildProcessError), \
                patch.object(coordinator.os, 'killpg') as kill:
            result = coordinator.finish_observer(child, True)
        self.assertFalse(result['verified'])
        kill.assert_not_called()
        path = self.root / 'large.json'
        path.write_bytes(b'x' * (coordinator.JSON_BYTES + 1))
        with self.assertRaises(ValueError):
            coordinator.read_record(path)


if __name__ == '__main__':
    unittest.main()
