"""Short synthetic telemetry plus real restoration/worker ownership; no GPU commands."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import clock_diagnostic_coordinator as clock
import readonly_trace_coordinator as shared
import readonly_trace_telemetry as telemetry
from supervised_worker import cleanup_owned, direct_children, subreaper
from test_clock_restoration_composition import POLICY, WORKER, WORKLOAD
from test_readonly_trace_telemetry import ENV, SOURCE, UUID

HERE = Path(__file__).resolve().parent


def supervised_observer(directory, mode):
    from supervised_worker import supervise
    directory = Path(directory)
    worker = [sys.executable, '-S', '-c',
              'from test_clock_diagnostic_coordinator import telemetry_worker; '
              f'telemetry_worker({str(directory)!r},{mode!r})']
    result = supervise(worker, cwd=HERE, env=dict(os.environ),
                       output_dir=directory.parent / 'observer-supervision',
                       timeout_seconds=15, parent_pid=os.getppid())
    assert direct_children() == [] and subreaper() == 0
    raise SystemExit(0 if result['status'] == 'complete' else 1)


def telemetry_worker(directory, mode):
    from test_readonly_trace_coordinator import observer_driver
    if mode == 'observer_crash':
        actual = telemetry.run_bounded
        def capture(*args, **kwargs):
            observe = kwargs['cancel_check']
            def fail_after_work_starts():
                if (Path(directory).parent / 'sequence_started').exists():
                    os.kill(os.getpid(), signal.SIGKILL)  # Own process, not a borrowed numeric PID.
                return observe()
            kwargs['cancel_check'] = fail_after_work_starts
            return actual(*args, **kwargs)
        telemetry.run_bounded = capture
        mode = 'lifecycle'
    observer_driver(directory, mode)


def collection_driver(root, mode):
    root = Path(root)
    directory = root / 'collection'
    initial = os.sched_getaffinity(0)
    shared.OBSERVER_CPU = telemetry.OBSERVER_CPU = 0
    shared.STARTUP_SECONDS = 3
    shared.OBSERVER_SECONDS = 15
    observer_mode = (mode if mode in ('observer_failure', 'bad_ready', 'observer_crash') else 'lifecycle')
    telemetry.SECONDS = 5 if observer_mode in ('lifecycle', 'observer_crash') else 1
    env = {'PATH': os.defpath, 'HOME': str(root), 'LANG': 'C', 'LC_ALL': 'C',
           'PYTHONNOUSERSITE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
    for name in ('previous', 'binaries'):
        (root / name).mkdir()
    policy = POLICY.replace("root/'restoration/obligation.json'", "root/'collection/restoration/obligation.json'")
    policy = policy.replace("if stage=='set':", "if stage=='set':\n    assert (root/'collection/telemetry/ready.json').is_file()")
    worker = WORKER.replace("root/'restoration/obligation.json'", "root/'collection/restoration/obligation.json'")
    worker = worker.replace("output=root/'collection'", "output=root/'collection/sequence'")
    workload = WORKLOAD.replace("if mode=='failure' and ordinal==2:",
                                "(root/'collection/sequence_started').touch()\nif mode=='failure' and ordinal==2:")
    for name, content in (('policy.py', policy), ('worker.py', worker), ('workload.py', workload)):
        (root / name).write_text(content)
    python = str(Path(sys.executable).resolve())
    policy_argv = [python, '-S', str(root / 'policy.py'), str(root)]
    worker_mode = ('deadline' if mode in ('observer_failure', 'interrupt', 'parent_loss', 'observer_crash') else
                   'failure' if mode == 'worker_failure' else 'success')
    worker_argv = [python, '-S', str(root / 'worker.py'), str(HERE), str(root), worker_mode,
                   str(Path(shutil.which('taskset')).resolve())]
    restorer_code = f'''import json,os
from supervised_restoration import supervise_restoration
from supervised_worker import direct_children,subreaper
result=supervise_restoration({policy_argv + ['set']!r},{worker_argv!r},{policy_argv + ['reset']!r},
    cwd={str(HERE)!r},env={env!r},output_dir={str(directory / 'restoration')!r},
    parent_pid=os.getppid(),lock_seconds={600 if mode == 'lifecycle_validation' else 10},
    cleanup_reserve_seconds={10 if mode == 'lifecycle_validation' else 1},
    command_seconds={10 if mode == 'lifecycle_validation' else 1})
assert direct_children()==[] and subreaper()==0,result
print(json.dumps({{'status':result['status']}}),flush=True)
raise SystemExit(0 if result['status']=='complete' else 1)
'''
    restorer = [python, '-S', '-c', restorer_code, str(directory / 'restoration')]
    observer = [python, '-S', '-c', 'from test_clock_diagnostic_coordinator import supervised_observer; '
                f'supervised_observer({str(directory / "telemetry")!r},{observer_mode!r})', str(directory / 'telemetry')]
    # No generic adoption-and-SIGKILL ancestor around the clock restorer.
    result = clock.coordinate(restorer, observer, cwd=HERE, env=env, output_dir=directory,
                              expected_cpu_source=SOURCE, gpu_uuid=UUID)
    assert os.sched_getaffinity(0) == initial
    assert direct_children() == []
    print(json.dumps(result))


def parent_loss_driver(root):
    """Test-only adoption/reaping, never generic KILL cleanup on the successful path."""
    root = Path(root)
    assert subreaper() == 0 and direct_children() == []
    subreaper(1)
    unrelated = subprocess.Popen([sys.executable, '-S', '-c', 'import time;time.sleep(40)'], env=ENV)
    child = subprocess.Popen([sys.executable, '-S', '-c',
                             'from test_clock_diagnostic_coordinator import collection_driver; '
                             f'collection_driver({str(root)!r},"parent_loss")'], cwd=HERE, env=ENV,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    reaped = []
    try:
        deadline = time.monotonic() + 5
        while not (root / 'collection/sequence_started').exists():
            assert child.poll() is None and time.monotonic() < deadline, 'startup failed'
            time.sleep(.01)
        child.kill()  # Only the owned coordinator, not its sessions/descendants.
        stdout, stderr = child.communicate(timeout=5)
        assert child.returncode == -signal.SIGKILL, (stdout, stderr)
        deadline = time.monotonic() + 20
        while set(direct_children()) != {unrelated.pid}:
            assert time.monotonic() < deadline, direct_children()
            for pid in direct_children():
                if pid != unrelated.pid:
                    exited = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                    if exited is not None:
                        waited, status = os.waitpid(pid, 0)
                        reaped.append({'pid': waited, 'returncode': os.waitstatus_to_exitcode(status)})
            time.sleep(.01)
        assert unrelated.poll() is None
        assert len(reaped) == 2, reaped  # Separate observer/restoration supervisors only.
        print(json.dumps({'reaped': reaped, 'unrelated_survived': True}))
    finally:
        # Test failure cleanup is fake-policy-only; it is not the production ancestry.
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=5)
        if unrelated.poll() is None:
            unrelated.terminate()
        unrelated.wait(timeout=5)
        assert cleanup_owned()['complete']
        subreaper(0)


@unittest.skipUnless(sys.platform == 'linux' and shutil.which('taskset')
                     and {0, 2} <= os.sched_getaffinity(0), 'Linux with CPU0/CPU2 and taskset required')
class ClockCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_case(self, mode):
        root = self.root / mode
        root.mkdir()
        script = ('from test_clock_diagnostic_coordinator import collection_driver; '
                  f'collection_driver({str(root)!r},{mode!r})')
        command = [sys.executable, '-S', '-c', script]
        if mode == 'interrupt':
            with subprocess.Popen(command, cwd=HERE, env=ENV, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True) as child:
                deadline = time.monotonic() + 5
                while not (root / 'collection/sequence_started').exists():
                    self.assertIsNone(child.poll(), 'coordinator exited before workload startup')
                    self.assertLess(time.monotonic(), deadline, 'workload startup timed out')
                    time.sleep(.01)
                child.send_signal(signal.SIGTERM)
                stdout, stderr = child.communicate(timeout=25)
                self.assertEqual(child.returncode, 0, stdout + stderr)
        else:
            run = subprocess.run(command, cwd=HERE, env=ENV, capture_output=True, text=True, timeout=35)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            stdout = run.stdout
        result = json.loads(stdout)
        self.assertEqual(result, json.loads((root / 'collection/collection.json').read_bytes()))
        for key in ('launch_provenance_verified', 'telemetry_contents_qualified',
                    'sequence_coverage_verified', 'benchmark_outputs_qualified'):
            self.assertFalse(result[key])
        self.assertNotIn('sequence_status', result)
        self.assertIn('restoration-aware ancestry and independent observer cleanup', result['scope'])
        self.assertFalse(Path(f'/proc/{result["observer_pid"]}').exists())
        self.assertTrue(result['observer_shutdown']['verified'])
        supervision = json.loads((root / 'collection/observer-supervision/supervisor.json').read_bytes())
        self.assertTrue(supervision['cleanup']['complete'])
        self.assertTrue(supervision['subreaper_restored'])
        for reaped in supervision['cleanup']['reaped']:
            self.assertFalse(Path(f'/proc/{reaped["pid"]}').exists())
        return root, result

    def restored(self, root):
        result = json.loads((root / 'collection/restoration/restoration.json').read_bytes())
        self.assertTrue(result['reset_command_succeeded'])
        self.assertTrue(result['subreaper_restored'])
        self.assertFalse(result['hardware_policy_restoration_verified'])
        self.assertFalse((root / 'fake-lock').exists())
        self.assertEqual((root / 'policy-events').read_text().splitlines(), ['set', 'reset'])
        for pid in (root / 'workload-pids').read_text().splitlines():
            self.assertFalse(Path('/proc/' + pid).exists())
        return result

    def test_ready_then_eighteen_jobs_and_reset_preserve_full_sampler(self):
        root, result = self.run_case('success')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['restoration_supervisor_status'], 'complete')
        self.assertEqual(result['telemetry_status'], 'complete')
        self.assertGreaterEqual(result['elapsed_seconds_including_cleanup'], 5)
        self.assertEqual(self.restored(root)['status'], 'complete')
        sequence = json.loads((root / 'collection/sequence/sequence-result.json').read_bytes())
        self.assertEqual(len(sequence['jobs']), 18)
        self.assertEqual(sequence['status'], 'complete')

    def test_worker_failure_still_resets_and_preserves_full_sampler(self):
        root, result = self.run_case('worker_failure')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['restoration_supervisor_status'], 'failed')
        self.assertEqual(result['telemetry_status'], 'complete')
        self.assertGreaterEqual(result['elapsed_seconds_including_cleanup'], 5)
        self.assertEqual(self.restored(root)['worker']['returncode'], 1)
        sequence = json.loads((root / 'collection/sequence/sequence-result.json').read_bytes())
        self.assertEqual(len(sequence['jobs']), 2)

    def test_observer_failure_cancels_inflight_work_but_not_restoration(self):
        root, result = self.run_case('observer_failure')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['restoration_supervisor_status'], 'failed')
        self.assertEqual(result['telemetry_status'], 'failed')
        self.assertEqual(self.restored(root)['status'], 'failed')
        self.assertEqual(len((root / 'workload-pids').read_text().splitlines()), 1)
        capture = json.loads((root / 'collection/restorer.receipt.json').read_bytes())
        self.assertEqual(capture['termination_signal'], 'sent')
        self.assertFalse(capture['termination_grace_exhausted'])

    def test_coordinator_interrupt_allows_reset_before_observer_cleanup(self):
        root, result = self.run_case('interrupt')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['interrupt_signal'], signal.SIGTERM)
        self.assertEqual(result['restoration_supervisor_status'], 'failed')
        self.assertEqual(self.restored(root)['status'], 'failed')
        self.assertEqual(len((root / 'workload-pids').read_text().splitlines()), 1)
        capture = json.loads((root / 'collection/restorer.receipt.json').read_bytes())
        self.assertEqual(capture['termination_signal'], 'sent')
        self.assertFalse(capture['termination_grace_exhausted'])

    def test_observer_worker_crash_cleans_escaped_samplers_and_restores(self):
        root, result = self.run_case('observer_crash')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(self.restored(root)['status'], 'failed')
        supervision = json.loads((root / 'collection/observer-supervision/supervisor.json').read_bytes())
        self.assertEqual(supervision['worker']['returncode'], -signal.SIGKILL)
        self.assertTrue(supervision['orphan_observed'])
        self.assertEqual(len(supervision['cleanup']['reaped']), 2)
        self.assertEqual(len((root / 'workload-pids').read_text().splitlines()), 1)

    def test_coordinator_loss_leaves_two_independent_cleanup_branches(self):
        root = self.root / 'parent_loss'
        root.mkdir()
        script = ('from test_clock_diagnostic_coordinator import parent_loss_driver; '
                  f'parent_loss_driver({str(root)!r})')
        run = subprocess.run([sys.executable, '-S', '-c', script], cwd=HERE, env=ENV,
                             capture_output=True, text=True, timeout=35)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        summary = json.loads(run.stdout)
        self.assertTrue(summary['unrelated_survived'])
        for reaped in summary['reaped']:
            self.assertFalse(Path(f'/proc/{reaped["pid"]}').exists())
            # Parent loss closes the restorer's output pipe: Python may exit 120
            # while flushing, rather than 1. Neither is success; require retained
            # cleanup evidence below instead of inferring it from an exit code.
            self.assertNotEqual(reaped['returncode'], 0)
        restoration = self.restored(root)
        self.assertEqual(restoration['status'], 'failed')
        self.assertTrue(restoration['parent_lost'])
        supervision = json.loads((root / 'collection/observer-supervision/supervisor.json').read_bytes())
        self.assertEqual(supervision['status'], 'failed')
        self.assertTrue(supervision['parent_lost'])
        self.assertTrue(supervision['cleanup']['complete'])
        self.assertTrue(supervision['subreaper_restored'])
        self.assertEqual(len(supervision['cleanup']['reaped']), 2)
        for reaped in supervision['cleanup']['reaped']:
            self.assertFalse(Path(f'/proc/{reaped["pid"]}').exists())
        self.assertFalse((root / 'collection/collection.json').exists())

    def test_readiness_rejects_wrong_branch_or_dead_worker(self):
        good = ['456', '456 (python name ) suffix) S 123']
        bad = [['789', good[1]], ['456 789', good[1]],
               ['456', '456 (python) S 789'], ['456', '456 (python) Z 123'],
               ['456', '456 (python) X 123'], ['456', '789 (python) S 123']]
        with patch.object(clock, 'validate_ready'):
            with patch.object(Path, 'read_text', side_effect=good):
                clock.validate_supervised_ready({'worker_pid': 456}, 123, SOURCE, UUID)
            for snapshots in bad:
                with self.subTest(snapshots=snapshots), patch.object(Path, 'read_text', side_effect=snapshots):
                    with self.assertRaises(ValueError):
                        clock.validate_supervised_ready({'worker_pid': 456}, 123, SOURCE, UUID)
            for pid in (123, 0, True, '456'):
                with self.subTest(pid=pid), self.assertRaises(ValueError):
                    clock.validate_supervised_ready({'worker_pid': pid}, 123, SOURCE, UUID)

    def test_invalid_readiness_never_attempts_policy_or_work(self):
        root, result = self.run_case('bad_ready')
        self.assertEqual(result['status'], 'failed')
        self.assertIsNone(result['restoration_supervisor_status'])
        self.assertFalse((root / 'policy-events').exists())
        self.assertFalse((root / 'collection/restoration').exists())
        self.assertFalse((root / 'collection/restorer.capture').exists())

    def test_invalid_restorer_arguments_prevent_observer_spawn(self):
        directory = self.root / 'collection'
        for argv in (['/bin/true'], ['relative', str(directory / 'restoration')], None):
            with self.subTest(argv=argv), patch.object(shared.subprocess, 'Popen') as launch:
                with self.assertRaises(ValueError):
                    clock.coordinate(argv, ['/fake', str(directory / 'telemetry')], cwd=HERE, env=ENV,
                                     output_dir=directory, expected_cpu_source=SOURCE, gpu_uuid=UUID)
                launch.assert_not_called()
                self.assertFalse(directory.exists())


if __name__ == '__main__':
    unittest.main()
