"""Short synthetic observers only; never invokes nvidia-smi or real telemetry."""
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

import readonly_trace_telemetry as telemetry
from readonly_trace_sequence import ORDER, run_sequence

ENV = {'PATH': os.defpath, 'PYTHONNOUSERSITE': '1'}
UUID = 'GPU-12345678-1234-1234-1234-123456789abc'
SOURCE = 'a' * 64
HERE = Path(__file__).parent


def fixture_commands(directory, mode, observer, seconds=1):
    metadata = dict(kind='metadata', schema_version=1, source_sha256=SOURCE, target_cpu=2,
                    observer_affinity=[observer], seconds=seconds, interval_seconds=1)
    if mode == 'bad_source':
        metadata['source_sha256'] = 'b' * 64
    cpu_code = f'''
import json,sys,time
from datetime import datetime,timezone
with open(sys.argv[1],'x') as f:
    def emit(row): f.write(json.dumps(row)+'\\n'); f.flush()
    def sample(index):
        before=time.monotonic_ns()
        utc=datetime.now(timezone.utc).isoformat()
        after=time.monotonic_ns()
        emit(dict(kind='sample',index=index,monotonic_start_ns=before,monotonic_end_ns=after,utc=utc))
    emit({metadata!r})
    sample(0)
    for index in range(1,{seconds + 1}):
        time.sleep({10 if mode in ('gpu_late_failure', 'interrupt') else 1.05})
        sample(index)
    if {mode!r}!='incomplete':
        emit({dict(kind='completion', status='complete', samples=seconds + 1, actual_elapsed_seconds=seconds * 1.05)!r})
'''
    row = ['2026/11/11 12:00:00.000', UUID, '0 %', '1 MiB', '40', '20 W', '200 MHz', '400 MHz']
    gpu_code = ('import sys,time; '
                f'print({", ".join(telemetry.GPU_HEADER)!r},flush=True); '
                f'print({", ".join(row)!r},flush=True); time.sleep(30)')
    if mode == 'gpu_failure':
        gpu_code = 'raise SystemExit(2)'
    elif mode == 'gpu_late_failure':
        marker = directory.parent / 'sequence_started'
        gpu_code = gpu_code.replace('time.sleep(30)',
                                    f'\nfrom pathlib import Path\nwhile not Path({str(marker)!r}).exists(): time.sleep(.01)\nraise SystemExit(2)')
    elif mode == 'startup':
        gpu_code = 'import time; time.sleep(30)'
    elif mode == 'overflow':
        gpu_code = 'import sys,time; print("x"*10000,flush=True); time.sleep(30)'
    return [sys.executable, '-c', cpu_code, str(directory / 'cpu.jsonl')], [sys.executable, '-c', gpu_code]


def fixture_driver(root, mode):
    root = Path(root)
    initial = os.sched_getaffinity(0)
    telemetry.OBSERVER_CPU = min(initial)
    telemetry.SECONDS = 1
    telemetry.CPU_TIMEOUT_SECONDS = 15 if mode in ('gpu_late_failure', 'interrupt') else 3
    telemetry.STARTUP_SECONDS = .15 if mode in ('startup', 'delayed_readiness') else 2
    if mode == 'delayed_readiness':
        actual = telemetry.validate_gpu_prefix
        def delayed(lines, uuid):
            value = actual(lines, uuid)
            time.sleep(.25)
            return value
        telemetry.validate_gpu_prefix = delayed
    if mode == 'overflow':
        telemetry.FILE_LIMIT = 4096
    cpu, gpu = fixture_commands(root / 'observers', mode, telemetry.OBSERVER_CPU)
    result = telemetry.collect(cpu, gpu, cwd=root, env=ENV, output_dir=root / 'observers',
                               expected_cpu_source=SOURCE, gpu_uuid=UUID)
    assert os.sched_getaffinity(0) == initial
    try:
        os.waitpid(-1, os.WNOHANG)
    except ChildProcessError:
        pass
    else:
        raise AssertionError('observer child was not reaped')
    if result['ready']:
        ready = json.loads((root / 'observers/ready.json').read_text())
        assert ready['worker_pid'] == os.getpid()
        assert ready['observer_affinity'] == [telemetry.OBSERVER_CPU]
    return result


@unittest.skipUnless(sys.platform == 'linux', 'Linux owned-observer contract')
class TelemetryWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_fixture(self, mode):
        root = self.root / mode
        root.mkdir()
        script = ('import json; from test_readonly_trace_telemetry import fixture_driver; '
                  f'print(json.dumps(fixture_driver({str(root)!r},{mode!r})))')
        run = subprocess.run([sys.executable, '-c', script], cwd=HERE, env=ENV,
                             text=True, capture_output=True, timeout=10, check=True)
        return json.loads(run.stdout)

    def test_fixed_readonly_commands_and_sampler_startup_affinity(self):
        cpu, gpu = telemetry.commands(self.root, self.root / 'obs', '/usr/bin/python',
                                      '/usr/bin/taskset', '/usr/bin/nvidia-smi', UUID)
        self.assertEqual(cpu[:3], ['/usr/bin/taskset', '-c', '0,2'])
        self.assertEqual(cpu[-8:], ['--cpu', '2', '--observer-cpu', '0', '--seconds', '900', '--interval', '1'])
        self.assertEqual(gpu[:3], ['/usr/bin/nvidia-smi', '-i', UUID])
        self.assertNotIn('--lock-gpu-clocks', ' '.join(gpu))
        with self.assertRaises(ValueError):
            telemetry.commands(self.root, self.root, 'python', '/taskset', '/smi', UUID)

    def test_prefix_identity_affinity_protocol_and_gpu_validation(self):
        meta = dict(kind='metadata', schema_version=1, source_sha256=SOURCE, target_cpu=2,
                    observer_affinity=[0], seconds=900, interval_seconds=1)
        lines = [json.dumps(meta), json.dumps({'kind': 'sample', 'index': 0})]
        self.assertEqual(telemetry.validate_cpu_prefix(lines, SOURCE), meta)
        for key, bad in (('observer_affinity', [2]), ('source_sha256', 'b' * 64),
                         ('seconds', 899), ('schema_version', True)):
            changed = dict(meta, **{key: bad})
            with self.subTest(key=key), self.assertRaises(ValueError):
                telemetry.validate_cpu_prefix([json.dumps(changed), lines[1]], SOURCE)
        with self.assertRaises(ValueError):
            telemetry.validate_gpu_prefix([', '.join(telemetry.GPU_HEADER), 'bad, wrong'], UUID)

    def test_complete_fake_lifecycle_remains_unqualified(self):
        result = self.run_fixture('complete')
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(result['ready'])
        self.assertTrue(result['gpu_shutdown']['verified'])
        for field in ('telemetry_contents_qualified', 'sequence_coverage_verified', 'launch_provenance_verified'):
            self.assertFalse(result[field])

    def test_early_failure_source_startup_and_overflow_stop_without_ready(self):
        for mode in ('gpu_failure', 'bad_source', 'startup', 'overflow', 'delayed_readiness'):
            with self.subTest(mode=mode):
                result = self.run_fixture(mode)
                self.assertEqual(result['status'], 'failed')
                self.assertFalse(result['ready'])
                self.assertFalse((self.root / mode / 'observers/ready.json').exists())

    def test_missing_completion_is_not_promoted(self):
        result = self.run_fixture('incomplete')
        self.assertEqual(result['status'], 'failed')
        self.assertIn('CPU_completion_error:ValueError', result['errors'])

    def test_interrupt_stops_and_reaps_both_observers(self):
        script = ('import json; from test_readonly_trace_telemetry import fixture_driver; '
                  f'print(json.dumps(fixture_driver({str(self.root)!r},"interrupt")))')
        worker = subprocess.Popen([sys.executable, '-c', script], cwd=HERE, env=ENV,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            end = time.monotonic() + 5
            while not (self.root / 'observers/ready.json').exists():
                self.assertIsNone(worker.poll())
                self.assertLess(time.monotonic(), end)
                time.sleep(.01)
            worker.send_signal(signal.SIGTERM)
            stdout, stderr = worker.communicate(timeout=5)
            self.assertEqual(worker.returncode, 0, stderr)
            result = json.loads(stdout)
            self.assertEqual(result['status'], 'failed')
            self.assertTrue(result['gpu_shutdown']['verified'])
            self.assertIn('interrupted', result['cpu_receipt']['issues'])
        finally:
            if worker.poll() is None:
                worker.terminate()
            worker.communicate(timeout=5)

    def test_gpu_cleanup_ownership_and_timeout_escalation(self):
        child = SimpleNamespace(pid=123, poll=Mock(), wait=Mock())
        with patch.object(telemetry.os, 'waitid', side_effect=ChildProcessError), \
                patch.object(telemetry.os, 'killpg') as kill:
            result = telemetry.stop_owned_gpu(child)
        kill.assert_not_called()
        self.assertFalse(result['verified'])
        child.wait.side_effect = [subprocess.TimeoutExpired('fake', 2), -signal.SIGKILL]
        with patch.object(telemetry.os, 'waitid', return_value=None), \
                patch.object(telemetry.os, 'killpg') as kill:
            result = telemetry.stop_owned_gpu(child)
        self.assertEqual([call.args for call in kill.call_args_list], [(123, signal.SIGTERM), (123, signal.SIGKILL)])
        self.assertTrue(result['kill_escalation'])

    def test_observer_failure_cancels_inflight_sequence_without_replacement(self):
        script = ('import json; from test_readonly_trace_telemetry import fixture_driver; '
                  f'r=fixture_driver({str(self.root)!r},"gpu_late_failure"); print(json.dumps(r)); '
                  'raise SystemExit(0 if r["status"]=="complete" else 1)')
        observer = subprocess.Popen([sys.executable, '-c', script], cwd=HERE, env=ENV,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            end = time.monotonic() + 5
            ready = self.root / 'observers/ready.json'
            while not ready.exists():
                self.assertIsNone(observer.poll())
                self.assertLess(time.monotonic(), end)
                time.sleep(.01)
            code = ('import time; from pathlib import Path; '
                    f'Path({str(self.root / "sequence_started")!r}).write_text("ready"); time.sleep(10)')
            jobs = [dict(condition=c, argv=[sys.executable, '-c', code],
                         cwd=str(self.root), env=dict(ENV)) for c in ORDER]
            with patch('readonly_trace_sequence.shutil.disk_usage', return_value=SimpleNamespace(free=100 * 1024**3)):
                result = run_sequence(jobs, self.root / 'sequence', cancel_check=lambda: observer.poll() is not None)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(len(result['jobs']), 1)
            self.assertIn('external_cancelled', result['jobs'][0]['receipt']['issues'])
            self.assertFalse((self.root / 'sequence/02-CP.capture').exists())
            stdout, stderr = observer.communicate(timeout=5)
            self.assertEqual(observer.returncode, 1, stderr)
            self.assertEqual(json.loads(stdout)['status'], 'failed')
        finally:
            if observer.poll() is None:
                observer.terminate()
            observer.communicate(timeout=5)


if __name__ == '__main__':
    unittest.main()
