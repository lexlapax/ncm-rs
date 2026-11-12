import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from bounded_child import run_bounded

CHILD_ENV = {'PATH': os.defpath, 'PYTHONNOUSERSITE': '1'}


@unittest.skipUnless(sys.platform == 'linux', 'Linux process ownership contract')
class BoundedChildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.capture, self.receipt = self.root / 'capture', self.root / 'receipt.json'

    def run_child(self, code, **changes):
        kwargs = dict(cwd=self.root, env=dict(CHILD_ENV), capture_path=self.capture,
                      receipt_path=self.receipt, timeout_seconds=3, output_limit_bytes=4096)
        kwargs.update(changes)
        result = run_bounded([sys.executable, '-c', code], **kwargs)
        self.assertEqual(json.loads(self.receipt.read_bytes()), result)
        self.assertEqual(result['captured_bytes'], self.capture.stat().st_size)
        self.assertEqual(result['capture_sha256'], hashlib.sha256(self.capture.read_bytes()).hexdigest())
        self.assertFalse(result['workload_output_qualified'])
        self.assertFalse(result['executable_identity_verified'])
        return result

    def test_success_merged_streams_and_signal_restoration(self):
        handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        result = self.run_child("import os; os.write(1,b'hello'); os.write(2,b'error')", output_limit_bytes=10)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['issues'], [])
        self.assertEqual(self.capture.read_bytes(), b'helloerror')
        self.assertTrue(result['capture_eof'] and result['direct_child_reaped'])
        self.assertEqual(result['returncode'], 0)
        self.assertEqual({s: signal.getsignal(s) for s in handlers}, handlers)

    def test_nonzero_exit_is_not_complete(self):
        result = self.run_child("import sys; print('failed'); sys.exit(7)")
        self.assertEqual(result['status'], 'failed')
        self.assertIn('nonzero_exit', result['issues'])
        self.assertEqual(result['returncode'], 7)
        self.assertTrue(result['direct_child_reaped'])

    def test_output_limit_is_enforced_even_after_leader_exit(self):
        result = self.run_child("import os; os.write(1,b'x'*100000)", output_limit_bytes=1000)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('output_limit', result['issues'])
        self.assertEqual(result['captured_bytes'], 1000)
        self.assertGreater(result['observed_bytes'], 1000)
        self.assertTrue(result['direct_child_reaped'])

    def test_timeout_and_sequence_deadline(self):
        for which in ('child', 'sequence'):
            with self.subTest(which=which):
                changes = {'timeout_seconds': 0.15} if which == 'child' else {
                    'sequence_deadline': time.monotonic() + 0.15}
                changes.update(capture_path=self.root / which, receipt_path=self.root / (which + '.json'))
                result = run_bounded([sys.executable, '-c', 'import time; time.sleep(10)'],
                                     cwd=self.root, env=CHILD_ENV, output_limit_bytes=4096,
                                     **({'timeout_seconds': 3} | changes))
                self.assertEqual(result['status'], 'failed')
                self.assertIn('child_timeout' if which == 'child' else 'sequence_deadline', result['issues'])
                self.assertEqual(result['returncode'], -signal.SIGKILL)
                self.assertTrue(result['direct_child_reaped'])

    def test_group_signal_precedes_reap_and_no_poll_is_used(self):
        actual = os.killpg
        checked = []

        def kill_owned(pid, sig):
            # WNOWAIT must still succeed: the numeric PID is not available for reuse.
            state = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            self.assertIsNotNone(state)
            checked.append((pid, sig))
            actual(pid, sig)

        with patch('bounded_child.os.killpg', side_effect=kill_owned), patch.object(
                subprocess.Popen, 'poll', side_effect=AssertionError('poll would reap early')):
            result = self.run_child('pass')
        self.assertEqual(checked, [(result['pid'], signal.SIGKILL)])
        with self.assertRaises(ChildProcessError):
            os.waitpid(result['pid'], os.WNOHANG)

    def test_descendant_pipe_holder_stops_and_unrelated_test_child_survives(self):
        other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'],
                                 start_new_session=True, env=CHILD_ENV)
        try:
            result = self.run_child("import subprocess,sys; p=subprocess.Popen([sys.executable,'-c',"
                                    "'import time; time.sleep(10)']); print(p.pid,flush=True)")
            self.assertEqual(result['status'], 'complete')
            self.assertIsNone(other.poll())
            descendant = int(self.capture.read_text())
            stat = Path(f'/proc/{descendant}/stat')
            end = time.monotonic() + 2
            while stat.exists() and time.monotonic() < end:
                try:
                    if stat.read_text().rsplit(') ', 1)[1].split()[0] == 'Z':
                        break  # Grandchild is not ours to wait/reap.
                except FileNotFoundError:
                    break
                time.sleep(0.01)
            else:
                self.assertFalse(stat.exists(), 'owned descendant is still running')
        finally:
            other.terminate()
            other.wait(timeout=3)

    def test_launch_error_and_cleanup_failure_are_retained(self):
        result = run_bounded(['/nonexistent/ncm-fake-command'], cwd=self.root, env=CHILD_ENV,
                             capture_path=self.capture, receipt_path=self.receipt,
                             timeout_seconds=1, output_limit_bytes=10)
        self.assertEqual(result['status'], 'failed')
        self.assertIsNone(result['pid'])
        self.assertIn('capture_error:FileNotFoundError', result['issues'])
        self.capture.unlink(); self.receipt.unlink()
        with patch('bounded_child.os.killpg', side_effect=PermissionError('fake cleanup failure')):
            result = self.run_child('pass')  # Already exited; mock does not leave a live child.
        self.assertEqual(result['status'], 'failed')
        self.assertIn('group_cleanup_failed', result['issues'])
        self.assertTrue(result['direct_child_reaped'])

    def test_capture_write_failure_stops_owned_child(self):
        original = Path.open

        class BrokenCapture:
            def __init__(self, file):
                self.file = file

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.file.close()

            def write(self, _data):
                raise OSError('fake capture write failure')

        def open_file(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get('mode', 'r')
            file = original(path, *args, **kwargs)
            return BrokenCapture(file) if path == self.capture and mode == 'xb' else file

        with patch.object(Path, 'open', autospec=True, side_effect=open_file):
            result = self.run_child("import time; print('data',flush=True); time.sleep(10)")
        self.assertEqual(result['status'], 'failed')
        self.assertIn('capture_error:OSError', result['issues'])
        self.assertTrue(result['direct_child_reaped'])

    def test_late_receipt_interrupt_is_not_swallowed(self):
        previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        actual_kill = os.killpg
        pids = []

        def kill_owned(pid, sig):
            pids.append(pid)
            actual_kill(pid, sig)

        def interrupted_write(*_args, **_kwargs):
            self.assertEqual({s: signal.getsignal(s) for s in previous}, previous)
            raise KeyboardInterrupt

        with patch('bounded_child.json.dump', side_effect=interrupted_write), patch(
                'bounded_child.os.killpg', side_effect=kill_owned):
            with self.assertRaises(KeyboardInterrupt):
                self.run_child('pass')
        self.assertEqual(len(pids), 1)
        with self.assertRaises(ChildProcessError):
            os.waitpid(pids[0], os.WNOHANG)
        self.assertEqual(self.receipt.read_bytes(), b'')

    def test_unconfirmed_reaping_ownership_never_signals_numeric_group(self):
        actual = os.waitid
        exited_seen = False

        def ownership_lost(*args):
            nonlocal exited_seen
            if exited_seen:
                raise ChildProcessError('simulate another reaper')
            result = actual(*args)
            exited_seen = result is not None
            return result

        with patch('bounded_child.os.waitid', side_effect=ownership_lost), patch('bounded_child.os.killpg') as kill:
            result = self.run_child('pass')
            kill.assert_not_called()
        self.assertIn('child_ownership_unconfirmed', result['issues'])
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['direct_child_reaped'])
        try:
            os.waitpid(result['pid'], 0)  # Reap only this exited fixture if Popen did not.
        except ChildProcessError:
            pass

    def test_invalid_limits_and_paths_never_launch(self):
        base = dict(cwd=self.root, env={}, capture_path=self.capture, receipt_path=self.receipt,
                    timeout_seconds=1, output_limit_bytes=10)
        for changed in ({'timeout_seconds': 0}, {'timeout_seconds': float('nan')},
                        {'output_limit_bytes': True}, {'output_limit_bytes': 0},
                        {'sequence_deadline': time.monotonic() - 1},
                        {'receipt_path': self.capture}, {'env': None},
                        {'termination_grace_seconds': -1}, {'termination_grace_seconds': 11},
                        {'termination_grace_seconds': True}, {'termination_grace_seconds': float('nan')}):
            with self.subTest(changed=changed), patch('bounded_child.subprocess.Popen') as launch:
                with self.assertRaises(ValueError):
                    run_bounded(['fake'], **(base | changed))
                launch.assert_not_called()
        with patch('bounded_child.signal.getsignal', return_value=signal.SIG_IGN):
            with self.assertRaisesRegex(ValueError, 'SIGCHLD'):
                run_bounded(['fake'], **base)
        self.assertFalse(self.capture.exists())
        self.capture.write_text('preserve')
        with patch('bounded_child.subprocess.Popen') as launch:
            with self.assertRaises(FileExistsError):
                run_bounded(['fake'], **base)
            launch.assert_not_called()
        self.assertEqual(self.capture.read_text(), 'preserve')

    def test_interrupt_during_setup_prevents_child_creation(self):
        actual = signal.signal
        previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}

        def install_then_interrupt(sig, handler):
            old = actual(sig, handler)
            if sig == signal.SIGTERM and getattr(handler, '__name__', None) == 'interrupt':
                handler(sig, None)
            return old

        with patch('bounded_child.signal.signal', side_effect=install_then_interrupt), patch(
                'bounded_child.subprocess.Popen', side_effect=AssertionError('must not spawn')) as launch:
            result = self.run_child('pass')
            launch.assert_not_called()
        self.assertEqual(result['issues'], ['interrupted'])
        self.assertEqual(result['status'], 'failed')
        self.assertIsNone(result['pid'])
        self.assertEqual(result['group_kill'], 'not_attempted')
        self.assertEqual({s: signal.getsignal(s) for s in previous}, previous)

    def test_deadline_expiring_during_setup_prevents_child_creation(self):
        actual = Path.open
        with patch('bounded_child.time.monotonic', return_value=100.) as clock:
            def delayed_open(path, *args, **kwargs):
                file = actual(path, *args, **kwargs)
                mode = args[0] if args else kwargs.get('mode', 'r')
                if path == self.receipt and mode == 'x':
                    clock.return_value = 104.  # Setup consumed the three-second allowance.
                return file

            with patch.object(Path, 'open', autospec=True, side_effect=delayed_open), patch(
                    'bounded_child.subprocess.Popen', side_effect=AssertionError('must not spawn')) as launch:
                result = self.run_child('pass')
                launch.assert_not_called()
        self.assertEqual(result['issues'], ['child_timeout'])
        self.assertEqual(result['status'], 'failed')
        self.assertIsNone(result['pid'])
        self.assertEqual(result['group_kill'], 'not_attempted')

    def test_sigint_and_sigterm_in_isolated_controller(self):
        module_root = str(Path(__file__).resolve().parent)
        for sig in (signal.SIGINT, signal.SIGTERM):
            root = self.root / str(sig); root.mkdir()
            code = f'''
import json,os,signal,sys
from pathlib import Path
sys.path.insert(0,{module_root!r})
from bounded_child import run_bounded
root=Path({str(root)!r})
def interrupt_when_ready(_sig,_frame):
    p=root/'capture'
    if p.exists() and b'ready' in p.read_bytes():
        signal.setitimer(signal.ITIMER_REAL,0)
        os.kill(os.getpid(),{int(sig)})
signal.signal(signal.SIGALRM,interrupt_when_ready)
signal.setitimer(signal.ITIMER_REAL,0.02,0.02)
try:
    result=run_bounded([sys.executable,'-c',"import time; print('ready',flush=True); time.sleep(2)"],
        cwd=root,env={CHILD_ENV!r},capture_path=root/'capture',receipt_path=root/'receipt',
        timeout_seconds=4,output_limit_bytes=4096)
finally:
    signal.setitimer(signal.ITIMER_REAL,0)
print(json.dumps(result))
'''
            child = subprocess.run([sys.executable, '-c', code], env=CHILD_ENV,
                                   capture_output=True, text=True, timeout=8)
            with self.subTest(signal=sig):
                self.assertEqual(child.returncode, 0, child.stderr)
                result = json.loads(child.stdout)
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['interrupt_signal'], sig)
                self.assertIn('interrupted', result['issues'])
                self.assertTrue(result['direct_child_reaped'])


if __name__ == '__main__':
    unittest.main()
