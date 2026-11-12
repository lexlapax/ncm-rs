"""Outer shutdown-budget tests: fake policy files and CPU-only processes, no GPU."""
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import clock_restoration_capture as outer

HERE = Path(__file__).resolve().parent
POLICY = '''import os,sys,time
from pathlib import Path
root=Path(sys.argv[1]);stage=sys.argv[2]
with (root/'events').open('a') as file:file.write(stage+'\\n')
if stage=='set':(root/'fake-lock').touch()
else:
    assert not Path('/proc/'+(root/'worker.pid').read_text()).exists()
    time.sleep(6)
    if (root/'events').read_text().splitlines().count('reset')==1:raise SystemExit(9)
    (root/'fake-lock').unlink()
'''


@unittest.skipUnless(sys.platform == 'linux', 'Linux owned-child supervision')
class RestorationCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = {'PATH': os.defpath, 'HOME': str(self.root), 'LANG': 'C', 'LC_ALL': 'C',
                    'PYTHONNOUSERSITE': '1'}
        self.capture, self.receipt = self.root / 'capture', self.root / 'receipt.json'

    def capture_child(self, code, cancel_check):
        return outer.capture_restorer([sys.executable, '-S', '-c', code], cwd=HERE, env=self.env,
                                      capture_path=self.capture, receipt_path=self.receipt,
                                      cancel_check=cancel_check)

    def test_fixed_outer_budget_does_not_delay_normal_exit(self):
        receipt = self.capture_child("print('complete')", lambda: False)
        self.assertEqual(receipt['status'], 'complete')
        self.assertEqual(receipt['timeout_seconds'], 660)
        self.assertEqual(receipt['termination_grace_seconds'], 60)
        self.assertEqual(receipt['termination_signal'], 'not_attempted')
        self.assertTrue(receipt['direct_child_reaped'])
        self.assertFalse(receipt['executable_identity_verified'])

    def test_cancelled_supervisor_finishes_twelve_second_cleanup_retry_and_large_output(self):
        policy = self.root / 'policy.py'
        policy.write_text(POLICY)
        python = str(Path(sys.executable).resolve())
        worker_code = (f"import os,time;from pathlib import Path;root=Path({str(self.root)!r});"
                       "(root/'worker.pid').write_text(str(os.getpid()));(root/'ready').touch();time.sleep(30)")
        commands = [[python, '-S', str(policy), str(self.root), 'set'],
                    [python, '-S', '-c', worker_code],
                    [python, '-S', str(policy), str(self.root), 'reset']]
        supervisor = f'''import json,os,sys
from pathlib import Path
from supervised_restoration import supervise_restoration
from supervised_worker import direct_children,subreaper
result=supervise_restoration(*{commands!r},cwd={str(HERE)!r},env={self.env!r},
    output_dir={str(self.root / 'restoration')!r},parent_pid=os.getppid())
assert direct_children()==[] and subreaper()==0,result
print('receipt-prefix:'+'x'*131072,flush=True)
print(json.dumps(result),flush=True)
raise SystemExit(0 if result['status']=='complete' else 1)
'''
        driver = f'''import json,os
from pathlib import Path
from clock_restoration_capture import capture_restorer
receipt=capture_restorer({[python, '-S', '-c', supervisor]!r},cwd={str(HERE)!r},env={self.env!r},
    capture_path={str(self.capture)!r},receipt_path={str(self.receipt)!r},
    cancel_check=lambda:Path({str(self.root / 'ready')!r}).exists())
print(json.dumps(receipt))
'''
        run = subprocess.run([python, '-S', '-c', driver], cwd=HERE, env=self.env,
                             capture_output=True, text=True, timeout=35)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        receipt = json.loads(run.stdout)
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(receipt['termination_signal'], 'sent')
        self.assertFalse(receipt['termination_grace_exhausted'])
        self.assertEqual(receipt['returncode'], 1)
        self.assertTrue(receipt['direct_child_reaped'] and receipt['capture_eof'])
        self.assertGreaterEqual(receipt['elapsed_seconds_including_cleanup'], 12)
        self.assertGreater(receipt['captured_bytes'], 131072)
        self.assertIn(outer.WARNING, run.stderr)
        result = json.loads(self.capture.read_text().splitlines()[-1])
        self.assertEqual(result['status'], 'failed')  # First reset failure/interrupt stay failed.
        self.assertTrue(result['reset_command_succeeded'])
        self.assertEqual(len(result['reset_attempts']), 2)
        self.assertEqual(result['reset_attempts'][0]['returncode'], 9)
        self.assertFalse(result['hardware_policy_restoration_verified'])
        self.assertFalse((self.root / 'fake-lock').exists())
        self.assertEqual((self.root / 'events').read_text().splitlines(), ['set', 'reset', 'reset'])
        self.assertFalse(Path(f'/proc/{receipt["pid"]}').exists())

    def test_exhausted_grace_is_bounded_failure_and_warns(self):
        ready = self.root / 'ready'
        code = ("import signal,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                f"Path({str(ready)!r}).touch();time.sleep(30)")
        stderr = io.StringIO()
        with patch.object(outer, 'RESTORATION_GRACE_SECONDS', .2), redirect_stderr(stderr):
            receipt = self.capture_child(code, ready.exists)
        self.assertEqual(receipt['status'], 'failed')
        self.assertTrue(receipt['termination_grace_exhausted'])
        self.assertEqual(receipt['returncode'], -signal.SIGKILL)
        self.assertTrue(receipt['direct_child_reaped'])
        self.assertFalse(Path(f'/proc/{receipt["pid"]}').exists())
        self.assertIn(outer.WARNING, stderr.getvalue())

    def test_capture_setup_error_warns_and_missing_monitor_never_starts(self):
        stderr = io.StringIO()
        self.capture.touch()
        with redirect_stderr(stderr), self.assertRaises(FileExistsError):
            self.capture_child("raise AssertionError('must not execute')", lambda: False)
        self.assertIn(outer.WARNING, stderr.getvalue())
        self.capture.unlink()
        with self.assertRaisesRegex(ValueError, 'mandatory cancellation'):
            self.capture_child("raise AssertionError('must not execute')", None)
        self.assertFalse(self.capture.exists())
        with redirect_stderr(stderr):
            receipt = self.capture_child("raise AssertionError('must not execute')", lambda: None)
        self.assertEqual(receipt['status'], 'failed')
        self.assertIsNone(receipt['pid'])
        self.assertIn('capture_error:ValueError', receipt['issues'])


if __name__ == '__main__':
    unittest.main()
