"""Synthetic tests for the dedicated worker drop; no GPU or real privilege change."""
from types import SimpleNamespace
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import clock_drop_privileges as drop


@unittest.skipUnless(sys.platform == 'linux', 'Linux process credentials required')
class DropTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        def record(name):
            def call(*args): self.calls.append((name, args))
            return call
        def fatal(code):
            self.calls.append(('exit', (code,)))
            raise RuntimeError('fatal process exit')
        self.fake_os = SimpleNamespace(getresuid=lambda:(0, 0, 0), getresgid=lambda:(0, 0, 0),
            setgroups=record('groups'), setresgid=record('gid'), setresuid=record('uid'), _exit=fatal)
        self.fake_sys = SimpleNamespace(platform='linux', flags=SimpleNamespace(no_site=1))
        self.patches = [patch.object(drop, 'os', self.fake_os), patch.object(drop, 'sys', self.fake_sys),
                        patch.object(drop, '_no_extra_fds'),
                        patch.object(drop, 'verify', side_effect=lambda *args: self.calls.append(('verify', args)) or {
                            'sampled_unprivileged_identity_verified':True, 'launch_ready':False}),
                        patch.object(drop, '_prctl', side_effect=self.prctl)]
        for item in self.patches: item.start(); self.addCleanup(item.stop)

    def prctl(self, option, value):
        self.calls.append(('prctl', (option, value)))
        return 1 if option == drop.PR_GET_NO_NEW_PRIVS else 0

    def test_drop_order_and_untrusted_readiness(self):
        result = drop.drop_and_verify(1000, 1001, [1002, 1002, 1003])
        self.assertEqual([v[0] for v in self.calls], ['prctl', 'groups', 'gid', 'uid', 'verify', 'prctl'])
        self.assertEqual(self.calls[1][1], ((1002, 1002, 1003),))
        self.assertEqual(self.calls[2][1], (1001, 1001, 1001))
        self.assertEqual(self.calls[3][1], (1000, 1000, 1000))
        self.assertTrue(result['no_new_privs_verified'])
        self.assertFalse(result['privilege_drop_authenticated'])
        self.assertFalse(result['launch_ready'])

    def test_partial_transition_failure_exits_without_continuing(self):
        def fail(*args): raise OSError('simulated failure')
        self.fake_os.setresgid = fail
        with self.assertRaisesRegex(RuntimeError, 'fatal process exit'):
            drop.drop_and_verify(1000, 1000, [])
        self.assertEqual([v[0] for v in self.calls], ['prctl', 'groups', 'exit'])
        self.assertEqual(self.calls[-1][1], (125,))
        drop.verify.assert_not_called()

    def test_failed_verification_and_no_new_privs_are_fatal(self):
        for failing in ('verify', 'prctl'):
            with self.subTest(failing=failing):
                self.calls.clear()
                if failing == 'verify': drop.verify.side_effect = ValueError('bad credentials')
                else:
                    drop.verify.side_effect = lambda *args: {}
                    drop._prctl.side_effect = lambda option, value: 0
                with self.assertRaisesRegex(RuntimeError, 'fatal process exit'):
                    drop.drop_and_verify(1000, 1000, [])
                self.assertEqual(self.calls[-1], ('exit', (125,)))

    def test_invalid_target_and_nonroot_rejected_before_mutation(self):
        for uid, gid, groups in ((0, 1, []), (1, 0, []), (True, 1, []),
                                 (1, 1, [-1]), (1, 1, [2**32-1]), (2**32-1, 1, [])):
            with self.subTest(uid=uid, gid=gid, groups=groups), self.assertRaises(ValueError):
                drop.drop_and_verify(uid, gid, groups)
        self.fake_os.getresuid = lambda:(1000, 1000, 1000)
        with self.assertRaises(ValueError): drop.drop_and_verify(1000, 1000, [])
        self.assertEqual(self.calls, [])
        drop._prctl.assert_not_called()

    def test_unexpected_open_fd_rejected_before_mutation(self):
        drop._no_extra_fds.side_effect = ValueError('unexpected inherited descriptor')
        with self.assertRaisesRegex(ValueError, 'descriptor'): drop.drop_and_verify(1000, 1000, [])
        self.assertEqual(self.calls, [])
        drop._prctl.assert_not_called()


@unittest.skipUnless(sys.platform == 'linux', 'Linux proc descriptors required')
class DescriptorTests(unittest.TestCase):
    def test_real_prctl_in_disposable_unprivileged_child(self):
        source = str(Path(__file__).resolve().parent)
        program = (f"import sys; sys.path.insert(0,{source!r}); "
                   "import clock_drop_privileges as d; "
                   "assert d._prctl(d.PR_SET_NO_NEW_PRIVS,1)==0; "
                   "assert d._prctl(d.PR_GET_NO_NEW_PRIVS,0)==1")
        result = subprocess.run([sys.executable, '-S', '-c', program], capture_output=True,
                                text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_real_open_descriptor_is_rejected(self):
        read_fd, write_fd = os.pipe()
        try:
            with self.assertRaisesRegex(ValueError, 'descriptor'):
                drop._no_extra_fds()
        finally:
            os.close(read_fd)
            os.close(write_fd)


if __name__ == '__main__': unittest.main()
