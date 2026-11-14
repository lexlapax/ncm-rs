"""Credential checks are sampled, fail-closed, and do not change identity."""
import os
import sys
import threading
import unittest
from unittest.mock import patch

import clock_unprivileged_identity as identity


def status(uid='1000', gid='1000', groups='1000 1001', caps='0'):
    return (f'Name:\ttest\nUid:\t{uid} {uid} {uid} {uid}\n'
            f'Gid:\t{gid} {gid} {gid} {gid}\nGroups:\t{groups}\n'
            f'CapInh:\t0\nCapPrm:\t0\nCapEff:\t{caps}\nCapAmb:\t0\n').encode()


@unittest.skipUnless(sys.platform == 'linux', 'Linux proc credentials required')
class CredentialTests(unittest.TestCase):
    def test_parse_complete_credentials(self):
        self.assertEqual(identity._credentials(status()), ((1000,) * 4, (1000,) * 4, (1000, 1001), (0,) * 4))

    def test_reject_missing_duplicate_malformed_and_oversize_fields(self):
        examples = [status().replace(b'CapAmb:\t0\n', b''), status() + b'Uid:\t1000 1000 1000 1000\n',
                    status().replace(b'CapPrm:\t0', b'CapPrm:\tz'),
                    status(uid='-1'), b'x' * (identity.STATUS_LIMIT + 1)]
        for raw in examples:
            with self.subTest(raw=raw[:100]), self.assertRaises(ValueError): identity._credentials(raw)

    def test_reject_invalid_supplied_identity_before_read(self):
        for uid, gid, groups in ((0, 1000, []), (True, 1000, []), (1000, 0, []),
                                 (1000, 1000, [False])):
            with self.subTest(uid=uid, gid=gid, groups=groups), self.assertRaises(ValueError):
                identity.verify(uid, gid, groups)

    def test_duplicate_groups_are_preserved(self):
        parsed = identity._credentials(status(groups='1000 1000 1001'))
        self.assertEqual(parsed[2], (1000, 1000, 1001))

    def test_reject_active_capability_and_all_id_mismatches(self):
        from io import BytesIO
        def check(raw):
            with patch('builtins.open', return_value=BytesIO(raw)), \
                 patch.object(identity.os, 'getresuid', return_value=(1000,) * 3), \
                 patch.object(identity.os, 'getresgid', return_value=(1000,) * 3), \
                 patch.object(identity.os, 'getgroups', return_value=[1000, 1001]):
                return identity.verify(1000, 1000, [1001, 1000])
        self.assertTrue(check(status())['sampled_unprivileged_identity_verified'])
        for raw in (status(caps='1'), status().replace(b'1000 1000 1000 1000', b'1000 1000 1000 0', 1),
                    status(gid='1002'), status(groups='1000'), status().replace(b'CapAmb:\t0', b'CapAmb:\t1'),
                    status().replace(b'CapInh:\t0', b'CapInh:\t1'),
                    status().replace(b'CapPrm:\t0', b'CapPrm:\t1')):
            with self.subTest(raw=raw), self.assertRaises(ValueError): check(raw)

    def test_reject_process_with_another_thread(self):
        ready, done = threading.Event(), threading.Event()
        def wait():
            ready.set()
            done.wait(5)
        thread = threading.Thread(target=wait)
        thread.start()
        try:
            self.assertTrue(ready.wait(2))
            with self.assertRaises(ValueError): identity.verify(1000, 1000, [])
        finally:
            done.set()
            thread.join()

    @unittest.skipUnless(sys.platform == 'linux' and hasattr(os, 'getuid') and os.getuid() > 0 and os.getgid() > 0,
                         'requires a non-root Linux process')
    def test_actual_process_credentials_without_modification(self):
        before = os.getresuid(), os.getresgid(), os.getgroups()
        result = identity.verify(os.getuid(), os.getgid(), os.getgroups())
        self.assertTrue(result['sampled_unprivileged_identity_verified'])
        self.assertFalse(result['launch_ready'])
        self.assertEqual(before, (os.getresuid(), os.getresgid(), os.getgroups()))


if __name__ == '__main__': unittest.main()
