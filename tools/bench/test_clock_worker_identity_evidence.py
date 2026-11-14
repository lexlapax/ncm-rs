"""Real private identity writer and offline receipt binding, no workloads or GPU."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from bounded_child import run_bounded

import clock_worker_identity_evidence as evidence
import validate_clock_worker_identity as validator
import test_validate_clock_restoration as restoration_fixture


@unittest.skipUnless(sys.platform == 'linux' and os.getuid() > 0 and os.getgid() > 0,
                     'unprivileged Linux process required')
class WriterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.uid, self.gid, self.groups = os.getuid(), os.getgid(), os.getgroups()

    def write(self):
        return evidence.write(self.root, self.uid, self.gid, self.groups, require_no_new_privs=False)

    def test_private_exclusive_real_sample(self):
        record = self.write()
        path = self.root/evidence.NAME
        self.assertEqual(json.loads(path.read_bytes()), record)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(record['lifetime']['pid'], os.getpid())
        self.assertEqual(record['target']['groups'], sorted(self.groups))
        self.assertFalse(record['identity_declaration_authenticated'])
        self.assertFalse(record['continuous_identity_verified'])
        self.assertFalse(record['launch_ready'])
        with self.assertRaises(FileExistsError): self.write()

    def test_nonprivate_directory_and_existing_symlink_rejected(self):
        self.root.chmod(0o755)
        with self.assertRaises(ValueError): self.write()
        self.root.chmod(0o700)
        (self.root/evidence.NAME).symlink_to(self.root/'other')
        with self.assertRaises(FileExistsError): self.write()
        self.assertFalse((self.root/'other').exists())

    def test_restrictive_umask_does_not_publish_a_readable_record(self):
        old = os.umask(0o777)
        try:
            with self.assertRaisesRegex(ValueError, 'owned private worker identity file'):
                self.write()
        finally:
            os.umask(old)
        self.assertEqual((self.root/evidence.NAME).stat().st_mode & 0o777, 0)
        self.assertEqual((self.root/evidence.NAME).stat().st_size, 0)

    def test_wrong_identity_fails_before_publication(self):
        with self.assertRaises(ValueError):
            evidence.write(self.root, self.uid+1, self.gid, self.groups, require_no_new_privs=False)
        self.assertFalse((self.root/evidence.NAME).exists())


    def test_required_no_new_privs_rejects_unset_state(self):
        with patch.object(evidence, 'no_new_privs', return_value=False):
            with self.assertRaisesRegex(ValueError, 'credential sample incomplete'):
                evidence.write(self.root,self.uid,self.gid,self.groups,require_no_new_privs=True)
        self.assertFalse((self.root/evidence.NAME).exists())

    def test_real_child_record_binds_to_owned_receipt_and_sequence(self):
        tools = str(Path(__file__).resolve().parent)
        program = f"""import json,os,sys
sys.path.insert(0,{tools!r})
import clock_worker_identity_evidence as evidence
import diagnostic_clock
root=sys.argv[1]
evidence.write(root,os.getuid(),os.getgid(),os.getgroups(),require_no_new_privs=False)
life=diagnostic_clock.begin()
print(json.dumps({{'schema_version':1,'lifetime':diagnostic_clock.finish(life)}}))
"""
        env={'PATH':os.defpath,'HOME':str(self.root),'LANG':'C','LC_ALL':'C'}
        receipt=run_bounded([sys.executable,'-S','-c',program,str(self.root)],cwd=self.root,env=env,
            capture_path=self.root/'child.out',receipt_path=self.root/'child.receipt.json',
            timeout_seconds=5,output_limit_bytes=4096)
        self.assertEqual(receipt['status'],'complete',receipt)
        sequence=json.loads((self.root/'child.out').read_bytes())
        record=json.loads((self.root/evidence.NAME).read_bytes())
        declared={'uid':self.uid,'gid':self.gid,'groups':self.groups,'require_no_new_privs':False}
        value=validator.validate(record,declared,receipt,sequence)
        self.assertTrue(value['worker_identity_record_consistent'])
        self.assertFalse(value['producer_authenticated'])
        self.assertEqual(record['lifetime']['pid'],receipt['pid'])
        self.assertFalse(Path('/proc').joinpath(str(receipt['pid'])).exists())


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        fixture=restoration_fixture.RestorationValidationTests(); fixture.setUp()
        self.worker=deepcopy(fixture.restoration['worker'])
        self.sequence=deepcopy(fixture.sequence)
        self.target={'uid':1000,'gid':1000,'groups':[1001,1000], 'require_no_new_privs':True}
        self.record={'schema_version':1,'kind':'clock-worker-identity',
            'target':{'uid':1000,'gid':1000,'groups':[1000,1001], 'require_no_new_privs':True},
            'no_new_privs':True,'lifetime':restoration_fixture.life(104,105,52,50),
            'sampled_unprivileged_identity_verified':True,'identity_declaration_authenticated':False,
            'continuous_identity_verified':False,'launch_ready':False}

    def test_consistent_record_is_narrow_and_input_preserving(self):
        before=deepcopy((self.record,self.target,self.worker,self.sequence))
        with patch('builtins.open',side_effect=AssertionError('no file IO')), \
             patch('subprocess.Popen',side_effect=AssertionError('no processes')):
            result=validator.validate(self.record,self.target,self.worker,self.sequence)
        self.assertTrue(result['worker_identity_record_consistent'])
        for key in ('producer_authenticated','continuous_identity_verified','identity_declaration_authenticated','launch_ready'):
            self.assertFalse(result[key])
        self.assertEqual(before,(self.record,self.target,self.worker,self.sequence))

    def test_identity_scope_and_clock_tampering_fail(self):
        for mode in ('uid','gid','groups','target_bool','nnp','nnp_type','sample','claim','producer',
                     'schema','pid','parent','after_sequence','outside_receipt','extra'):
            value=deepcopy(self.record)
            if mode=='uid': value['target']['uid']=1001
            if mode=='gid': value['target']['gid']=1001
            if mode=='groups': value['target']['groups']=[1000]
            if mode=='target_bool': value['target']['require_no_new_privs']=1
            if mode=='nnp': value['no_new_privs']=False
            if mode=='nnp_type': value['no_new_privs']=1
            if mode=='sample': value['sampled_unprivileged_identity_verified']=False
            if mode=='claim': value['identity_declaration_authenticated']=True
            if mode=='producer': value['continuous_identity_verified']=True
            if mode=='schema': value['schema_version']=True
            if mode=='pid': value['lifetime']['pid']=53
            if mode=='parent': value['lifetime']['parent_pid']=49
            if mode=='after_sequence': value['lifetime']=restoration_fixture.life(106,107,52,50)
            if mode=='outside_receipt': value['lifetime']=restoration_fixture.life(103,104,52,50)
            if mode=='extra': value['unexpected']=True
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validator.validate(value,self.target,self.worker,self.sequence)

    def test_invalid_expected_identity_and_failed_receipt_fail(self):
        for bad in ({**self.target,'uid':True}, {**self.target,'groups':[True]},
                    {**self.target,'require_no_new_privs':1}, {**self.target,'uid':2**32-1},
                    {**self.target,'groups':[2**32-1]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validator.validate(self.record,bad,self.worker,self.sequence)
        receipt=deepcopy(self.worker); receipt['status']='failed'
        with self.assertRaises(ValueError): validator.validate(self.record,self.target,receipt,self.sequence)


if __name__ == '__main__': unittest.main()
