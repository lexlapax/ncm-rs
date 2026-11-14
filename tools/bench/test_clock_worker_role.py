"""Worker-role composition with stubbed handoff/sequence; no workloads or GPU."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import clock_worker_role as worker


class WorkerRoleTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve(); self.launch=self.root/'launch'; self.launch.mkdir(mode=0o700)
        self.collection=self.root/'collection'
        self.env={'PATH':os.defpath,'HOME':str(self.root),'LANG':'C','LC_ALL':'C',
                  'PYTHONNOUSERSITE':'1','TOKENIZERS_PARALLELISM':'false'}
        self.injection={'path':str(self.root/'.reference/phase2-cupti/libcupti_trace_injection.so')}
        self.contract={'taskset':{'path':'/usr/bin/taskset'},'workloads':{},'libraries':{},'known_libraries':[]}
        self.snapshot={'schema_version':1,'kind':'clock-file-snapshot','project_root':str(self.root),
            'collection_root':str(self.collection),'environment':self.env,
            'tool_identities':[{'path':str(self.root/'tools/bench/clock_diagnostic_entry.py')}],
            'utilities':[{'path':'/usr/bin/python3.14'},{'path':'/usr/bin/nvidia-smi'},{'path':'/usr/bin/git'}],
            'inventory':{'library_contract':self.contract},'native':{'file_identities':[self.injection]},'jobs':[]}
        for key in ('launch_ready','collection_started','gpu_identity_verified','hardware_policy_verified',
                    'permissions_verified','restoration_verified','availability_confirmed'): self.snapshot[key]=False
        self.write_snapshot()
        self.gpu='GPU-11111111-1111-1111-1111-111111111111'; self.coord=['/usr/bin/python3.14','-S','/fixture/root.py']
        self.factory=object()
        self.patches=[patch.object(worker,'os',SimpleNamespace(getppid=lambda:200, getresuid=lambda:(1000,)*3, getresgid=lambda:(1000,)*3, getgroups=lambda:[1000])),
            patch.object(worker,'inspect_binding',return_value={'role':'worker','sampled_role_identity_verified':True,'launch_ready':False}),
            patch.object(worker,'validate_snapshot',side_effect=lambda x:x),
            patch.object(worker,'validate_handoff',return_value=1234.5),
            patch.object(worker,'drop_and_verify'),
            patch.object(worker,'verify_unprivileged',return_value={
                'sampled_unprivileged_identity_verified':True,'launch_ready':False}),
            patch.object(worker,'process_factory',return_value=self.factory),
            patch.object(worker,'run_sequence',return_value={'status':'complete'}),
            patch('subprocess.Popen',side_effect=AssertionError('no workload in wrapper tests'))]
        for p in self.patches: p.start(); self.addCleanup(p.stop)
    def write_snapshot(self):
        raw=json.dumps(self.snapshot).encode(); self.sha=hashlib.sha256(raw).hexdigest()
        path=self.launch/'snapshot.json'; path.write_bytes(raw); path.chmod(0o600)
    def run_role(self,quiet=True,privilege_target=None):
        return worker.run(launch=self.launch,snapshot_sha256=self.sha,coordinator_pid=123,
                          coordinator_argv=self.coord,gpu_uuid=self.gpu,quiet_window_confirmed=quiet,
                          privilege_target=privilege_target)
    def test_handoff_deadline_and_observer_are_forwarded_without_restarting(self):
        self.assertEqual(self.run_role()['status'],'complete')
        worker.verify_unprivileged.assert_called_once_with(1000,1000,(1000,))
        worker.drop_and_verify.assert_not_called()
        handoff=worker.validate_handoff.call_args
        self.assertEqual(handoff.kwargs['parent_pid'],200)
        self.assertEqual(handoff.args[0],self.collection/'restoration/obligation.json')
        self.assertEqual(set(handoff.kwargs['commands']),{'set','worker','reset'})
        sequence=worker.run_sequence.call_args
        self.assertEqual(sequence.kwargs['work_deadline'],1234.5)
        self.assertIs(sequence.kwargs['process_observer_factory'],self.factory)
        self.assertFalse(sequence.kwargs['cancel_check']())
        self.assertEqual(sequence.args[2],self.collection/'sequence')
        self.assertFalse(self.collection.exists())
    def test_quiet_binding_handoff_and_injection_fail_before_work(self):
        with self.assertRaises(ValueError): self.run_role(quiet=False)
        worker.inspect_binding.assert_not_called(); worker.run_sequence.assert_not_called()
        worker.inspect_binding.return_value={'role':'worker','sampled_role_identity_verified':False,'launch_ready':False}
        with self.assertRaises(ValueError): self.run_role()
        worker.run_sequence.assert_not_called()
        worker.inspect_binding.return_value={'role':'worker','sampled_role_identity_verified':True,'launch_ready':False}
        worker.validate_handoff.side_effect=ValueError('handoff failed')
        with self.assertRaisesRegex(ValueError,'handoff failed'): self.run_role()
        worker.process_factory.assert_not_called(); worker.run_sequence.assert_not_called()
        worker.validate_handoff.side_effect=None; self.snapshot['native']['file_identities']=[]; self.write_snapshot()
        with self.assertRaisesRegex(ValueError,'injection'): self.run_role()
        worker.process_factory.assert_not_called(); worker.run_sequence.assert_not_called()
    def test_root_requires_target_and_drops_after_handoff_before_work(self):
        worker.os.getresuid=lambda:(0,)*3
        worker.os.getresgid=lambda:(0,)*3
        with self.assertRaisesRegex(ValueError, 'explicit privilege target'): self.run_role()
        worker.inspect_binding.assert_not_called()
        target={'uid':1000,'gid':1000,'groups':[1000,1001]}
        def drop(uid,gid,groups):
            worker.validate_handoff.assert_called_once()
            worker.process_factory.assert_not_called()
            worker.run_sequence.assert_not_called()
            self.assertEqual((uid,gid,groups), (1000,1000,(1000,1001)))
            worker.os.getresuid=lambda:(1000,)*3
            worker.os.getresgid=lambda:(1000,)*3
            worker.os.getgroups=lambda:[1001,1000]
            return {'sampled_unprivileged_identity_verified':True, 'no_new_privs_verified':True,
                    'launch_ready':False}
        worker.drop_and_verify.side_effect=drop
        self.assertEqual(self.run_role(privilege_target=target)['status'],'complete')
        worker.drop_and_verify.assert_called_once()
        worker.verify_unprivileged.assert_called_once_with(1000,1000,(1000,1001))
        worker.run_sequence.assert_called_once()

    def test_drop_failure_and_mixed_credentials_never_start_work(self):
        worker.os.getresuid=lambda:(0,)*3
        worker.os.getresgid=lambda:(0,)*3
        target={'uid':1000,'gid':1000,'groups':[]}
        worker.drop_and_verify.side_effect=ValueError('drop failed')
        with self.assertRaisesRegex(ValueError, 'drop failed'): self.run_role(privilege_target=target)
        worker.process_factory.assert_not_called(); worker.run_sequence.assert_not_called()
        worker.os.getresuid=lambda:(1000,1000,0)
        with self.assertRaisesRegex(ValueError, 'mixed or root credentials'): self.run_role()
        worker.process_factory.assert_not_called(); worker.run_sequence.assert_not_called()

    def test_unverified_or_incomplete_drop_never_starts_work(self):
        worker.os.getresuid=lambda:(0,)*3
        worker.os.getresgid=lambda:(0,)*3
        worker.drop_and_verify.return_value={'sampled_unprivileged_identity_verified':True,
            'no_new_privs_verified':True,'launch_ready':False}
        with self.assertRaisesRegex(ValueError, 'drop unverified'):
            self.run_role(privilege_target={'uid':1000,'gid':1000,'groups':[]})
        worker.process_factory.assert_not_called(); worker.run_sequence.assert_not_called()

    def test_nonroot_worker_cannot_request_another_target(self):
        with self.assertRaisesRegex(ValueError, 'explicit privilege target'):
            self.run_role(privilege_target={'uid':1001,'gid':1001,'groups':[]})
        worker.inspect_binding.assert_not_called()
        worker.drop_and_verify.assert_not_called()
        worker.run_sequence.assert_not_called()

    def test_unprivileged_identity_failure_stops_before_work(self):
        worker.verify_unprivileged.side_effect=ValueError('active capability')
        with self.assertRaisesRegex(ValueError, 'active capability'): self.run_role()
        worker.drop_and_verify.assert_not_called()
        worker.process_factory.assert_not_called(); worker.run_sequence.assert_not_called()

    def test_parent_loss_callback_is_not_overridden(self):
        self.run_role(); callback=worker.run_sequence.call_args.kwargs['cancel_check']
        worker.os.getppid=lambda:201
        self.assertTrue(callback())


if __name__ == '__main__': unittest.main()
