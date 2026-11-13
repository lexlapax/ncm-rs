"""Read-only GPU metadata contract with stubbed commands; no GPU access."""
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

import prepare_clock_gpu as gpu


class GpuPreparationTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve(); self.tool=self.root/'nvidia-smi'; self.tool.write_bytes(b'fake'); self.tool.chmod(0o700)
        self.output=self.root/'evidence'; self.uuid='GPU-11111111-1111-1111-1111-111111111111'
        self.pci='00000000:01:00.0'; self.name='Fixture GPU'
        self.env={'PATH':os.defpath,'HOME':str(self.root),'LANG':'C','LC_ALL':'C','PYTHONNOUSERSITE':'1'}
        self.identity=f'{self.uuid}, {self.pci}, {self.name}\n'; self.clocks='1800 MHz\n1710 MHz\n900 MHz\n'
        self.calls=[]
    def probe(self,argv,cwd,env,directory,name):
        self.calls.append(argv)
        return (self.identity if name=='identity' else self.clocks), {'fixture':name}
    def inspect(self):
        with patch.object(gpu,'metadata_probe',side_effect=self.probe):
            return gpu.inspect(self.root,self.output,nvidia_smi=self.tool,environment=self.env,
                               expected_uuid=self.uuid,expected_pci=self.pci,expected_name=self.name)
    def test_only_bounded_readonly_queries_and_false_hardware_claims(self):
        result=self.inspect(); self.assertEqual(len(self.calls),2)
        self.assertEqual(self.calls[0][1:],["--query-gpu=uuid,pci.bus_id,name","--format=csv,noheader"])
        self.assertEqual(self.calls[1][1:],['-i',self.uuid,'--query-supported-clocks=graphics','--format=csv,noheader'])
        self.assertTrue(result['requested_graphics_in_application_list'])
        self.assertEqual(result['supported_graphics_row_count'],3)
        for key in ('gpu_lock_support_verified','gpu_lock_permission_verified','clock_enforcement_verified',
                    'hardware_policy_verified','availability_confirmed','launch_ready'): self.assertFalse(result[key])
    def test_identity_clock_and_contamination_fail_closed(self):
        for mode in ('uuid','pci','name','missing','unit','multiple','env'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                self.output=Path(temporary).resolve()/'evidence'; self.calls=[]
                identity, clocks, env=self.identity,self.clocks,dict(self.env)
                if mode=='uuid': self.identity=self.identity.replace(self.uuid,'GPU-22222222-2222-2222-2222-222222222222')
                if mode=='pci': self.identity=self.identity.replace(self.pci,'00000000:02:00.0')
                if mode=='name': self.identity=self.identity.replace(self.name,'Different GPU')
                if mode=='missing': self.clocks='1800 MHz\n'
                if mode=='unit': self.clocks='1710 GHz\n'
                if mode=='multiple': self.identity+=self.identity
                if mode=='env': self.env['LD_PRELOAD']='unreviewed'
                with self.assertRaises(ValueError): self.inspect()
                if mode in ('uuid','pci','name','multiple'): self.assertEqual(len(self.calls),1)
                if mode=='env': self.assertEqual(len(self.calls),0)
                self.identity,self.clocks,self.env=identity,clocks,env
    def test_utility_replacement_during_query_is_rejected(self):
        real=self.probe
        def mutate(argv,cwd,env,directory,name):
            result=real(argv,cwd,env,directory,name)
            if name=='supported-graphics':
                self.tool.rename(self.root/'old-tool'); self.tool.write_bytes(b'fake'); self.tool.chmod(0o700)
            return result
        with patch.object(gpu,'metadata_probe',side_effect=mutate):
            with self.assertRaisesRegex(ValueError,'utility replaced'):
                gpu.inspect(self.root,self.output,nvidia_smi=self.tool,environment=self.env,
                            expected_uuid=self.uuid,expected_pci=self.pci,expected_name=self.name)


if __name__ == '__main__': unittest.main()
