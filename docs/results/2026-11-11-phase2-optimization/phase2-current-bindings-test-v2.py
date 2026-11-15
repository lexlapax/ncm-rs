"""Negative cases for the current source/evidence binding."""
import importlib.util,json,unittest
from pathlib import Path
from unittest.mock import patch

p=Path(__file__).with_name('phase2-current-bindings-v2.py')
spec=importlib.util.spec_from_file_location('binding_v2',p);binding=importlib.util.module_from_spec(spec);spec.loader.exec_module(binding)

class BindingTests(unittest.TestCase):
 def alter_member(self,target,mutate):
  real=binding.member
  def wrapper(stream,name):
   raw=real(stream,name)
   if name==target:
    value=json.loads(raw);mutate(value);return json.dumps(value).encode()
   return raw
  return patch.object(binding,'member',side_effect=wrapper)
 def test_rejects_untracked_guarded_source(self):
  real=binding.subprocess.check_output
  def call(args,**kwargs):
   if args[:3]==['git','ls-files','--others']:return 'crates/example.rs\n'
   return real(args,**kwargs)
  with patch.object(binding.subprocess,'check_output',side_effect=call),self.assertRaisesRegex(ValueError,'untracked guarded'):
   binding.main()
 def test_rejects_oversized_live_pool(self):
  target='reports/phase2-cuda-bounded-linux-control-v1/candidate/codec-rust-1.json'
  def mutate(v):v['measurements'][0]['pool_bytes']['used_current']*=2
  with self.alter_member(target,mutate),self.assertRaisesRegex(ValueError,'pool ratio'):
   binding.main()
 def test_rejects_apple_executable_mismatch(self):
  target='ncm-bounded-mlx-evidence-v1/verification.json'
  def mutate(v):v['binary_sha256']['mlx_codec_bench']='0'*64
  with self.alter_member(target,mutate),self.assertRaisesRegex(ValueError,'benchmark binary'):
   binding.main()
 def test_rejects_apple_library_mismatch(self):
  target='reports/phase2-mlx-gate18-control-v1/environment.json'
  def mutate(v):v['native_libraries_sha256']['libmlx.dylib']='0'*64
  with self.alter_member(target,mutate),self.assertRaisesRegex(ValueError,'native libraries'):
   binding.main()
if __name__=='__main__':unittest.main()
