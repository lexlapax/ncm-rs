"""Reject tampering in CUDA trace and dependency receipts."""
import importlib.util,json,unittest
from pathlib import Path
from unittest.mock import patch

p=Path(__file__).with_name('cuda-bound-trace-review-v2.py');spec=importlib.util.spec_from_file_location('trace_review',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class ReceiptTests(unittest.TestCase):
 def altered(self,suffix,mutate):
  real=mod.member
  def call(t,name):
   raw=real(t,name)
   if name.endswith(suffix):return mutate(raw)
   return raw
  return patch.object(mod,'member',side_effect=call)
 def test_rejects_changed_capture_identity(self):
  def modify(raw):
   obj=json.loads(raw);obj['identities_after']['binary']='0'*64;return json.dumps(obj).encode()
  with self.altered('/metadata.json',modify),self.assertRaisesRegex(ValueError,'identity changed'):mod.main()
 def test_rejects_modified_dependency_receipt(self):
  with self.altered('/codec_bench.d',lambda raw:raw+b' '),self.assertRaisesRegex(ValueError,'depfile differs'):mod.main()
if __name__=='__main__':unittest.main()
