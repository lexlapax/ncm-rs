"""Reject tampering in Metal capture identity and qualification receipts."""
import importlib.util,json,unittest
from pathlib import Path
from unittest.mock import patch
p=Path(__file__).with_name('mlx-bound-trace-review-v2.py');spec=importlib.util.spec_from_file_location('metal_trace_review',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class ReceiptTests(unittest.TestCase):
 def altered(self,suffix,mutate):
  real=mod.member
  def call(t,name):
   raw=real(t,name)
   if name.endswith(suffix):return mutate(raw)
   return raw
  return patch.object(mod,'member',side_effect=call)
 def test_rejects_changed_native_library_identity(self):
  def modify(raw):
   obj=json.loads(raw);obj['identities_after']['mlx']='0'*64;return json.dumps(obj).encode()
  with self.altered('/metadata.json',modify),self.assertRaisesRegex(ValueError,'capture identity'):mod.main()
 def test_rejects_modified_qualification(self):
  with self.altered('/qualification.json',lambda raw:raw+b' '),self.assertRaisesRegex(ValueError,'capture output'):mod.main()
if __name__=='__main__':unittest.main()
