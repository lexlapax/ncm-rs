"""Reject widened exceptions or missing required closure evidence."""
import hashlib,importlib.util,json,unittest
from pathlib import Path
from unittest.mock import patch
p=Path(__file__).with_name('phase2-final-audit-v1.py');spec=importlib.util.spec_from_file_location('final_audit',p);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
class FinalAuditTests(unittest.TestCase):
 def test_rejects_changed_recorded_exception(self):
  original=Path.read_bytes
  def read(path):
   raw=original(path)
   return raw+b' changed' if path==audit.BASE/'apple-immediate-memory-exception-v1.md' else raw
  with patch.object(Path,'read_bytes',read),self.assertRaisesRegex(ValueError,'recorded exception differs'):audit.main()
 def test_rejects_changed_locked_contract(self):
  original=Path.read_bytes
  def read(path):
   raw=original(path)
   return raw+b' ' if path==audit.BASE/'acceptance-v1.json' else raw
  with patch.object(Path,'read_bytes',read),self.assertRaisesRegex(ValueError,'locked contract changed'):audit.main()
 def test_rejects_new_memory_violation_even_with_rehashed_receipt(self):
  original=audit.member;target='reports/phase2-mlx-live-memory-v1/candidate-1.json';metadata='reports/phase2-mlx-live-memory-v1/metadata.json'
  def alter_report(raw):
   x=json.loads(raw);row=next(z for z in x['rows'] if z['batch']==1 and z['scope']=='encode');row['immediate']['active']=100000000;return json.dumps(x).encode()
  def member(t,name):
   raw=original(t,name)
   if name==target:return alter_report(raw)
   if name==metadata:
    x=json.loads(raw);changed=alter_report(original(t,target));entry=next(z for z in x['commands'] if z['role']=='candidate' and z['repeat']==1);entry['report_sha256']=hashlib.sha256(changed).hexdigest();return json.dumps(x).encode()
   return raw
  with patch.object(audit,'member',side_effect=member),self.assertRaisesRegex(ValueError,'exception scope differs'):audit.main()
if __name__=='__main__':unittest.main()
