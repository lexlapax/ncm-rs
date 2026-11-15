"""Review captured CUDA identity, depfile and quality receipts, not GPU provenance."""
import hashlib,importlib.util,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];BASE=Path(__file__).resolve().parent;RAW=BASE/'raw'
spec=importlib.util.spec_from_file_location('previous_binding',BASE/'phase2-closure-bindings-v1.py');previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
sha=lambda x:hashlib.sha256(x).hexdigest()
def archive(name,digest):
 p=RAW/name
 if sha(p.read_bytes())!=digest:raise ValueError('archive hash differs')
 return tarfile.open(p,'r:xz')
def member(t,name):
 e=t.getmember(name)
 if not e.isfile():raise ValueError('expected regular file')
 return t.extractfile(e).read()
def main():
 with archive('cuda-bound-trace-capture-v2.tar.xz','2c6a6b4f9e6934ed9cb2bffb52d2b73e3041e615dba5cb874cac059f5b91c33c') as t,archive('cuda-bounded-linux-gate-v1.tar.xz','a2c3d9bcb754d7b511b8ce11812b56e3389565277edcbee8c56cc7ce181044d8') as gate:
  base='ncm-cuda-bound-trace-evidence-v2/'
  meta=json.loads(member(t,base+'metadata.json'));qraw=member(t,base+'qualification.json');q=json.loads(qraw);report=member(t,base+'capture-report.json');observation=json.loads(member(t,base+'current-build-observation.json'))
  if meta['status']!='complete' or meta['identities_before']!=meta['identities_after']:raise ValueError('capture identity changed')
  expected={'binary':'2714ef7303eab002c838131d8729c78d2a7a1aee739db9e3f4ef4fd8a62ca181','source':'92686330ae3c661d60ed291b68a069b15ca25a1103c444deb502f19708014950','bounded':'5e5cd0fe6176dd3bd0079a73fb36e5687fa09e49f217564cffc5e56ae0740327','manifest':'ad6a99ad7d5ccc069e0fedf26a85f755e06cd76590463e1e04e17363e35d9832'}
  if meta['identities_before']!=expected or [r['label'] for r in meta['commands']]!=['capture','qualification'] or any(r['returncode']!=0 for r in meta['commands']):raise ValueError('capture receipt differs')
  if sha(qraw)!=meta['qualification_sha256'] or sha(report)!=meta['candidate_report_sha256'] or q['candidate_report_sha256']!=sha(report) or q['right_trace_sha256']!=meta['trace_sha256']:raise ValueError('capture output hash differs')
  previous.verify_quality(q,'rust-candle')
  earlier=member(gate,'reports/phase2-cuda-bounded-qualification-v1.json')
  if qraw!=earlier:raise ValueError('qualification differs from gated result')
  env=json.loads(member(gate,'reports/phase2-cuda-bounded-linux-control-v1/environment.json'));life=json.loads(member(gate,'reports/phase2-cuda-bounded-lifecycle-v1.json'))
  for name in ('codec_bench','codec_lifecycle','codec_trace'):
   data=member(t,base+name+'.d');entry=observation['examples'][name]
   if sha(data)!=entry['depfile_sha256'] or any(source.encode() not in data for source in ('crates/ncm-backend-cuda/src/fused_projection.rs','crates/ncm-backend-cuda/src/fused_projection/bounded.rs')):raise ValueError('depfile differs')
  if observation['examples']['codec_bench']['binary_sha256']!=env['binary_sha256']['candidate']['codec_bench'] or observation['examples']['codec_lifecycle']['binary_sha256']!=life['binary_sha256']['candidate'] or observation['examples']['codec_trace']['binary_sha256']!=expected['binary']:raise ValueError('binary observation differs')
  print(json.dumps({'scope':'archived local receipt consistency; not immutable process provenance or performance acceptance','capture_binary_sha256':expected['binary'],'qualified_traces':642,'decisions':2500,'matches_gated_qualification_bytes':True,'depfile_sources_present':True,'trace_sha256':meta['trace_sha256'],'phase_2_complete':False},indent=2))
if __name__=='__main__':main()
