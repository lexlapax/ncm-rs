"""Review Metal trace identity and qualification receipts, not execution provenance."""
import hashlib,importlib.util,json,subprocess,tarfile
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
 with archive('mlx-bound-trace-capture-v2.tar.xz','c17eab4f29dbb5fda0dd74bd1764a96bdfbd5528288321e176e5c242c0f7ec94') as t,archive('apple-gate18-resource-memory-v1.tar.xz','524293f724bb1c047af88d481db476056f365b0a01aa3af50ddd5bd3a27348b6') as gate:
  base='ncm-mlx-bound-trace-evidence-local-v2/';meta=json.loads(member(t,base+'ncm-mlx-bound-trace-evidence-v2/metadata.json'));qraw=member(t,base+'qualification.json');q=json.loads(qraw);report=member(t,base+'capture-report.json');local=json.loads(member(t,base+'local-qualification.json'))
  if meta['status']!='complete' or meta['returncode']!=0 or meta['identities_before']!=meta['identities_after']:raise ValueError('capture identity or status differs')
  ids=meta['identities_before']; expected={'binary':'8f67496922c7c10376293b53532fd514a7d2f337e075d653c1c070e69ef31381','codec':'2283893a503fbf18de4a92024971cd4cac010a2032acde1f752c14fe39bdea4f','ops':'43339b8f874de5192850304d7ef96b87171bc3d1ba5d60f60883c2b45d6ab91e','blocks':'efbc829cfb79d45d027f0013ee81fa24be52f74c5384ad254a9deb7fe30ed7b0','core':'d0da4ab23c149973ba9e02643f34c3832461312c76c11444675766921d95132b','manifest':'ad6a99ad7d5ccc069e0fedf26a85f755e06cd76590463e1e04e17363e35d9832','mlx':'d24c7a9b9d55a76bfd3bbcd1d042251a185cbadcb6340c3244a6ffa3dcb7c7e8','metal':'dc59d1cceb1a5c7e578232e6e41e28e2c73c9463ac6dbc3886c3ee17ffc270ed'}
  if ids!=expected:raise ValueError('expected identities differ')
  for key,path in {'codec':'crates/ncm-backend-mlx/src/codec.rs','ops':'crates/ncm-backend-mlx/src/ops.rs','blocks':'crates/ncm-model/src/blocks.rs','core':'crates/ncm-core/src/lib.rs'}.items():
   if sha(subprocess.check_output(['git','show',f'f155794:{path}'],cwd=ROOT))!=ids[key]:raise ValueError('source differs from candidate')
  env=json.loads(member(gate,'reports/phase2-mlx-gate18-control-v1/environment.json'))
  if ids['mlx']!=env['native_libraries_sha256']['libmlx.dylib'] or ids['metal']!=env['native_libraries_sha256']['mlx.metallib']:raise ValueError('native library differs')
  if sha(report)!=meta['candidate_report_sha256'] or q['candidate_report_sha256']!=sha(report) or q['right_trace_sha256']!=meta['trace_sha256'] or sha(qraw)!=local['qualification_sha256'] or local['returncode']!=0:raise ValueError('capture output differs')
  previous.verify_quality(q,'rust-mlx')
  if qraw!=member(gate,'.reference/phase2-mlx-gate18-preflight-v1/qualification.json'):raise ValueError('qualification differs from gated result')
  print(json.dumps({'scope':'archived local and remote receipt consistency; not immutable process provenance or performance acceptance','capture_binary_sha256':ids['binary'],'qualified_traces':642,'decisions':2500,'matches_gated_qualification_bytes':True,'native_libraries_match_control':True,'trace_sha256':meta['trace_sha256'],'phase_2_complete':False},indent=2))
if __name__=='__main__':main()
