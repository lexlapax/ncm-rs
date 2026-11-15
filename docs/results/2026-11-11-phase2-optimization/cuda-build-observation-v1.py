"""Observe local CUDA build dependencies and retained trace hashes; no gate claim."""
import hashlib,json,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
RAW=ROOT/'docs/results/2026-11-11-phase2-optimization/raw'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def read(stream,name):
 member=stream.getmember(name)
 if not member.isfile():raise ValueError('non-file archive entry')
 return stream.extractfile(member).read()
def main():
 archive=RAW/'cuda-bounded-linux-gate-v1.tar.xz'
 if digest(archive)!='a2c3d9bcb754d7b511b8ce11812b56e3389565277edcbee8c56cc7ce181044d8':raise ValueError('gate archive differs')
 apple=RAW/'cuda-bounded-mlx-check-v1.tar.xz'
 if digest(apple)!='9c6a9f314fe061932bccda06b5832a2022cb0b7e7672f4c34b808cfdb80eb3c2':raise ValueError('Apple check archive differs')
 with tarfile.open(archive,'r:xz') as t,tarfile.open(apple,'r:xz') as a:
  env=json.loads(read(t,'reports/phase2-cuda-bounded-linux-control-v1/environment.json'))
  life=json.loads(read(t,'reports/phase2-cuda-bounded-lifecycle-v1.json'))
  cuda_q=json.loads(read(t,'reports/phase2-cuda-bounded-qualification-v1.json'))
  mlx_q=json.loads(read(a,'ncm-bounded-mlx-evidence-v1/qualification.json'))
  sources=('crates/ncm-backend-cuda/src/fused_projection.rs','crates/ncm-backend-cuda/src/fused_projection/bounded.rs')
  source_hashes={}
  for source in sources:
   local=digest(ROOT/source);revision=sha(subprocess.check_output(['git','show',f'f155794:{source}'],cwd=ROOT))
   if local!=revision:raise ValueError('current source differs: '+source)
   source_hashes[source]=local
  examples={}
  for name in ('codec_bench','codec_lifecycle','codec_trace'):
   exe=ROOT/f'target/release/examples/{name}';dep=exe.with_suffix('.d');depraw=dep.read_bytes();parts=shlex.split(depraw.decode().replace('\\\n',' '));
   if not parts or not parts[0].endswith(name+':'):raise ValueError('unexpected dependency target')
   paths={str(Path(x).resolve()) for x in parts[1:]}
   if not all(str((ROOT/s).resolve()) in paths for s in sources):raise ValueError('missing CUDA source in depfile: '+name)
   got=digest(exe)
   if name=='codec_bench' and got!=env['binary_sha256']['candidate']['codec_bench']:raise ValueError('benchmark binary differs from gate')
   if name=='codec_lifecycle' and got!=life['binary_sha256']['candidate']:raise ValueError('lifecycle binary differs from gate')
   examples[name]={'binary_sha256':got,'depfile_sha256':sha(depraw),'both_cuda_sources_listed':True}
  traces={}
  for name,path,q in (('cuda','reports/phase2-codec/cuda-workspace-candidate-v1',cuda_q),('metal','reports/phase2-codec/mlx-bounded-workspace-v1',mlx_q)):
   d=ROOT/path; report=d/'report.json'; tensors=d/'traces.safetensors'
   if digest(report)!=q['candidate_report_sha256'] or digest(tensors)!=q['right_trace_sha256']:raise ValueError('qualified local trace differs: '+name)
   traces[name]={'report_sha256':digest(report),'trace_sha256':digest(tensors),'trace_bytes':tensors.stat().st_size}
  oracle=ROOT/'reports/phase2-oracle/v1/sudoku-cuda/traces.safetensors'
  if digest(oracle)!=cuda_q['left_trace_sha256'] or digest(oracle)!=mlx_q['left_trace_sha256']:raise ValueError('oracle trace differs')
  traces['oracle']={'trace_sha256':digest(oracle),'trace_bytes':oracle.stat().st_size}
  print(json.dumps({'scope':'local build and trace hash observation, not execution provenance or an acceptance decision','source_commit':'f155794','cuda_source_sha256':source_hashes,'examples':examples,'qualified_local_traces':traces,'limitations':['Current depfiles are not a cryptographic proof of historical build inputs.','Current codec_trace executable hash was not captured by the prior trace launch; its identity is not asserted retroactively.','Large trace arrays remain local and licensed publication requires separate authorization.'],'phase_2_complete':False},indent=2))
if __name__=='__main__':main()
