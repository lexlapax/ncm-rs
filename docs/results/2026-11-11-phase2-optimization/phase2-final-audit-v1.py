"""Review Phase 2 evidence with the recorded, narrowly scoped user exception.

This verifies recorded artifacts and arithmetic, not the identity of the person
who approved the exception, immutable execution provenance or future behavior.
"""
import contextlib,hashlib,importlib.util,io,json,math,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];BASE=Path(__file__).resolve().parent;RAW=BASE/'raw'
sha=lambda x:hashlib.sha256(x).hexdigest()
def load(name,file):
 spec=importlib.util.spec_from_file_location(name,BASE/file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def run(module):
 output=io.StringIO()
 with contextlib.redirect_stdout(output):module.main()
 return json.loads(output.getvalue())
def archive(name,digest):
 p=RAW/name
 if sha(p.read_bytes())!=digest:raise ValueError('archive hash differs: '+name)
 return tarfile.open(p,'r:xz')
def member(t,name):
 e=t.getmember(name)
 if not e.isfile():raise ValueError('expected regular member: '+name)
 return t.extractfile(e).read()
def main():
 contract=(BASE/'acceptance-v1.json').read_bytes()
 if sha(contract)!='8d130f74a4dca3085d4bed71586591473bb215e9aa533c8d654ea725738d6aaa':raise ValueError('locked contract changed')
 limits=json.loads(contract)['maximum_candidate_over_comparator_ratio']
 if limits!={'median_latency':1.05,'p95_latency':1.10,'comparable_process_peak_host_memory':1.05,'comparable_steady_state_live_memory':1.05}:raise ValueError('locked limits differ')
 exception=(BASE/'apple-immediate-memory-exception-v1.md').read_bytes()
 if sha(exception)!='11ab4be2844363d0fe5850398dc3d826cbebcfccc47da51a4c8209d15cab8895':raise ValueError('recorded exception differs')
 binding=run(load('binding','phase2-current-bindings-v2.py'))
 if binding['candidate']!='f155794' or binding['linux_cells']!=78 or binding['apple_cells']!=78 or binding['native_cuda_pool_ratio_max']>1.05:raise ValueError('platform binding incomplete')
 cuda=run(load('cuda_capture','cuda-bound-trace-review-v2.py'));metal=run(load('metal_capture','mlx-bound-trace-review-v2.py'))
 if not cuda['matches_gated_qualification_bytes'] or not metal['matches_gated_qualification_bytes'] or not metal['native_libraries_match_control']:raise ValueError('trace receipt differs')
 old=load('quality','phase2-closure-bindings-v1.py');old.CANDIDATE='f155794'
 with archive('final-linux-closure-checks-v1.tar.xz','0775e2ea9bffcd262dd74dbfc74c26c5d2860b9689327d35371f629275576c93') as t:
  pb='ncm-final-tokenizer-linux-v1/';lb='ncm-final-cuda-lifetime-v1/'
  probe=member(t,pb+'probe.json');pm=json.loads(member(t,pb+'metadata.json'));lm=json.loads(member(t,lb+'metadata.json'));life_raw=member(t,lb+'report.json');life=json.loads(life_raw)
  if sha(probe)!=pm['report_sha256'] or pm['exit_code']!=0 or 'env -i PATH=/usr/bin:/bin' not in pm['command']:raise ValueError('Linux tokenizer receipt differs')
  if member(t,pb+'binary-before.sha256')!=member(t,pb+'binary-after.sha256'):raise ValueError('tokenizer binary changed')
  old.verify_tokenizer_probe(json.loads(probe))
  if sha(life_raw)!=lm['report_sha256'] or lm['exit_code']!=0 or member(t,lb+'binary-before.sha256')!=member(t,lb+'binary-after.sha256'):raise ValueError('CUDA lifetime receipt differs')
  if not life['all_checks_passed'] or life['manifest_sha256']!='ad6a99ad7d5ccc069e0fedf26a85f755e06cd76590463e1e04e17363e35d9832' or len(life['cycles'])!=10 or sum(len(c['requests']) for c in life['cycles'])!=144:raise ValueError('CUDA lifetime scope differs')
  baseline=life['initial']['cuda_pool']['used_current']
  for c in life['cycles']:
   if c['after_drop']['cuda_pool']['used_current']!=baseline or len({r['memory']['cuda_pool']['used_current'] for r in c['requests']})!=1:raise ValueError('CUDA lifetime growth')
  for path,digest in life['source_sha256'].items():
   if path.startswith('/') or '..' in Path(path).parts or sha(subprocess.check_output(['git','show',f'f155794:{path}'],cwd=ROOT))!=digest:raise ValueError('CUDA lifetime source differs')
 with archive('cuda-bounded-runtime-diagnostic-v1.tar.xz','ad2e5d3a6c8e9ca8638b8311badf0f1eac17fdadcefa489cac33361f8458d736') as t:
  log=member(t,'ncm-workspace-codec-evidence-v1/check-isolated-passed.log').decode()
  if 'Requested checks passed' not in log or '18 operator cases passed' not in log:raise ValueError('full CUDA check missing')
  for n in ('check-umask022-failed.log','check-umask077-failed.log'):member(t,'ncm-workspace-codec-evidence-v1/'+n)
 with archive('cuda-bounded-mlx-check-v1.tar.xz','9c6a9f314fe061932bccda06b5832a2022cb0b7e7672f4c34b808cfdb80eb3c2') as t:
  log=member(t,'ncm-bounded-mlx-evidence-v1/.reference/phase2-bounded-mlx-check-v1.log').decode()
  if 'Requested checks passed' not in log or 'Metal: 18 cases passed' not in log:raise ValueError('full Metal check missing')
  old.verify_tokenizer_probe(json.loads(member(t,'ncm-bounded-mlx-evidence-v1/.reference/phase2-bounded-mlx-tokenizers-v1.json')))
 with archive('mlx-memory-addmm-v1.tar.xz','af4f6961695404e20bfa46d974e1ce651ba18d07a839f688057eca0f4f4628be') as t:
  mlife=json.loads(member(t,'reports/phase2-mlx-long-lived-memory-addmm-v1.json'))
  if len(mlife['cycles'])!=3 or any(c['dropped']['active']!=0 or c['requests'][-1]['request']!=128 for c in mlife['cycles']):raise ValueError('MLX lifetime evidence incomplete')
  if len({c['dropped']['cache'] for c in mlife['cycles']})!=1:raise ValueError('MLX cache sample differs')
  if subprocess.check_output(['git','diff','--name-only','a323b16','f155794','--','crates/ncm-backend-mlx','crates/ncm-model','crates/ncm-core','crates/ncm-io'],cwd=ROOT):raise ValueError('MLX lifetime source changed')
 with archive('apple-gate18-resource-memory-v1.tar.xz','524293f724bb1c047af88d481db476056f365b0a01aa3af50ddd5bd3a27348b6') as t:
  base='reports/phase2-mlx-live-memory-v1/';meta=json.loads(member(t,base+'metadata.json'));roles=('previous','candidate','python')
  if meta['status']!='complete' or {(x['role'],x['repeat']) for x in meta['commands']}!={(r,i) for r in roles for i in (1,2,3)}:raise ValueError('MLX memory jobs incomplete')
  reports={r:[] for r in roles}
  for x in meta['commands']:
   raw=member(t,base+f"{x['role']}-{x['repeat']}.json")
   if x['returncode']!=0 or sha(raw)!=x['report_sha256']:raise ValueError('MLX memory report differs')
   report=json.loads(raw)
   if len(report['rows'])!=12 or report['after_model_drop']['active']!=0:raise ValueError('MLX memory report incomplete')
   reports[x['role']].append(report)
  violations=[];peak=None
  for batch in (1,2,4):
   for scope in ('encode','decode','reconstruct','resident_decoder_block'):
    rows={r:[next(x for x in report['rows'] if (x['batch'],x['scope'])==(batch,scope)) for report in group] for r,group in reports.items()}
    for key in ('input_ids_sha256','supplied_codes_sha256','output_ids_sha256','latent_sha256'):
     if len({x[key] for group in rows.values() for x in group})!=1:raise ValueError('MLX retained-root identity differs')
    for stage in ('immediate','synchronized','after_10ms','after_100ms'):
     values={r:max(x[stage]['active'] for x in group) for r,group in rows.items()}
     for r in ('previous','python'):
      ratio=values['candidate']/values[r]
      if not math.isfinite(ratio) or ratio<=0:raise ValueError('invalid MLX ratio')
      if ratio>1.05:violations.append({'batch':batch,'scope':scope,'stage':stage,'comparator':r,'ratio':ratio,'bytes':values})
    if (batch,scope)==(4,'encode'):
     peak={r:max(x['immediate']['peak'] for x in group) for r,group in rows.items()}
  if {(v['batch'],v['scope'],v['stage'],v['comparator']) for v in violations}!={(4,'resident_decoder_block','immediate','previous'),(4,'resident_decoder_block','immediate','python')} or len(violations)!=2:raise ValueError('exception scope differs')
  if any(v['bytes']!={'previous':65835532,'candidate':69702948,'python':65835528} for v in violations):raise ValueError('exception values differ')
  if peak!={'previous':129751100,'candidate':155866252,'python':129750040}:raise ValueError('Apple peak values differ')
  cpu=json.loads(member(t,'reports/phase2-mlx-gate18-resources-v1/qualified-summary.json'))
  if cpu['diagnostic_only'] is not True or cpu['controlled_audit_sha256']!='a3ed81700b05f39a50a29a44340ffb00fb826fa738539989eb41e78556d58e4e' or set(cpu['summary'])!={'codec','bert','gpt2','qwen3','tinystories'}:raise ValueError('Apple CPU evidence incomplete')
 ledger=json.loads((BASE/'experiments.json').read_bytes());experiments=ledger['experiments'];ids=[x['id'] for x in experiments]
 required={'prepared-state','hf-id-only','tiny-in-place-merges','reconstruction-latent-reuse','decoder-trace-label-elision','streamed-artifact-hash','packed-qkv-layout','mlx-vectorized-resampling','tiny-compact-lookup-or-heap','bert-decoder-cleanup','checked-raw-upload','cuda-layout-selection-resampling','mlx-fast-norm-compile-attention','cuda-graph-replay','level-zero-lookup-and-new-loader-ownership','cuda-bounded-workspace','mlx-stream-synchronize'}
 if len(ids)!=45 or len(set(ids))!=45 or not required.issubset(ids):raise ValueError('experiment inventory incomplete')
 for x in experiments:
  if not all(x.get(k) for k in ('id','hypothesis','status','evidence')) or 'pending' in x['status'].lower() or not (ROOT/x['evidence'].split('#')[0]).is_file():raise ValueError('experiment disposition incomplete: '+x['id'])
 print(json.dumps({'schema_version':1,'scope':'evidence and recorded exception consistency; human approval and historical execution are not machine-authenticated','candidate':'f155794','platform_audits':{'linux':78,'apple':78},'trained_traces_each':642,'decisions_each':2500,'tokenizer_profiles_each':4,'cuda_lifetimes':10,'cuda_requests':144,'apple_lifetimes':3,'apple_requests_per_lifetime':128,'native_cuda_pool_ratio_max':binding['native_cuda_pool_ratio_max'],'apple_live_memory_violations':violations,'recorded_exception_scope':'only B4 resident-block immediate snapshot against the two recorded comparators','apple_b4_encode_peak_bytes':peak,'peak_not_part_of_exception':True,'experiments_dispositioned':len(ids),'limitations':['User approval is recorded from the conversation and cannot be machine-authenticated by this script.','Native MLX peak/active/cache counters are not host RSS or disjoint physical memory.','Apple per-request CPU and allocation counts and energy remain unavailable.','Finite lifetime runs do not establish an unbounded leak-free guarantee.','Large licensed trace/tokenizer assets remain outside Git pending separate publication authorization.'],'evidence_audit_passed':True},indent=2))
if __name__=='__main__':main()
