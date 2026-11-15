"""Bind current candidate to archived platform evidence; not a closure decision."""
import hashlib, importlib.util, json, math, subprocess, tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'docs/results/2026-11-11-phase2-optimization'
RAW=BASE/'raw'
CANDIDATE='f155794'; PREVIOUS_APPLE='5200924'; BASELINE='3307dff'
spec=importlib.util.spec_from_file_location('prior_binding', BASE/'phase2-closure-bindings-v1.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior);prior.CANDIDATE=CANDIDATE
sha=lambda b:hashlib.sha256(b).hexdigest()
def checked_archive(name,digest):
 p=RAW/name
 if sha(p.read_bytes())!=digest:raise ValueError('archive mismatch: '+name)
 return tarfile.open(p,'r:xz')
def member(stream,name):
 entry=stream.getmember(name)
 if not entry.isfile():raise ValueError('expected regular member: '+name)
 return stream.extractfile(entry).read()
def audited(stream,member_name,published,digest,platform,quality_member):
 raw=member(stream,member_name)
 if sha(raw)!=digest or sha((BASE/published).read_bytes())!=digest:raise ValueError('audit mismatch: '+platform)
 a=json.loads(raw)
 if (a['all_passed'] is not True or a['quality_passed'] is not True or len(a['cells'])!=78
     or not all(x['passed'] is True for x in a['cells'])
     or a['limits']!={'median':1.05,'p95':1.10,'comparable_host_memory':1.05}
     or a['audit_source_sha256']!=sha((ROOT/'tools/bench/audit_phase2_performance.py').read_bytes())):raise ValueError('audit policy differs')
 qraw=member(stream,quality_member)
 if a['input_sha256'][quality_member]!=sha(qraw):raise ValueError('quality not bound')
 q=json.loads(qraw);prior.verify_quality(q,'rust-candle' if platform=='linux' else 'rust-mlx')
 prior.verify_coverage(a,platform)
 return a,q
def main():
 if prior.changed(CANDIDATE,'HEAD','--',*prior.GUARDED) or prior.changed('HEAD','--',*prior.GUARDED):raise ValueError('current guarded source differs')
 if subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',*prior.GUARDED],cwd=ROOT,text=True).strip():raise ValueError('untracked guarded source exists')
 if prior.changed(BASELINE,'HEAD','--',*prior.BASELINE_PINS) or prior.changed('HEAD','--',*prior.BASELINE_PINS):raise ValueError('baseline pins differ')
 changes=set(prior.changed(PREVIOUS_APPLE,CANDIDATE,'--',*prior.GUARDED))
 if changes!={'crates/ncm-backend-cuda/src/fused_projection.rs','crates/ncm-backend-cuda/src/fused_projection/bounded.rs'}:raise ValueError('unexpected guarded Apple-source changes')
 manifest='ad6a99ad7d5ccc069e0fedf26a85f755e06cd76590463e1e04e17363e35d9832'
 tracked_manifest=ROOT/'docs/results/2026-11-10-sudoku-artifacts/raw/artifact-manifest.json'
 if sha(tracked_manifest.read_bytes())!=manifest:raise ValueError('tracked artifact manifest differs')
 local_manifest=ROOT/'artifacts/canonical/sudoku-easy-qualified/manifest.json'
 if local_manifest.exists() and sha(local_manifest.read_bytes())!=manifest:raise ValueError('local artifact manifest differs')
 with checked_archive('cuda-bounded-linux-gate-v1.tar.xz','a2c3d9bcb754d7b511b8ce11812b56e3389565277edcbee8c56cc7ce181044d8') as linux, checked_archive('apple-gate18-resource-memory-v1.tar.xz','524293f724bb1c047af88d481db476056f365b0a01aa3af50ddd5bd3a27348b6') as apple, checked_archive('cuda-bounded-mlx-check-v1.tar.xz','9c6a9f314fe061932bccda06b5832a2022cb0b7e7672f4c34b808cfdb80eb3c2') as fresh, checked_archive('tiny-borrow-linux-gate-v1.tar.xz','b16be8834fa12e30b5c70d9ac4073461d9caf1f7ab5f2fb0673b3ce2fe1aa87e') as oldlinux:
  la,lq=audited(linux,'reports/phase2-cuda-bounded-linux-controlled-audit-v1.json','cuda-bounded-linux-acceptance-v1.json','5dee0d928c5825b58c23befa61d88d61cf0a072e5ce0890736e3579e8104ab57','linux','reports/phase2-cuda-bounded-qualification-v1.json')
  aa,aq=audited(apple,'reports/phase2-mlx-gate18-controlled-audit-v1.json','mlx-acceptance.json','a3ed81700b05f39a50a29a44340ffb00fb826fa738539989eb41e78556d58e4e','apple','.reference/phase2-mlx-gate18-preflight-v1/qualification.json')
  initial=json.loads(member(linux,'reports/phase2-cuda-bounded-linux-initial-audit-v1.json'))
  if initial['all_passed'] is not True or len(initial['cells'])!=78 or not all(c['passed'] for c in initial['cells']):raise ValueError('Linux initial audit incomplete')
  l_env=json.loads(member(linux,'reports/phase2-cuda-bounded-linux-control-v1/environment.json'))
  old_l_env=json.loads(member(oldlinux,'reports/phase2-tiny-borrow-linux-control-v1/environment.json'))
  if l_env['binary_sha256']['candidate']['asset_bench']!=old_l_env['binary_sha256']['candidate']['asset_bench']:raise ValueError('Linux tokenizer executable differs')
  old_probe=member(oldlinux,'reports/phase2-tiny-borrow-tokenizer-probe-v1.json')
  if sha(old_probe)!='6124d2dbd456dce9375e5d6bb17472875ae75674114eb51cd4f279046b08bee9':raise ValueError('Linux tokenizer probe differs')
  prior.verify_tokenizer_probe(json.loads(old_probe))
  observed=[]
  for phase,repeats in (('matrix',3),('control',9)):
   for i in range(1,repeats+1):
    rows={r:json.loads(member(linux,f'reports/phase2-cuda-bounded-linux-{phase}-v1/{r}/codec-rust-{i}.json')) for r in ('previous','candidate')}
    if any(data['manifest_sha256']!=manifest for data in rows.values()):raise ValueError('Linux report manifest differs')
    for b in (1,2,4):
     for scope in ('encode','decode','reconstruct','resident_decoder_block'):
      x={r:next(v for v in data['measurements'] if (v['batch'],v['scope'])==(b,scope)) for r,data in rows.items()}
      for k in ('input_ids_sha256','supplied_codes_sha256','output_ids_sha256','output_ids_count','output_float_elements'):
       if x['previous'][k]!=x['candidate'][k]:raise ValueError('Linux pool comparison identity mismatch')
      ratio=x['candidate']['pool_bytes']['used_current']/x['previous']['pool_bytes']['used_current']
      if not math.isfinite(ratio) or not 0<ratio<=1.05:raise ValueError('Linux native pool ratio exceeds band')
      observed.append(ratio)
  if len(observed)!=144:raise ValueError('Linux pool inventory incomplete')
  fresh_q=json.loads(member(fresh,'ncm-bounded-mlx-evidence-v1/qualification.json'));prior.verify_quality(fresh_q,'rust-mlx')
  if fresh_q['right_trace_sha256']!=aq['right_trace_sha256']:raise ValueError('Apple fresh trace differs')
  apple_probe=member(fresh,'ncm-bounded-mlx-evidence-v1/.reference/phase2-bounded-mlx-tokenizers-v1.json');prior.verify_tokenizer_probe(json.loads(apple_probe))
  observed_apple=json.loads(member(fresh,'ncm-bounded-mlx-evidence-v1/verification.json'))
  old_a_env=json.loads(member(apple,'reports/phase2-mlx-gate18-control-v1/environment.json'))
  old_a_codec=json.loads(member(apple,'reports/phase2-mlx-gate18-control-v1/candidate/codec-rust-1.json'))
  if old_a_codec['manifest_sha256']!=manifest:raise ValueError('Apple report manifest differs')
  for key in ('mlx_codec_bench','asset_bench'):
   if observed_apple['binary_sha256'][key]!=old_a_env['binary_sha256']['candidate'][key]:raise ValueError('Apple benchmark binary differs')
  lib_obs=json.loads((BASE/'mlx-native-library-identity-v1.json').read_bytes())
  if lib_obs['library_sha256']!=old_a_env['native_libraries_sha256'] or not lib_obs['matches_apple_control']:raise ValueError('Apple native libraries differ')
  if sha(member(fresh,'ncm-bounded-mlx-evidence-v1/.reference/phase2-bounded-mlx-check-v1.log'))!='aab6f6ac14483bbe6c1a2fd8f5195e5f256dd4a15e1dfae73904911d40dc96c3':raise ValueError('Apple check log differs')
  result={'schema_version':1,'scope':'current guarded source, historical gates and selected resource binding; not a Phase 2 completion decision','candidate':CANDIDATE,'historical_apple_candidate':PREVIOUS_APPLE,'linux_cells':len(la['cells']),'apple_cells':len(aa['cells']),'qualified_traces_each':642,'selection_decisions_each':2500,'native_cuda_pool_observations':len(observed),'native_cuda_pool_ratio_min':min(observed),'native_cuda_pool_ratio_max':max(observed),'tokenizer_profiles_each':4,'apple_benchmarks_match_prior_control':True,'apple_trace_hash_matches_prior_control':True,'apple_native_library_hashes_match_prior_control':True,'native_library_observation_limitations':lib_obs['limitations'],'apple_immediate_live_memory':'unresolved; both immediate and post-barrier measurements retained','phase_2_complete':False}
  print(json.dumps(result,indent=2))
if __name__=='__main__':main()
