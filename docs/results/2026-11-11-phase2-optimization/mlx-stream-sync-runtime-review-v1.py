"""Review full-codec synchronization diagnostic; not an acceptance audit."""
import hashlib,json,statistics,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
ARCHIVE=ROOT/'docs/results/2026-11-11-phase2-optimization/raw/mlx-stream-sync-runtime-v1.tar.xz'
EXPECTED='9bdf504384b397c5bcbc07a2eef6e380c3d213dcb89ca2c902918686648b0d08'
sha=lambda b:hashlib.sha256(b).hexdigest()
if sha(ARCHIVE.read_bytes())!=EXPECTED:raise ValueError('archive mismatch')
base='ncm-mlx-sync-runtime-evidence-v1/reports/phase2-mlx-sync-runtime-diagnostic-v1/'
with tarfile.open(ARCHIVE,'r:xz') as t:
 def read(name):
  item=t.getmember(name)
  if not item.isfile():raise ValueError('expected regular member')
  return t.extractfile(item).read()
 meta=json.loads(read(base+'metadata.json'))
 jobs={(k,r,i) for k in ('codec','memory') for r in ('current','sync') for i in (1,2,3)}
 if meta['status']!='complete' or {(x['kind'],x['role'],x['repeat']) for x in meta['commands']}!=jobs:raise ValueError('incomplete jobs')
 reports={kind:{role:{} for role in ('current','sync')} for kind in ('codec','memory')}
 for row in meta['commands']:
  kind,role,i=row['kind'],row['role'],row['repeat'];raw=read(base+f'{kind}-{role}-{i}.json')
  if sha(raw)!=row['report_sha256'] or row['returncode']!=0:raise ValueError('report mismatch')
  report=json.loads(raw)
  if report['manifest_sha256']!=meta['manifest_sha256']:raise ValueError('manifest mismatch')
  reports[kind][role][i]=report
 out=[]
 for batch in (1,2,4):
  for scope in ('encode','decode','reconstruct','resident_decoder_block'):
   sets={kind:{role:[next(x for x in report['measurements' if kind=='codec' else 'rows'] if (x['batch'],x['scope'])==(batch,scope)) for _,report in sorted(group.items())] for role,group in roles.items()} for kind,roles in reports.items()}
   for kind,roles in sets.items():
    keys=('input_ids_sha256','supplied_codes_sha256','output_ids_sha256')
    if kind=='codec':keys+=('output_ids_count','output_float_elements')
    else:keys+=('latent_sha256',)
    for key in keys:
     if len({x[key] for group in roles.values() for x in group})!=1:raise ValueError('identity mismatch')
   a=[statistics.median(x['trials_ms']) for x in sets['codec']['current']];b=[statistics.median(x['trials_ms']) for x in sets['codec']['sync']]
   if any(len(x['trials_ms'])!=30 for group in sets['codec'].values() for x in group):raise ValueError('trial count mismatch')
   out.append({'batch':batch,'scope':scope,'current_process_medians_ms':a,'sync_process_medians_ms':b,'paired_median_ratios':[y/x for x,y in zip(a,b)],'median_of_medians_ratio':statistics.median(b)/statistics.median(a),'immediate_active_bytes':{r:[x['immediate']['active'] for x in group] for r,group in sets['memory'].items()}})
 for group in reports['memory'].values():
  if any(x['after_model_drop']['active']!=0 for x in group.values()):raise ValueError('nonzero drop')
 print(json.dumps({'scope':'exploratory process comparisons; float parity and acceptance not established','archive_sha256':EXPECTED,'rows':out,'all_id_and_latent_hashes_match':True,'all_post_drop_active_zero':True},indent=2))
