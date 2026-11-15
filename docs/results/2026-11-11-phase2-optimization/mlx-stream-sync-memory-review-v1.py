"""Review isolated MLX stream synchronization memory reports; no parity or gate claim."""
import hashlib,json,statistics,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
ARCHIVE=ROOT/'docs/results/2026-11-11-phase2-optimization/raw/mlx-stream-sync-memory-v1.tar.xz'
EXPECTED='7c1d80b58dc34c78ffcd6a7d5201d75d1ad7e93a10556a9ddf18e9d9453369ad'
sha=lambda b:hashlib.sha256(b).hexdigest()
if sha(ARCHIVE.read_bytes())!=EXPECTED:raise ValueError('archive mismatch')
base='ncm-mlx-stream-sync-evidence-v1/reports/phase2-mlx-stream-sync-memory-v1/'
with tarfile.open(ARCHIVE,'r:xz') as t:
 def read(name):
  item=t.getmember(name)
  if not item.isfile():raise ValueError('expected regular file')
  return t.extractfile(item).read()
 meta=json.loads(read(base+'metadata.json'))
 if meta['status']!='complete' or {(x['role'],x['repeat']) for x in meta['commands']}!={(r,i) for r in ('event','sync') for i in (1,2,3)}:raise ValueError('incomplete jobs')
 reports={r:[] for r in ('event','sync')}
 for row in meta['commands']:
  raw=read(base+f"{row['role']}-{row['repeat']}.json")
  if sha(raw)!=row['report_sha256'] or row['returncode']!=0:raise ValueError('report mismatch')
  report=json.loads(raw)
  if len(report['rows'])!=12 or report['after_model_drop']['active']!=0:raise ValueError('bad scope or cleanup')
  reports[row['role']].append(report)
 rows=[]
 for batch in (1,2,4):
  for scope in ('encode','decode','reconstruct','resident_decoder_block'):
   values={role:[next(x for x in report['rows'] if (x['batch'],x['scope'])==(batch,scope)) for report in group] for role,group in reports.items()}
   for key in ('input_ids_sha256','supplied_codes_sha256','output_ids_sha256','latent_sha256'):
    if len({x[key] for group in values.values() for x in group})!=1:raise ValueError('identity mismatch')
   snapshots={role:{stage:[x[stage]['active'] for x in group] for stage in ('immediate','synchronized','after_10ms','after_100ms')} for role,group in values.items()}
   rows.append({'batch':batch,'scope':scope,'active_bytes':snapshots,'immediate_sync_over_event':statistics.median(snapshots['sync']['immediate'])/statistics.median(snapshots['event']['immediate'])})
 print(json.dumps({'scope':'memory diagnostic; float output parity and timing not measured','archive_sha256':EXPECTED,'all_identity_hashes_match':True,'after_drop_active_zero':True,'rows':rows},indent=2))
