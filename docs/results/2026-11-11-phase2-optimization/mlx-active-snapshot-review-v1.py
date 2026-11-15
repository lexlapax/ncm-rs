"""Review retained MLX active-allocation snapshots; not an acceptance audit."""
import hashlib
import importlib.util
import json
import statistics
import tarfile
from pathlib import Path

root=Path.cwd()
p=root/'docs/results/2026-11-11-phase2-optimization/mlx-all-windows-exploratory-v1.py'
spec=importlib.util.spec_from_file_location('analysis',p)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
groups, audits=module.load()
module.verify_original_points(groups, audits)
for window,(archive,prefix,_) in enumerate(module.ARCHIVES):
    groups={}
    memory_scopes=set()
    with tarfile.open(archive) as t:
        for role in module.RUNTIMES:
            suffix='python' if role=='python' else 'rust'
            reports=[]
            for i in range(1,10):
                raw=t.extractfile(f'{prefix}/{role}/codec-{suffix}-{i}.json').read()
                assert hashlib.sha256(raw).hexdigest()==audits[window]['controls']['codec']['report_sha256'][role][i-1]
                report=json.loads(raw);assert report['trials']==30 and report['warmups']==5
                reports.append(report)
            assert len({r['memory_scope'] for r in reports})==1
            memory_scopes.add(reports[0]['memory_scope'])
            groups[role]={(x['batch'],x['scope']):[next(y['allocator_bytes']['active'] for y in r['measurements'] if (y['batch'],y['scope'])==(x['batch'],x['scope'])) for r in reports] for x in reports[0]['measurements']}
            assert len(groups[role])==12
            assert all(len(v)==9 and all(type(n) is int and n>0 for n in v) for v in groups[role].values())
    assert len(memory_scopes)==1
    print('window',window+1)
    for key in sorted(groups['candidate']):
        a=groups['candidate'][key]; row={'batch':key[0],'scope':key[1],'candidate_range':[min(a),max(a)]}
        for role in ('previous','python'):
            b=groups[role][key]
            row[role+'_range']=[min(b),max(b)]
            row[role+'_median_ratio']=round(statistics.median(a)/statistics.median(b),6)
            row[role+'_max_ratio']=round(max(a)/max(b),6)
        print(json.dumps(row,sort_keys=True))
