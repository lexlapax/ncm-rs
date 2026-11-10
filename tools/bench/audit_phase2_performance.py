"""Fail-closed Phase 2 performance audit. Raw reports, not edited summaries, are inputs."""
import argparse
import hashlib
import json
from pathlib import Path
from phase2_summary import codec, tokenizers, reports, require
from summarize_tokenizer_control import summarize as control
from run_codec_lifecycle import validate

POLICY = {"primitive":{"atol":1e-5,"rtol":1e-4},"component":{"atol":1e-4,"rtol":2e-4},
          "distance":{"atol":1e-3,"rtol":2e-4},"accumulated":{"atol":5e-4,"rtol":2e-4}}
ORACLE = "89bee3830bf97485183a17d325f78183bcc6d5d573b2af312899b2ee9dfad7fa"


def quality(q):
    require(q['left_trace_sha256']==ORACLE and q['policy']==POLICY, 'oracle/tolerance mismatch')
    require(q['trace_count']==642 and len(q['traces'])==642 and q['all_passed'] is True and q['all_trace_elements_pass'] is True and q['failures']==[], 'incomplete/failed quality')
    require(all(r['elements']>0 and r['mismatches']==0 for r in q['traces']), 'trace failures')
    require(sum(r['decisions'] for r in q['selections'].values())==2500, 'selection scope mismatch')
    require(all(r['disagreements']==r['stable_disagreements']==r['candidate_selection_mismatches']==0 for r in q['selections'].values()), 'selection disagreement')


def audit(config):
    device=config['device'];require(device in ('cuda','metal'),'unsupported device')
    roots={name:Path(config[name]) for name in ('previous','candidate')}
    inputs={}
    for name,root in roots.items():
        for prefix in ['codec']+[f'tokenizer-{p}' for p in ('bert','gpt2','qwen3','tinystories')]:
            groups=reports(root,prefix)
            for runtime,runs in groups.items():
                expected=('rust-candle' if device=='cuda' else 'rust-mlx') if runtime=='rust' else ('pytorch-pinned-eager-snapshot-fix' if device=='cuda' else 'python-mlx-adapted')
                if prefix!='codec': expected='rust-tokenizers' if runtime=='rust' else 'pinned-python-tokenizers'
                for i,r in enumerate(runs,1):
                    require(r.get('runtime')==expected,'runtime mismatch')
                    require(r['source_sha256'] and all(isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v) for v in r['source_sha256'].values()),'invalid provenance')
                    path=root/f'{prefix}-{runtime}-{i}.json'
                    inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                    if runtime=='python': require(path.read_bytes()==(roots['previous']/path.name).read_bytes(),'unpaired Python reports')
        if device=='metal':
            from phase2_mlx_summary import check_baseline
            check_baseline(root)
    data={name:{'codec':codec(root,device),'tokenizers':tokenizers(root)} for name,root in roots.items()}
    qpath=Path(config['qualification']);q=json.loads(qpath.read_bytes());quality(q)
    inputs[str(qpath)]=hashlib.sha256(qpath.read_bytes()).hexdigest()
    life_path=Path(config['lifecycle']);life=json.loads(life_path.read_bytes())
    require(len(life['runs'])==9 and {(r['runtime'],r['repeat']) for r in life['runs']}=={(n,i) for n in ('previous','candidate','python') for i in (1,2,3)},'incomplete lifecycle')
    require(len({validate(r['phases']) for r in life['runs']})==1,'lifecycle output mismatch')
    inputs[str(life_path)]=hashlib.sha256(life_path.read_bytes()).hexdigest()
    controls={p:control(Path(path),p) for p,path in config.get('controls',{}).items()}
    rows=[]
    def timing(label,new,a,b,follow=None):
        for name,other in [('previous',a),('python',b)]:
            row={'cell':label,'comparator':name,'median_ratio':new['median_ms']/other['median_ms'],
                 'p95_ratio':new['p95_ms']/other['p95_ms'],
                 'process_median_envelope_ratio':max(new['process_medians_ms'])/min(other['process_medians_ms'])}
            row['original_point_pass']=row['median_ratio']<=1.05 and row['p95_ratio']<=1.10
            if follow:
                comparison=follow['candidate_over'][name]
                interval=comparison['paired_process_percentile_95_interval']
                row['controlled_followup']=comparison
                row['passed']=comparison['median_ratio']<=1.05 and comparison['p95_ratio']<=1.10 and interval['median_ratio'][1]<=1.05 and interval['p95_ratio'][1]<=1.10
            else: row['passed']=row['original_point_pass'] and row['process_median_envelope_ratio']<=1.05
            rows.append(row)
    old=data['previous'];new=data['candidate']
    for a,b in zip(old['codec']['measurements'],new['codec']['measurements']):
        timing(f"codec/{b['batch']}/{b['scope']}",b['runtimes']['rust']['wall'],a['runtimes']['rust']['wall'],b['runtimes']['python']['wall'])
    groups=[('codec',old['codec']['loads'],new['codec']['loads'])]
    for a,b in zip(old['tokenizers'],new['tokenizers']):
        profile=b['profile'];follow=controls.get(profile)
        if follow: require(follow['native_binaries_identical'],'identity follow-up cannot substitute changed binaries')
        for i,(x,y) in enumerate(zip(a['measurements'],b['measurements'])):
            timing(f"{profile}/{y['scope']}",y['runtimes']['rust']['timing'],x['runtimes']['rust']['timing'],y['runtimes']['python']['timing'],follow['measurements'][i] if follow else None)
        groups.append((profile,a['loads'],b['loads']))
    for label,a,b in groups:
        timing(label+'/checked_load',b['rust']['warm'],a['rust']['warm'],b['python']['warm'])
        for metric in ('VmHWM','VmRSS'):
            current=max(r[metric] for r in b['rust']['host_final'])
            for name,other in [('previous',a['rust']),('python',b['python'])]:
                baseline=max(r[metric] for r in other['host_final'])
                require(isinstance(current,int) and isinstance(baseline,int) and current>0 and baseline>0,'invalid host memory')
                ratio=current/baseline
                rows.append({'cell':label+'/'+metric,'comparator':name,'peak_observed_ratio':ratio,'passed':ratio<=1.05})
    return {'schema_version':1,'device':device,'all_passed':all(r['passed'] for r in rows),'quality_passed':True,
            'limits':{'median':1.05,'p95':1.10,'comparable_host_memory':1.05},'input_sha256':inputs,
            'controls':controls,'cells':rows,
            'limitations':['Observed per-process median envelope used for three-process cells; control intervals resample whole paired processes.',
                           'Memory uses maximum observed process RSS/HWM, not allocator counters as interchangeable device peaks.',
                           'Lifecycle is separately measured, not an invented disk-cold or steady-state timing.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('config',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();result=audit(json.loads(args.config.read_bytes()))
    result['audit_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x') as file:json.dump(result,file,indent=2,allow_nan=False);file.write('\n')
    print('all_passed:',result['all_passed'])
    for row in result['cells']:
        if not row['passed']:print('PENDING',row)
    if not result['all_passed']:raise SystemExit(1)


if __name__=='__main__':main()
