"""Run the predeclared matched retained-root MLX memory diagnostic."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[3]
os.chdir(ROOT)
OUT=ROOT/'reports/phase2-mlx-live-memory-v1'
AUDIT=ROOT/'reports/phase2-mlx-gate18-controlled-audit-v1.json'
ARTIFACT=ROOT/'artifacts/canonical/sudoku-easy-qualified'
SOURCES={role:ROOT/f'.reference/phase2-mlx-live-{role}-v1' for role in ('previous','candidate')}
BINARIES={role:ROOT/f'.reference/phase2-mlx-live-{role}-target-v1/release/examples/mlx_live_memory_probe' for role in SOURCES}
PYTHON=ROOT/'.reference/mlx-venv/bin/python'
SCRIPT=ROOT/'docs/results/2026-11-11-phase2-optimization/mlx-live-memory-probe-v1.py'

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    audit=json.loads(AUDIT.read_bytes())
    if audit['all_passed'] is not True or audit['quality_passed'] is not True or len(audit['cells'])!=78:
        raise ValueError('controlled gate did not pass')
    for path,digest in audit['input_sha256'].items():
        if sha(Path(path))!=digest:raise ValueError('gate input changed: '+path)
    if sha(ROOT/'tools/bench/audit_phase2_performance.py')!=audit['audit_source_sha256']:
        raise ValueError('gate audit source changed')
    control=ROOT/'reports/phase2-mlx-gate18-control-v1'
    for workload in ('codec','bert','gpt2','qwen3','tinystories'):
        item=audit['controls'][workload]
        if item['processes_per_runtime']!=18:raise ValueError('wrong control process count')
        stem='codec' if workload=='codec' else 'tokenizer-'+workload
        for role in ('previous','candidate','python'):
            hashes=item['report_sha256'][role]
            if len(hashes)!=18:raise ValueError('incomplete control hashes')
            suffix='python' if role=='python' else 'rust'
            for i,digest in enumerate(hashes,1):
                if sha(control/role/f'{stem}-{suffix}-{i}.json')!=digest:
                    raise ValueError('control report changed')
    if sha(ARTIFACT/'manifest.json')!='ad6a99ad7d5ccc069e0fedf26a85f755e06cd76590463e1e04e17363e35d9832':
        raise ValueError('unexpected artifact manifest')
    if OUT.exists():raise ValueError('output already exists')
    if any(os.environ.get(k) for k in ('DYLD_INSERT_LIBRARIES','MallocStackLogging','MallocStackLoggingNoCompact','LD_PRELOAD')):
        raise ValueError('instrumented environment')
    expected_source=sha(ROOT/'docs/results/2026-11-11-phase2-optimization/mlx-live-memory-probe-v1.rs')
    for role in SOURCES:
        if sha(SOURCES[role]/'crates/ncm-backend-mlx/examples/mlx_live_memory_probe.rs')!=expected_source:
            raise ValueError('diagnostic source mismatch')
        if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=SOURCES[role],text=True).strip():
            raise ValueError('modified tracked diagnostic checkout')
    previous=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCES['previous'],text=True).strip()
    if previous!='3307dff82a08f7251b64de0866a01d131528b74f':raise ValueError('wrong previous checkout')
    changed=subprocess.check_output(['git','diff','--name-only','5200924','HEAD','--','crates','Cargo.lock','vendor'],cwd=SOURCES['candidate'],text=True).strip()
    if changed:raise ValueError('candidate production sources differ')
    OUT.mkdir(mode=0o700,parents=True)
    meta={'schema_version':1,'status':'running','audit_sha256':sha(AUDIT),'manifest_sha256':sha(ARTIFACT/'manifest.json'),
          'example_sha256':expected_source,'python_source_sha256':sha(SCRIPT),'binaries_sha256':{k:sha(v) for k,v in BINARIES.items()},
          'source_heads':{k:subprocess.check_output(['git','rev-parse','HEAD'],cwd=v,text=True).strip() for k,v in SOURCES.items()},
          'source_sha256':{k:{name:sha(v/name) for name in ('crates/ncm-backend-mlx/src/codec.rs','crates/ncm-backend-mlx/src/ops.rs','crates/ncm-core/src/lib.rs','crates/ncm-model/src/blocks.rs','Cargo.lock')} for k,v in SOURCES.items()},
          'depfile_sha256':{k:sha(v.with_suffix('.d')) for k,v in BINARIES.items()},
          'commands':[],'thermal_before':subprocess.check_output(['/usr/bin/pmset','-g','therm'],text=True)}
    stop=threading.Event()
    def save(): (OUT/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    def sample():
        try:
            with (OUT/'activity.jsonl').open('x') as f:
                while not stop.is_set():
                    result=subprocess.check_output(['/bin/ps','-Ao','pcpu=,comm='],text=True)
                    entries=[]
                    for line in result.splitlines():
                        cpu,name=line.split(None,1);entries.append({'cpu_percent':float(cpu),'name':Path(name).name})
                    f.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'top_cpu':sorted(entries,key=lambda x:x['cpu_percent'],reverse=True)[:12]})+'\n');f.flush();stop.wait(2)
        except Exception as e:meta['activity_error']=str(e)
    save();thread=threading.Thread(target=sample);thread.start()
    try:
        names=('previous','python','candidate')
        for repeat in range(1,4):
            offset=(repeat-1)%3
            for role in names[offset:]+names[:offset]:
                report=OUT/f'{role}-{repeat}.json'
                if role=='python':argv=[str(PYTHON),str(SCRIPT),str(ARTIFACT),str(report)];cwd=ROOT
                else:argv=[str(BINARIES[role]),str(ARTIFACT),str(report)];cwd=SOURCES[role]
                entry={'role':role,'repeat':repeat,'argv':argv,'cwd':str(cwd),'start':datetime.now(timezone.utc).isoformat()};meta['commands'].append(entry);save()
                with (OUT/f'{role}-{repeat}.stdout').open('xb') as stdout,(OUT/f'{role}-{repeat}.stderr').open('xb') as stderr:
                    child=subprocess.run(argv,cwd=cwd,env={'PATH':'/usr/bin:/bin','HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'},stdout=stdout,stderr=stderr,timeout=120,check=False)
                entry['returncode']=child.returncode;entry['finish']=datetime.now(timezone.utc).isoformat()
                if child.returncode!=0:raise RuntimeError(f'{role} {repeat} failed')
                entry['report_sha256']=sha(report);save()
    except Exception as e:
        meta['status']='failed';meta['error']=str(e);raise
    finally:
        stop.set();thread.join();meta['thermal_after']=subprocess.check_output(['/usr/bin/pmset','-g','therm'],text=True);save()
    if 'activity_error' in meta:
        meta['status']='failed';save();raise RuntimeError(meta['activity_error'])
    meta['status']='complete';save()
    print('Nine live-memory diagnostic processes completed',flush=True)

if __name__=='__main__':main()
