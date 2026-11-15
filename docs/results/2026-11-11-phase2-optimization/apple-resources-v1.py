"""Post-gate Apple whole-process CPU diagnostics; never acceptance timings."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT/'tools/bench'))
from run_process_resources import run
from phase2_summary import codec, tokenizers
from phase2_mlx_summary import check_baseline

MATRIX = ROOT/'reports/phase2-mlx-gate18-matrix-v1'
AUDIT = ROOT/'reports/phase2-mlx-gate18-controlled-audit-v1.json'
OUTPUT = ROOT/'reports/phase2-mlx-gate18-resources-v1'
BASE = ROOT/'.reference/phase2-optimization-baseline'
OLD = ROOT/'.reference/phase2-optimization-mlx-bin'
WORK = ('codec', 'bert', 'gpt2', 'qwen3', 'tinystories')
ROLES = ('previous', 'candidate', 'python')

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def jobs():
    result=[]
    for workload in WORK:
        for role in ROLES:
            cwd = BASE if role == 'previous' else ROOT
            if workload == 'codec':
                bundle = str(ROOT/'artifacts/canonical/sudoku-easy-qualified')
                fixture = 'tests/fixtures/codec-bench-v1.json'
                if role == 'python':
                    argv = [str(ROOT/'.reference/mlx-venv/bin/python'), 'tools/reference/mlx_codec_bench.py', bundle, fixture, str(ROOT/'reports/phase2-codec/python-mlx-v1'), '{report}']
                else:
                    directory = OLD if role == 'previous' else ROOT/'target/release/examples'
                    argv = [str(directory/'mlx_codec_bench'), bundle, fixture, '{report}']
            else:
                fixture = 'tests/fixtures/tokenizers-v1.json'
                if role == 'python':
                    argv = [str(ROOT/'.reference/tokenizers-venv/bin/python'), 'tools/reference/text_tokenizer_bench.py', str(ROOT/'artifacts/upstream/tokenizers-v2'), str(ROOT/'.reference/upstream-ncm'), fixture, workload, '{report}']
                else:
                    directory = OLD if role == 'previous' else ROOT/'target/release/examples'
                    argv = [str(directory/'asset_bench'), str(ROOT/'artifacts/canonical/tokenizers-qualified'), fixture, workload, '{report}']
            result.append({'name': f'{role}-{workload}', 'argv': argv, 'cwd': str(cwd)})
    return result

def preflight():
    audit=json.loads(AUDIT.read_bytes())
    require(audit['all_passed'] is True and audit['quality_passed'] is True and len(audit['cells'])==78, 'controlled gate is not passing')
    require(audit['limits']=={'median':1.05,'p95':1.10,'comparable_host_memory':1.05}, 'acceptance limits differ')
    for path, digest in audit['input_sha256'].items():
        require(sha(Path(path))==digest, f'gate input changed: {path}')
    require(sha(ROOT/'tools/bench/audit_phase2_performance.py')==audit['audit_source_sha256'], 'audit code changed')
    control=ROOT/'reports/phase2-mlx-gate18-control-v1'
    for work in WORK:
        stem='codec' if work=='codec' else f'tokenizer-{work}'
        control_report=audit['controls'][work]
        require(control_report['processes_per_runtime']==18, 'wrong control process count')
        for role in ROLES:
            suffix='python' if role=='python' else 'rust'
            hashes=control_report['report_sha256'][role]
            require(len(hashes)==18, 'incomplete control hashes')
            for i,digest in enumerate(hashes,1):
                require(sha(control/role/f'{stem}-{suffix}-{i}.json')==digest, 'control report changed')
    env=json.loads((MATRIX/'environment.json').read_bytes())
    code = ('import importlib.metadata as m,json; '
            'print(json.dumps({d.metadata["Name"]:d.version for d in m.distributions()},sort_keys=True))')
    for name in ('mlx-venv','tokenizers-venv'):
        python=str(ROOT/'.reference'/name/'bin/python')
        version=subprocess.check_output([python,'--version'],text=True).strip()
        packages=json.loads(subprocess.check_output([python,'-c',code],text=True))
        require({'python':version,'packages':packages}==env[name], f'{name} environment changed')
    for role in ('previous','candidate'):
        directory = OLD if role == 'previous' else ROOT/'target/release/examples'
        for name in ('mlx_codec_bench','asset_bench'):
            require(sha(directory/name)==env['binary_sha256'][role][name], 'benchmark executable changed')
    require(not any(os.environ.get(k) for k in ('DYLD_INSERT_LIBRARIES','MallocStackLogging','MallocStackLoggingNoCompact','LD_PRELOAD')), 'instrumented environment')
    require(not OUTPUT.exists(), 'resource output exists')
    return audit

def qualify(resources):
    require(resources['status']=='complete' and len(resources['runs'])==45, 'incomplete resource run')
    require({(x['name'],x['repeat']) for x in resources['runs']}=={(f'{role}-{work}',i) for work in WORK for role in ROLES for i in range(1,4)}, 'wrong resource inventory')
    for row in resources['runs']:
        report=OUTPUT/f"{row['name']}-{row['repeat']}.json"
        require(row['exit_code']==0 and sha(report)==row['report_sha256'], 'resource report changed or failed')
    with tempfile.TemporaryDirectory(prefix='ncm-mlx-resource-audit-') as temp:
        root=Path(temp)
        for role in ('previous','candidate'):
            dest=root/role;dest.mkdir()
            for work in WORK:
                stem='codec' if work=='codec' else f'tokenizer-{work}'
                for i in range(1,4):
                    for source,suffix in ((role,'rust'),('python','python')):
                        report=OUTPUT/f'{source}-{work}-{i}.json'
                        shutil.copyfile(report,dest/f'{stem}-{suffix}-{i}.json')
                        raw=json.loads(report.read_bytes())
                        matrix=json.loads((MATRIX/role/f'{stem}-{suffix}-{i}.json').read_bytes())
                        require(raw['source_sha256']==matrix['source_sha256'], 'resource report source changed')
            check_baseline(dest);codec(dest,'metal');tokenizers(dest)
    summary={}
    for work in WORK:
        summary[work]={}
        for role in ROLES:
            rows=[x for x in resources['runs'] if x['name']==f'{role}-{work}']
            summary[work][role]={'cpu_ms_median':statistics.median(x['user_cpu_ms']+x['system_cpu_ms'] for x in rows),
                                 'cpu_ms': [x['user_cpu_ms']+x['system_cpu_ms'] for x in rows]}
    return summary

def main():
    audit=preflight()
    stop=threading.Event(); observations=[]
    def sample():
        try:
            while not stop.is_set():
                output=subprocess.check_output(['/bin/ps','-Ao','pcpu=,comm='],text=True)
                entries=[]
                for line in output.splitlines():
                    cpu,name=line.split(None,1);entries.append({'cpu_percent':float(cpu),'name':Path(name).name})
                observations.append({'utc':datetime.now(timezone.utc).isoformat(),'top_cpu':sorted(entries,key=lambda x:x['cpu_percent'],reverse=True)[:12]})
                stop.wait(2)
        except Exception as exc:
            observations.append({'error':str(exc)})
    before=subprocess.check_output(['/usr/bin/pmset','-g','therm'],text=True)
    thread=threading.Thread(target=sample);thread.start()
    try:
        resources=run(jobs(),OUTPUT,repeats=3)
    finally:
        stop.set();thread.join()
        if OUTPUT.exists():
            (OUTPUT/'activity.json').write_text(json.dumps({'samples':observations,'thermal_before':before,'thermal_after':subprocess.check_output(['/usr/bin/pmset','-g','therm'],text=True)},indent=2)+'\n')
    require(observations and not any('error' in x for x in observations),'activity observation failed')
    summary=qualify(resources)
    with (OUTPUT/'qualified-summary.json').open('x') as f:
        json.dump({'schema_version':1,'diagnostic_only':True,'scope':'whole process CPU; not per-request CPU or allocation counts',
                   'controlled_audit_sha256':sha(AUDIT),'jobs_source_sha256':sha(Path(__file__)),
                   'limitations':['includes startup, load, all trials, serialization and teardown','wait4 RSS is not acceptance memory','no allocation counts or GPU exclusivity claim'],
                   'summary':summary},f,indent=2);f.write('\n')
    print('qualified Apple CPU diagnostic complete',flush=True)

if __name__=='__main__':main()
