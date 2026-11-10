"""Counterbalanced three-way Apple measurements; no CPU-affinity claim on macOS."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import threading
from run_phase2_linux_comparison import order

ROOT = Path(__file__).resolve().parents[2]


def capture(command, cwd=ROOT):
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('previous_checkout', type=Path)
    p.add_argument('previous_binaries', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--workloads', nargs='+', choices=['codec','bert','gpt2','qwen3','tinystories'], default=['codec','bert','gpt2','qwen3','tinystories'])
    a = p.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64': raise ValueError('Apple Silicon required')
    if a.repeats < 3 or len(set(a.workloads)) != len(a.workloads): raise ValueError('invalid process/workload count')
    if any(os.environ.get(k) for k in ('DYLD_INSERT_LIBRARIES','MallocStackLogging','MallocStackLoggingNoCompact')): raise ValueError('instrumented environment')
    previous, old, output = a.previous_checkout.resolve(), a.previous_binaries.resolve(), a.output.resolve()
    if capture(['git','status','--porcelain'],previous): raise ValueError('previous checkout must be clean')
    bins = {'previous':old, 'candidate':ROOT/'target/release/examples'}
    output.mkdir(parents=True, exist_ok=False)
    for name in ('previous','candidate','python'): (output/name).mkdir()
    env = {'started_utc':datetime.now(timezone.utc).isoformat(), 'platform':'Darwin',
           'source_heads':{n:capture(['git','rev-parse','HEAD'], previous if n=='previous' else ROOT) for n in bins},
           'candidate_status':capture(['git','status','--short']),
           'binary_sha256':{n:{b:hashlib.sha256((d/b).read_bytes()).hexdigest() for b in ('mlx_codec_bench','asset_bench')} for n,d in bins.items()},
           'native_libraries_sha256':{n:hashlib.sha256((ROOT/'.reference/mlx-native/lib'/n).read_bytes()).hexdigest() for n in ('libmlx.dylib','mlx.metallib')},
           'rustc':capture(['rustc','--version']), 'os':capture(['sw_vers']), 'cpu':capture(['sysctl','-n','machdep.cpu.brand_string']),
           'power_settings':capture(['pmset','-g','custom']), 'thermal_before':capture(['pmset','-g','therm']),
           'repeats':a.repeats,'workloads':a.workloads,'cpu_affinity':None,'control_policy':'counterbalanced_fresh_processes',
           'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'commands':[],
           'limitations':['No CPU affinity/exclusive workstation reservation; CPU activity sampled every two seconds.',
                          'No privileged GPU telemetry, clock/power/global tooling changes.',
                          'Warm filesystem cache; import/context startup excluded from existing benchmark scopes.',
                          'One Python report shared by both comparisons, not independent duplicate evidence.']}
    for v in ('mlx-venv','tokenizers-venv'):
        py = str(ROOT/f'.reference/{v}/bin/python')
        env[v] = {'python':capture([py,'--version']), 'packages':json.loads(capture([py,'-c',"import importlib.metadata as m,json;print(json.dumps({d.metadata['Name']:d.version for d in m.distributions()}))"]))}
    stop = threading.Event()
    def save(): (output/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
    def sample():
        try:
            with (output/'cpu-activity.jsonl').open('x') as f:
                while not stop.is_set():
                    entries=[]
                    for line in capture(['ps','-Ao','pid=,pcpu=,comm=']).splitlines():
                        pid,cpu,name=line.split(None,2);entries.append({'pid':int(pid),'cpu_percent':float(cpu),'name':Path(name).name})
                    f.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'load_average':os.getloadavg(),'top_cpu':sorted(entries,key=lambda r:r['cpu_percent'],reverse=True)[:12]})+'\n');f.flush();stop.wait(2)
        except Exception as exc: env['telemetry_error']=str(exc)
    thread = threading.Thread(target=sample);thread.start()
    try:
        for repeat in range(1,a.repeats+1):
            for profile in a.workloads:
                for runtime in order(repeat):
                    prefix='codec' if profile=='codec' else f'tokenizer-{profile}'
                    report=output/runtime/f"{prefix}-{'python' if runtime=='python' else 'rust'}-{repeat}.json"
                    cwd=previous if runtime=='previous' else ROOT
                    if profile=='codec':
                        bundle=str(ROOT/'artifacts/canonical/sudoku-easy-qualified')
                        fixture='tests/fixtures/codec-bench-v1.json'
                        if runtime=='python': command=[str(ROOT/'.reference/mlx-venv/bin/python'),'tools/reference/mlx_codec_bench.py',bundle,fixture,str(ROOT/'reports/phase2-codec/python-mlx-v1'),str(report)]
                        else: command=[str(bins[runtime]/'mlx_codec_bench'),bundle,fixture,str(report)]
                    else:
                        if runtime=='python': command=[str(ROOT/'.reference/tokenizers-venv/bin/python'),'tools/reference/text_tokenizer_bench.py',str(ROOT/'artifacts/upstream/tokenizers-v2'),str(ROOT/'.reference/upstream-ncm')]
                        else: command=[str(bins[runtime]/'asset_bench'),str(ROOT/'artifacts/canonical/tokenizers-qualified')]
                        command+=['tests/fixtures/tokenizers-v1.json',profile,str(report)]
                    entry={'argv':command,'cwd':str(cwd),'runtime':runtime,'started_utc':datetime.now(timezone.utc).isoformat()};env['commands'].append(entry);save()
                    child_env={'PATH':'/usr/bin:/bin','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1'}
                    subprocess.run(command,cwd=cwd,env=child_env,check=True)
                    entry.update(finished_utc=datetime.now(timezone.utc).isoformat(),report_sha256=hashlib.sha256(report.read_bytes()).hexdigest());save()
                    if runtime=='python':
                        for name in bins: shutil.copyfile(report,output/name/report.name)
    finally:
        stop.set();thread.join();env['finished_utc']=datetime.now(timezone.utc).isoformat();env['thermal_after']=capture(['pmset','-g','therm']);save()
    if 'telemetry_error' in env: raise RuntimeError(env['telemetry_error'])
    print('Three-way measurements complete:',output)


if __name__=='__main__': main()
