import copy, importlib.util, json, sys, tarfile, tempfile, time
from pathlib import Path
root = Path.cwd()
sys.path.insert(0, str(root / 'tools/bench'))
from summarize_codec_load_control import summarize as codec
from summarize_tokenizer_control import summarize as tokenizer
spec = importlib.util.spec_from_file_location('existing_analysis', root/'docs/results/2026-11-11-phase2-optimization/mlx-all-windows-exploratory-v1.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
groups, audits = module.load()
module.verify_original_points(groups, audits)
with tempfile.TemporaryDirectory(prefix='ncm-gate18-probe-') as temp:
    directory=Path(temp); [ (directory/name).mkdir() for name in ('previous','candidate','python') ]
    envs=[]
    for window,(path,prefix,audit) in enumerate(module.ARCHIVES):
        with tarfile.open(path) as t:
            envs.append(json.load(t.extractfile(prefix+'/environment.json')))
            for profile in module.PROFILES:
                stem='codec' if profile=='codec' else f'tokenizer-{profile}'
                for role in module.RUNTIMES:
                    suffix='python' if role=='python' else 'rust'
                    for i in range(1,10):
                        src=f'{prefix}/{role}/{stem}-{suffix}-{i}.json'
                        dest=directory/role/f'{stem}-{suffix}-{i+9*window}.json'
                        dest.write_bytes(t.extractfile(src).read())
            if window == 0: (directory/'cpu-activity.jsonl').write_bytes(t.extractfile(prefix+'/cpu-activity.jsonl').read())
    for key in ('binary_sha256','native_libraries_sha256','harness_sha256','mlx-venv','tokenizers-venv','os','rustc','workloads','cpu_affinity','control_policy'):
        assert envs[0][key]==envs[1][key], key
    environment=copy.deepcopy(envs[0]);environment['repeats']=18;environment['synthetic_offline_probe_not_acceptance']=True
    (directory/'environment.json').write_text(json.dumps(environment))
    matrix=directory/'matrix'; (matrix/'previous').mkdir(parents=True); (matrix/'candidate').mkdir()
    quiet_path=module.ARCHIVES[1][0]; matrix_prefix='reports/phase2-tiny-borrow-mlx-quiet-matrix-v1'
    with tarfile.open(quiet_path) as t:
        (matrix/'environment.json').write_bytes(t.extractfile(matrix_prefix+'/environment.json').read())
        (directory/'qualification.json').write_bytes(t.extractfile('reports/qualification.json').read())
        (directory/'lifecycle.json').write_bytes(t.extractfile('reports/phase2-tiny-borrow-mlx-lifecycle-v1.json').read())
        for role in ('previous','candidate'):
            for profile in module.PROFILES:
                stem='codec' if profile=='codec' else f'tokenizer-{profile}'
                for suffix in ('rust','python'):
                    for i in range(1,4):
                        name=f'{stem}-{suffix}-{i}.json'; (matrix/role/name).write_bytes(t.extractfile(f'{matrix_prefix}/{role}/{name}').read())
    started=time.monotonic(); c=codec(directory,'metal',include_execution=True)
    assert c['processes_per_runtime']==18 and len(c['measurements'])==13
    assert all(len(c['report_sha256'][name])==18 for name in module.RUNTIMES)
    print('codec execution and load: 13 scopes, 18 reports per runtime',flush=True)
    for profile in module.PROFILES[1:]:
        s=tokenizer(directory,profile,include_load=True)
        assert s['processes_per_runtime']==18 and len(s['measurements'])==4
        assert all(len(s['report_sha256'][name])==18 for name in module.RUNTIMES)
        print(profile,'4 scopes, 18 reports per runtime',flush=True)
    from audit_phase2_performance import audit
    config={'device':'metal','previous':str(matrix/'previous'),'candidate':str(matrix/'candidate'),
            'qualification':str(directory/'qualification.json'),'lifecycle':str(directory/'lifecycle.json'),
            'controls':{name:str(directory) for name in module.PROFILES},'load_controls':list(module.PROFILES),
            'codec_execution_controls':True,'control_matrix_environment':str(matrix/'environment.json')}
    result=audit(config)
    assert len(result['cells'])==78 and result['quality_passed'] is True
    assert all(result['controls'][name]['processes_per_runtime']==18 for name in module.PROFILES)
    print('full audit constructed 78 cells with 18-report controls; synthetic, not acceptance',flush=True)
    (directory/'candidate/codec-rust-18.json').unlink()
    try: audit(config)
    except (ValueError, FileNotFoundError): print('missing declared codec process rejected',flush=True)
    else: raise AssertionError('missing declared process accepted')
    print('completed offline probe; no acceptance status; elapsed seconds',round(time.monotonic()-started,1),flush=True)
