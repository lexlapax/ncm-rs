"""Review recorded Linux gate report, source and executable identities."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / 'reports'
AUDIT = REPORTS / 'phase2-cuda-bounded-linux-controlled-audit-v1.json'
EXPECTED_AUDIT = '5dee0d928c5825b58c23befa61d88d61cf0a072e5ce0890736e3579e8104ab57'
WORK = ('codec', 'bert', 'gpt2', 'qwen3', 'tinystories')
ROLES = ('previous', 'candidate', 'python')

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def main():
    raw = AUDIT.read_bytes()
    if sha(raw) != EXPECTED_AUDIT:
        raise ValueError('audit differs')
    audit = json.loads(raw)
    if not audit['all_passed'] or len(audit['cells']) != 78:
        raise ValueError('audit is not complete')
    cache = {}; counts = {role: 0 for role in ROLES}; copies = 0
    environments = []
    for phase, repeats in (('matrix', 3), ('control', 9)):
        base = REPORTS / f'phase2-cuda-bounded-linux-{phase}-v1'
        env = json.loads((base/'environment.json').read_bytes()); environments.append(env)
        if env['source_heads']['previous'][:7] != '3307dff' or env['source_heads']['candidate'][:7] != 'f155794':
            raise ValueError('wrong source heads')
        if len(env['commands']) != repeats * len(WORK) * len(ROLES):
            raise ValueError('incomplete command inventory')
        command_hashes = {Path(row['argv'][-1]).resolve(): row.get('report_sha256') for row in env['commands']}
        if len(command_hashes) != len(env['commands']):
            raise ValueError('duplicate report commands')
        for work in WORK:
            stem = 'codec' if work == 'codec' else 'tokenizer-' + work
            for role in ROLES:
                typ = 'python' if role == 'python' else 'rust'
                revision = '3307dff' if role == 'previous' else 'f155794'
                for i in range(1, repeats + 1):
                    path = base/role/f'{stem}-{typ}-{i}.json'; data=path.read_bytes()
                    if sha(data) != command_hashes.get(path.resolve()):
                        raise ValueError('command report mismatch: ' + str(path))
                    if phase == 'control' and sha(data) != audit['controls'][work]['report_sha256'][role][i-1]:
                        raise ValueError('control audit hash mismatch: ' + str(path))
                    if phase == 'matrix' and role != 'python' and sha(data) != audit['input_sha256'][str(path.relative_to(ROOT))]:
                        raise ValueError('matrix audit hash mismatch: ' + str(path))
                    inv=json.loads(data).get('source_sha256')
                    if not isinstance(inv,dict) or not inv: raise ValueError('missing inventory')
                    for source,digest in inv.items():
                        if role == 'python' and work == 'codec':
                            aliases = {'benchmark': 'tools/reference/codec_bench.py', 'inputs_helper': 'tools/reference/codec_trace.py', 'stages': 'tools/reference/codec_stages.py'}
                            source = aliases[source]
                        if source.startswith('/') or '..' in Path(source).parts: raise ValueError('bad source path')
                        key=(revision,source)
                        if key not in cache:
                            cache[key]=sha(subprocess.check_output(['git','show',f'{revision}:{source}'],cwd=ROOT))
                        if cache[key]!=digest: raise ValueError('source mismatch: '+str(path)+' '+source)
                    counts[role]+=1
                    if role=='python':
                        for dest in ('previous','candidate'):
                            copy=base/dest/path.name
                            if copy.read_bytes()!=data: raise ValueError('Python copy mismatch')
                            if phase=='matrix' and sha(data)!=audit['input_sha256'][str(copy.relative_to(ROOT))]:
                                raise ValueError('matrix copy not bound by audit')
                            copies+=1
    if counts != {role:60 for role in ROLES} or copies != 120:
        raise ValueError('wrong inventory totals')
    if environments[0]['binary_sha256'] != environments[1]['binary_sha256']:
        raise ValueError('matrix/control binary mismatch')
    for file in ('crates/ncm-backend-cuda/src/fused_projection.rs', 'crates/ncm-backend-cuda/src/fused_projection/bounded.rs'):
        if sha((ROOT/file).read_bytes()) != sha(subprocess.check_output(['git','show',f'f155794:{file}'],cwd=ROOT)):
            raise ValueError('supplemental CUDA source differs')
    print(json.dumps({'reports':counts,'python_copies':copies,'source_pairs':len(cache),'supplemental_cuda_sources':2,
                      'audit_sha256':EXPECTED_AUDIT,'mismatches':0},indent=2))
if __name__=='__main__': main()
