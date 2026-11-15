"""Verify archived Apple gate report identities against clean Git sources."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / 'docs/results/2026-11-11-phase2-optimization/raw/apple-gate18-resource-memory-v1.tar.xz'
ARCHIVE_SHA = '524293f724bb1c047af88d481db476056f365b0a01aa3af50ddd5bd3a27348b6'
AUDIT_SHA = 'a3ed81700b05f39a50a29a44340ffb00fb826fa738539989eb41e78556d58e4e'
WORK = ('codec', 'bert', 'gpt2', 'qwen3', 'tinystories')
ROLES = ('previous', 'candidate', 'python')

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    if sha(ARCHIVE.read_bytes()) != ARCHIVE_SHA:
        raise ValueError('archive differs')
    cache = {}
    count = {role: 0 for role in ROLES}
    copies = 0
    with tarfile.open(ARCHIVE, 'r:xz') as stream:
        def read(name):
            item = stream.getmember(name)
            if not item.isfile():
                raise ValueError('expected regular archive member: ' + name)
            return stream.extractfile(item).read()
        audit_raw = read('reports/phase2-mlx-gate18-controlled-audit-v1.json')
        if sha(audit_raw) != AUDIT_SHA:
            raise ValueError('audit differs')
        audit = json.loads(audit_raw)
        for phase, repeats in (('matrix', 3), ('control', 18)):
            base = f'reports/phase2-mlx-gate18-{phase}-v1'
            for work in WORK:
                stem = 'codec' if work == 'codec' else 'tokenizer-' + work
                for role in ROLES:
                    typ = 'python' if role == 'python' else 'rust'
                    revision = '3307dff' if role == 'previous' else '5200924'
                    for i in range(1, repeats + 1):
                        member = f'{base}/{role}/{stem}-{typ}-{i}.json'
                        raw = read(member)
                        if phase == 'control':
                            expected = audit['controls'][work]['report_sha256'][role][i - 1]
                            if sha(raw) != expected:
                                raise ValueError('controlled report differs: ' + member)
                        elif role != 'python' and sha(raw) != audit['input_sha256'][member]:
                            raise ValueError('matrix report differs: ' + member)
                        report = json.loads(raw)
                        inventory = report.get('source_sha256')
                        if not isinstance(inventory, dict) or not inventory:
                            raise ValueError('source inventory missing: ' + member)
                        for path, digest in inventory.items():
                            if path.startswith('/') or '..' in Path(path).parts:
                                raise ValueError('invalid source path')
                            key = (revision, path)
                            if key not in cache:
                                blob = subprocess.check_output(['git', 'show', f'{revision}:{path}'], cwd=ROOT)
                                cache[key] = sha(blob)
                            if cache[key] != digest:
                                raise ValueError('source inventory differs: ' + member + ' ' + path)
                        count[role] += 1
                        if role == 'python':
                            for dest in ('previous', 'candidate'):
                                copy = f'{base}/{dest}/{stem}-python-{i}.json'
                                if raw != read(copy):
                                    raise ValueError('Python report copy differs: ' + copy)
                                if phase == 'matrix' and sha(raw) != audit['input_sha256'][copy]:
                                    raise ValueError('matrix Python copy is not audit-bound: ' + copy)
                                copies += 1
    if count != {role: 105 for role in ROLES} or copies != 210:
        raise ValueError('incomplete report inventory')
    print(json.dumps({'scope': 'archived report/source binding, not fresh execution or live-memory acceptance',
                      'reports': count, 'python_copies': copies, 'git_source_pairs': len(cache),
                      'mismatches': 0, 'archive_sha256': ARCHIVE_SHA, 'audit_sha256': AUDIT_SHA}, indent=2))

if __name__ == '__main__':
    main()
