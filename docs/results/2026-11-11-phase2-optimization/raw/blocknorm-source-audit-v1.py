"""Verify retained native and Python report inventories against clean exports.

Run from the repository root with the two gate archives extracted there; pass
clean git-archive directories for 3307dff, 621c565 and 884d8f6 in that order.
Python comparator reports appear in three directories but are shared copies.
"""
import collections
import hashlib
import json
from pathlib import Path
import sys

if len(sys.argv) != 4:
    raise SystemExit('usage: python3 blocknorm-source-audit-v1.py BASELINE LINUX APPLE')
roots = dict(zip(('baseline', 'linux', 'apple'), map(Path, sys.argv[1:])))
archives = (
    ('linux-matrix', Path('reports/phase2-blocknorm-linux-threeway-v1'), 'linux', 30, 45),
    ('linux-controls', Path('reports/phase2-blocknorm-linux-controls-v1'), 'linux', 72, 108),
    ('apple-matrix', Path('reports/phase2-blocknorm-apple-threeway-v1'), 'apple', 30, 45),
    ('apple-controls', Path('reports/phase2-blocknorm-apple-controls-v1'), 'apple', 18, 27),
)
aliases = {'benchmark': 'tools/reference/codec_bench.py',
           'inputs_helper': 'tools/reference/codec_trace.py',
           'stages': 'tools/reference/codec_stages.py'}
expected_heads = {'baseline': '3307dff82a08f7251b64de0866a01d131528b74f',
                  'linux': '621c56525b137b6f27f09df66ecfbc8771a3344d',
                  'apple': '884d8f6326669a11240e78da73b97756051165ec'}
counts = collections.Counter()
failures = []
for label, directory, candidate, native_count, python_count in archives:
    env = json.loads((directory / 'environment.json').read_bytes())
    if env['source_heads'] != {'previous': expected_heads['baseline'],
                               'candidate': expected_heads[candidate]}:
        failures.append((label, 'source_heads'))
    for runtime in ('previous', 'candidate', 'python'):
        for path in sorted((directory / runtime).glob('*.json')):
            report = json.loads(path.read_bytes())
            inventory = report.get('source_sha256')
            if not isinstance(inventory, dict) or not inventory:
                failures.append((str(path), 'missing inventory'))
                continue
            is_python = '-python-' in path.name
            root = roots[candidate if is_python or runtime != 'previous' else 'baseline']
            for name, expected in inventory.items():
                source = root / aliases.get(name, name)
                if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                    failures.append((str(path), name))
            counts[(label, 'python' if is_python else 'native')] += 1
    for kind, expected in (('native', native_count), ('python', python_count)):
        if counts[(label, kind)] != expected:
            failures.append((label, f'{kind} count'))
    for path in (directory / 'python').glob('*-python-*.json'):
        digest = hashlib.sha256(path.read_bytes()).digest()
        for runtime in ('previous', 'candidate'):
            copy = directory / runtime / path.name
            if not copy.is_file() or hashlib.sha256(copy.read_bytes()).digest() != digest:
                failures.append((str(copy), 'shared Python copy mismatch'))
print(json.dumps({'counts': {f'{a}/{b}': n for (a, b), n in sorted(counts.items())},
                  'mismatches': failures, 'python_reports_are_shared_copies': True}, indent=2))
if failures:
    raise SystemExit(1)
