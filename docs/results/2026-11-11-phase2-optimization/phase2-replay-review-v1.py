"""Compare clean-export audit replays with the retained platform audits."""
import json
import math
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
base = ROOT / 'docs/results/2026-11-11-phase2-optimization'
linux, apple_initial, apple_controlled = map(Path, sys.argv[1:])
assert linux.read_bytes() == (base / 'linux-acceptance.json').read_bytes()
assert json.loads(linux.read_bytes())['all_passed'] is True
with tarfile.open(base / 'raw/apple-gate18-resource-memory-v1.tar.xz', 'r:xz') as tar:
    original_initial = tar.extractfile('reports/phase2-mlx-gate18-initial-audit-v1.json').read()
assert apple_initial.read_bytes() == original_initial
assert json.loads(original_initial)['all_passed'] is False
original = json.loads((base / 'mlx-acceptance.json').read_bytes())
replayed = json.loads(apple_controlled.read_bytes())
assert original['all_passed'] is True and replayed['all_passed'] is True
differences = []
def compare(a, b, path=''):
    if type(a) is not type(b):
        raise ValueError('type differs: ' + path)
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError('keys differ: ' + path)
        for key in a:
            compare(a[key], b[key], path + '/' + str(key))
    elif isinstance(a, list):
        if len(a) != len(b):
            raise ValueError('length differs: ' + path)
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, path + '/' + str(i))
    elif a != b:
        if not (path.endswith('/stddev_ms') and isinstance(a, float)
                and math.isclose(a, b, rel_tol=1e-14, abs_tol=0.0)):
            raise ValueError('material difference: ' + path)
        differences.append({'path': path, 'original': a, 'replayed': b})
compare(original, replayed)
print(json.dumps({'linux_byte_identical': True, 'apple_initial_byte_identical': True,
                  'apple_controlled_decisions_identical': True,
                  'apple_controlled_last_digit_stddev_differences': differences,
                  'scope': 'cross-environment audit replay; not new hardware evidence'}, indent=2))
