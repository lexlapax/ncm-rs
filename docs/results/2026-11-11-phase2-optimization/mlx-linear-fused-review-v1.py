"""Verify and summarize the archived isolated MLX interpolation experiment."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1]) / 'reports/phase2-mlx-linear-fused-poc-v1'
meta = json.loads((root / 'run-metadata.json').read_bytes())
assert meta['status'] == 'complete' and not meta['activity_errors']
assert {(r['mode'], r['repeat']) for r in meta['runs']} == {(mode, i) for mode in ('gather', 'fused') for i in (1, 2, 3)}
data = {'gather': [], 'fused': []}
for run in meta['runs']:
    path = root / f"{run['mode']}-{run['repeat']}.json"
    assert run['exit_code'] == 0 and hashlib.sha256(path.read_bytes()).hexdigest() == run['report_sha256']
    report = json.loads(path.read_bytes())
    assert report['source_sha256'] == meta['source_sha256'] and report['mode'] == run['mode']
    assert len(report['cases']) == 10
    assert all(c['bit_differences'] == 0 and c['max_abs_difference'] == 0 and len(c['trials_ms']) == 30 for c in report['cases'])
    data[run['mode']].append(report)
for i in range(10):
    groups = {role: [r['cases'][i] for r in runs] for role, runs in data.items()}
    shapes = {(c['batch'], c['source'], c['length'], c['channels']) for rows in groups.values() for c in rows}
    assert len(shapes) == 1
    times = {role: statistics.median(statistics.median(c['trials_ms']) for c in cases) for role, cases in groups.items()}
    peaks = {role: max(c['peak_active_bytes'] for c in cases) for role, cases in groups.items()}
    print(next(iter(shapes)), 'fused/gather median:', round(times['fused'] / times['gather'], 4), 'peak bytes:', peaks)
