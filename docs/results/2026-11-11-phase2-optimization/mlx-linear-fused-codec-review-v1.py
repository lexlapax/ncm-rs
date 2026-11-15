"""Verify and summarize the isolated paired MLX codec reports."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools/bench'))
from phase2_summary import codec

base = Path(sys.argv[1]) / 'reports'
evidence = base / 'phase2-mlx-linear-fused-codec-v1'
reports = base / 'phase2-mlx-linear-fused-codec-pairs-v1'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
metadata = json.loads((reports / 'metadata.json').read_bytes())
assert metadata['status'] == 'complete' and 'activity_error' not in metadata
assert {(x['name'], x['repeat']) for x in metadata['commands']} == {(r, i) for r in ('current', 'fused') for i in (1, 2, 3)}
assert metadata['qualification_sha256'] == sha(evidence / 'qualification.json')
qual = json.loads((evidence / 'qualification.json').read_bytes())
assert qual['all_passed'] and qual['trace_count'] == 642
assert sum(x['decisions'] for x in qual['selections'].values()) == 2500
assert sum(x['disagreements'] for x in qual['selections'].values()) == 0
assert qual['candidate_report_sha256'] == sha(evidence / 'trace/report.json')
assert metadata['patch_sha256'] == sha(ROOT / 'docs/results/2026-11-11-phase2-optimization/mlx-linear-fused-codec-v1.patch')
data = {'current': [None] * 3, 'fused': [None] * 3}
for x in metadata['commands']:
    path = reports / f"{x['name']}-{x['repeat']}.json"
    assert x['exit_code'] == 0 and sha(path) == x['report_sha256']
    data[x['name']][x['repeat'] - 1] = json.loads(path.read_bytes())
for role in data:
    assert all(r['source_sha256'] == data[role][0]['source_sha256'] for r in data[role])
a = data['current'][0]['source_sha256']; b = data['fused'][0]['source_sha256']
assert set(a) == set(b) and {k for k in a if a[k] != b[k]} == {'crates/ncm-backend-mlx/src/ops.rs'}
assert all(metadata['source_sha256'][role] == data[role][0]['source_sha256']['crates/ncm-backend-mlx/src/ops.rs'] for role in data)
# The existing checker labels its two comparator slots rust/python; both are
# native Rust here. It verifies complete shapes, IDs, logits and timing protocol.
summary = codec(reports, 'metal', runs={'rust': data['current'], 'python': data['fused']})
for batch in (1, 2, 4):
    for scope in ('encode', 'decode', 'reconstruct', 'resident_decoder_block'):
        row = next(x for x in summary['measurements'] if x['batch'] == batch and x['scope'] == scope)
        pair = []
        for i in range(3):
            current = next(x for x in data['current'][i]['measurements'] if x['batch'] == batch and x['scope'] == scope)
            fused = next(x for x in data['fused'][i]['measurements'] if x['batch'] == batch and x['scope'] == scope)
            pair.append(statistics.median(fused['trials_ms']) / statistics.median(current['trials_ms']))
        r = row['runtimes']
        print(batch, scope, 'pooled median/p95 ratios', round(r['python']['wall']['median_ms']/r['rust']['wall']['median_ms'], 4), round(r['python']['wall']['p95_ms']/r['rust']['wall']['p95_ms'], 4), 'paired medians', [round(x, 4) for x in pair])
