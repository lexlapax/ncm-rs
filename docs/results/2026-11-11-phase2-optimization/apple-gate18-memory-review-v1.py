"""Verify matched diagnostic reports in an extracted evidence archive."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
p = root / 'reports/phase2-mlx-live-memory-v1'
meta = json.loads((p / 'metadata.json').read_bytes())
assert meta['status'] == 'complete'
audit_path = root / 'reports/phase2-mlx-gate18-controlled-audit-v1.json'
assert hashlib.sha256(audit_path.read_bytes()).hexdigest() == meta['audit_sha256']
assert json.loads(audit_path.read_bytes())['all_passed'] is True
assert meta['manifest_sha256'] == 'ad6a99ad7d5ccc069e0fedf26a85f755e06cd76590463e1e04e17363e35d9832'
roles = ('previous', 'candidate', 'python')
reports = {role: [] for role in roles}
assert {(x['role'], x['repeat']) for x in meta['commands']} == {(r, i) for r in roles for i in (1, 2, 3)}
for x in meta['commands']:
    path = p / f"{x['role']}-{x['repeat']}.json"
    assert x['returncode'] == 0 and hashlib.sha256(path.read_bytes()).hexdigest() == x['report_sha256']
    report = json.loads(path.read_bytes())
    assert report['manifest_sha256'] == meta['manifest_sha256']
    assert len(report['rows']) == 12 and report['warmups_and_trials'] == 35
    assert report['runtime'] == ('python-mlx-adapted' if x['role'] == 'python' else 'rust-mlx')
    reports[x['role']].append(report)
assert all(report['after_model_drop']['active'] == 0 for group in reports.values() for report in group)
gate = json.loads((root / 'reports/phase2-mlx-gate18-matrix-v1/candidate/codec-rust-1.json').read_bytes())
violations = []
for batch in (1, 2, 4):
    for scope in ('encode', 'decode', 'reconstruct', 'resident_decoder_block'):
        rows = {role: [next(x for x in report['rows'] if (x['batch'], x['scope']) == (batch, scope)) for report in reports[role]] for role in roles}
        for key in ('input_ids_sha256', 'supplied_codes_sha256', 'output_ids_sha256', 'latent_sha256'):
            assert len({row[key] for group in rows.values() for row in group}) == 1
        reference = next(x for x in gate['measurements'] if (x['batch'], x['scope']) == (batch, scope))
        for key in ('input_ids_sha256', 'supplied_codes_sha256', 'output_ids_sha256'):
            assert rows['candidate'][0][key] == reference[key]
        for stage in ('immediate', 'synchronized', 'after_10ms', 'after_100ms'):
            values = {role: max(row[stage]['active'] for row in rows[role]) for role in roles}
            for other in ('previous', 'python'):
                ratio = values['candidate'] / values[other]
                if ratio > 1.05:
                    violations.append({'batch': batch, 'scope': scope, 'stage': stage, 'comparator': other, 'ratio': ratio, 'bytes': values})
        if batch == 4 and scope == 'encode':
            peaks = {role: max(row['immediate']['peak'] for row in rows[role]) for role in roles}
print(json.dumps({'all_processes_verified': True, 'matched_root_and_output_hashes': True,
                  'all_post_drop_active_zero': True, 'active_snapshot_over_1_05': violations,
                  'b4_encode_peak_bytes': peaks,
                  'b4_encode_peak_ratio_previous': peaks['candidate'] / peaks['previous'],
                  'b4_encode_peak_ratio_python': peaks['candidate'] / peaks['python'],
                  'steady_state_acceptance': 'unresolved: cross-runtime live graphs and immediate snapshot interpretation'}, indent=2))
