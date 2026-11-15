"""Inspect synchronized retained-root CUDA pool snapshots from the fixed gate."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[3]
base = ROOT / 'docs/results/2026-11-11-phase2-optimization'
archive = base / 'raw/tiny-borrow-linux-gate-v1.tar.xz'
sha = lambda value: hashlib.sha256(value).hexdigest()
if sha(archive.read_bytes()) != 'b16be8834fa12e30b5c70d9ac4073461d9caf1f7ab5f2fb0673b3ce2fe1aa87e':
    raise ValueError('archive hash differs')
audit_raw = (base / 'linux-acceptance.json').read_bytes()
if sha(audit_raw) != '50960a48e6e026b377b547fcc3df39096a5408ed27c576114d3d79a9b52500e3':
    raise ValueError('audit hash differs')
audit = json.loads(audit_raw)
rows = []
with tarfile.open(archive, 'r:xz') as tar:
    for phase, repeats in (('matrix', 3), ('control', 9)):
        directory = f'reports/phase2-tiny-borrow-linux-{phase}-v1'
        records = {}
        for role in ('previous', 'candidate', 'python'):
            kind = 'python' if role == 'python' else 'rust'
            for i in range(1, repeats + 1):
                name = f'{directory}/{role}/codec-{kind}-{i}.json'
                raw = tar.extractfile(name).read()
                if phase == 'control':
                    expected = audit['controls']['codec']['report_sha256'][role][i - 1]
                else:
                    # Matrix Python reports are audit-bound through the candidate copy.
                    key = name if role != 'python' else f'{directory}/candidate/codec-python-{i}.json'
                    expected = audit['input_sha256'][key]
                if sha(raw) != expected:
                    raise ValueError('report hash differs: ' + name)
                report = json.loads(raw)
                indexed = {(r['batch'], r['scope']): r for r in report['measurements']}
                expected_scopes = {(batch, scope) for batch in (1, 2, 4)
                                   for scope in ('encode', 'decode', 'reconstruct', 'resident_decoder_block')}
                if len(indexed) != 12 or set(indexed) != expected_scopes:
                    raise ValueError('scope inventory differs')
                records[role, i] = indexed
        for batch, scope in sorted(expected_scopes):
            grouped = {role: [records[role, i][batch, scope] for i in range(1, repeats + 1)]
                       for role in ('previous', 'candidate', 'python')}
            for key in ('input_ids_sha256', 'supplied_codes_sha256', 'output_ids_sha256', 'output_ids_count', 'output_float_elements'):
                if len({row[key] for group in grouped.values() for row in group}) != 1:
                    raise ValueError('workload/output identity differs: ' + key)
            old = [r['pool_bytes']['used_current'] for r in grouped['previous']]
            new = [r['pool_bytes']['used_current'] for r in grouped['candidate']]
            torch = [r['allocator_bytes']['allocated_current'] for r in grouped['python']]
            if any(type(x) is not int or x <= 0 for x in old + new + torch):
                raise ValueError('invalid allocator counter')
            ratios = [c / p for c in new for p in old]
            deltas = sorted({c - p for c in new for p in old})
            rows.append({'phase': phase, 'batch': batch, 'scope': scope,
                         'native_previous_used_bytes': sorted(set(old)),
                         'native_candidate_used_bytes': sorted(set(new)),
                         'native_delta_bytes': deltas,
                         'native_ratio_min': min(ratios), 'native_ratio_max': max(ratios),
                         'python_torch_allocated_bytes': sorted(set(torch)),
                         'native_ratio_over_1_05': max(ratios) > 1.05})
print(json.dumps({'scope': 'synchronized post-output-drop pool snapshots; no hardware rerun',
                  'rows': rows, 'limit': 1.05,
                  'all_native_pool_snapshots_over_limit': all(x['native_ratio_over_1_05'] for x in rows),
                  'limitations': ['CUDA default-pool used bytes exclude non-pool allocations',
                                  'PyTorch allocator counters have different coverage and are not ratio-compared',
                                  'snapshot roots are not an exhaustive accounting of all device memory',
                                  '78-cell audit uses host RSS/HWM and does not adjudicate this metric']}, indent=2))
