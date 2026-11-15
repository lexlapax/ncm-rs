"""Render descriptive platform tables from the published controlled audits."""
import hashlib
import json
import math
import statistics
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2] / 'tools/bench'))
from run_codec_lifecycle import validate as validate_lifecycle
WORK = ('codec', 'bert', 'gpt2', 'qwen3', 'tinystories')
ARCHIVES = {
    'Linux/CUDA': ('cuda-bounded-linux-acceptance-v1.json', '5dee0d928c5825b58c23befa61d88d61cf0a072e5ce0890736e3579e8104ab57', 'raw/cuda-bounded-linux-gate-v1.tar.xz',
                   'a2c3d9bcb754d7b511b8ce11812b56e3389565277edcbee8c56cc7ce181044d8',
                   'reports/phase2-cuda-bounded-linux-control-v1'),
    'Apple/MLX': ('mlx-acceptance.json', 'a3ed81700b05f39a50a29a44340ffb00fb826fa738539989eb41e78556d58e4e', 'raw/apple-gate18-resource-memory-v1.tar.xz',
                  '524293f724bb1c047af88d481db476056f365b0a01aa3af50ddd5bd3a27348b6',
                  'reports/phase2-mlx-gate18-control-v1'),
}

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def main():
    print('# Controlled Phase 2 performance summary\n')
    print('These tables describe the retained controlled audits, not a new run. '
          'The [current source and evidence binding](phase2-current-bindings-v2.md) '
          'checks all 78 cells per platform. The current Linux native CUDA pool '
          'is below 1.05× prior Rust; Apple immediate live-memory comparability '
          'remains unresolved.\n')
    print('Timings are pooled controlled process trials in milliseconds; each '
          'cell has 30 trials per process (nine processes per runtime on Linux, '
          '18 on Apple). Rate is calculated from the candidate median: codec batch '
          'items/s or tokenizer fixture strings/s. Checked-load rows have no rate. '
          'This is not generated-token throughput. Confidence bounds and every '
          'original result remain in the linked audit JSON.\n')
    for platform, (audit_name, audit_digest, archive_name, digest, control) in ARCHIVES.items():
        audit_bytes = (ROOT / audit_name).read_bytes()
        if sha(audit_bytes) != audit_digest:
            raise ValueError('published audit differs')
        audit = json.loads(audit_bytes)
        if audit['all_passed'] is not True or audit['quality_passed'] is not True:
            raise ValueError('controlled audit did not pass')
        archive = ROOT / archive_name
        if sha(archive.read_bytes()) != digest:
            raise ValueError('archive differs')
        with tarfile.open(archive, 'r:xz') as tar:
            life_path = ('reports/phase2-cuda-bounded-lifecycle-v1.json' if platform == 'Linux/CUDA'
                         else 'reports/phase2-tiny-borrow-mlx-lifecycle-v1.json')
            life_raw = tar.extractfile(life_path).read()
            if sha(life_raw) != audit['input_sha256'][life_path]:
                raise ValueError('lifecycle evidence differs')
            lifecycle = json.loads(life_raw)
            runs = lifecycle['runs']
            if len(runs) != 9 or {(r['runtime'], r['repeat']) for r in runs} != {(role, i) for role in ('previous', 'candidate', 'python') for i in (1, 2, 3)}:
                raise ValueError('lifecycle inventory differs')
            if len({validate_lifecycle(run['phases']) for run in runs}) != 1:
                raise ValueError('lifecycle output identities differ')
            strings = {}
            startup = {profile: {role: [] for role in ('previous', 'candidate', 'python')} for profile in WORK}
            first_execution = {role: [] for role in ('previous', 'candidate', 'python')}
            for profile in WORK:
                repeats = audit['controls'][profile]['processes_per_runtime']
                for role in ('previous', 'candidate', 'python'):
                    kind = 'python' if role == 'python' else 'rust'
                    stem = 'codec' if profile == 'codec' else 'tokenizer-' + profile
                    hashes = audit['controls'][profile]['report_sha256'][role]
                    if len(hashes) != repeats:
                        raise ValueError('incomplete startup inventory')
                    for i, digest in enumerate(hashes, 1):
                        name = f'{control}/{role}/{stem}-{kind}-{i}.json'
                        raw = tar.extractfile(name).read()
                        if sha(raw) != digest:
                            raise ValueError('startup report differs')
                        report = json.loads(raw)
                        first_load = report['first_load_ms']
                        if not isinstance(first_load, (float, int)) or isinstance(first_load, bool) or not math.isfinite(first_load) or first_load <= 0:
                            raise ValueError('invalid first load')
                        startup[profile][role].append(first_load)
                        if profile == 'codec':
                            first = report['first_execution_ms']
                            if not isinstance(first, (float, int)) or isinstance(first, bool) or not math.isfinite(first) or first <= 0:
                                raise ValueError('invalid first execution')
                            first_execution[role].append(first)
                        else:
                            count = report['strings_per_trial']
                            if not isinstance(count, int) or isinstance(count, bool) or count <= 0 or profile in strings and strings[profile] != count:
                                raise ValueError('invalid or differing string count')
                            strings[profile] = count
        print(f'## {platform}\n')
        print(f'[Complete controlled audit]({audit_name}).\n')
        print('| Workload | Candidate median / p95 ms | Candidate rate | Previous median / p95 ms | Python median / p95 ms |')
        print('| --- | ---: | ---: | ---: | ---: |')
        for workload in WORK:
            rows = audit['controls'][workload]['measurements']
            labels = [(row.get('batch'), row['scope']) for row in rows]
            expected = ({(batch, scope) for batch in (1, 2, 4) for scope in ('encode', 'decode', 'reconstruct', 'resident_decoder_block')} | {(None, 'checked_load')}) if workload == 'codec' else {(None, scope) for scope in ('encode', 'decode', 'decode_skip_special', 'checked_load')}
            if len(labels) != len(expected) or set(labels) != expected:
                raise ValueError('control scope inventory differs: ' + workload)
            for measurement in rows:
                scope = measurement['scope']; batch = measurement.get('batch')
                label = f'{workload}/{batch}/{scope}' if batch is not None else f'{workload}/{scope}'
                timings = measurement['timings']
                values = [timings[role] for role in ('candidate', 'previous', 'python')]
                if any(v['median_ms'] <= 0 or v['p95_ms'] <= 0 for v in values):
                    raise ValueError('invalid timing')
                fmt = lambda v: f"{v['median_ms']:.4f} / {v['p95_ms']:.4f}"
                if scope == 'checked_load':
                    rate = '—'
                elif batch is not None:
                    rate = f"{1000 * batch / values[0]['median_ms']:.1f} items/s"
                else:
                    rate = f"{1000 * strings[workload] / values[0]['median_ms']:.1f} strings/s"
                print(f'| {label} | {fmt(values[0])} | {rate} | {fmt(values[1])} | {fmt(values[2])} |')
        print('\nFirst-load and first-execution values below summarize all controlled '
              'processes as median [minimum, maximum] in milliseconds. Process, '
              'OS disk and compiler caches were not cleared; process-local allocator '
              'handling follows the benchmark harness. First execution starts '
              'after loading and is not a disk-cold or compilation-only measure.\n')
        print('| Observation | Previous | Candidate | Python |')
        print('| --- | ---: | ---: | ---: |')
        def range_ms(values):
            return f'{statistics.median(values):.3f} [{min(values):.3f}, {max(values):.3f}]'
        for workload in WORK:
            values = startup[workload]
            print(f"| {workload} first load | " + ' | '.join(range_ms(values[role]) for role in ('previous', 'candidate', 'python')) + ' |')
        print('| codec first execution | ' + ' | '.join(range_ms(first_execution[role]) for role in ('previous', 'candidate', 'python')) + ' |')
        print('\nSeparate lifecycle capture: three fresh processes per runtime; '
              'median [minimum, maximum] milliseconds. Parent startup includes '
              'process creation and initialization; parent exit is measured after '
              'the drop phase. These are not the controlled warm timings.\n')
        print('| Lifecycle observation | Previous | Candidate | Python |')
        print('| --- | ---: | ---: | ---: |')
        for label, key in (('parent startup', 'startup_wall_ms'), ('child load', 'load'),
                           ('child execution', 'execute'), ('child drop', 'drop'),
                           ('parent post-drop exit', 'post_drop_exit_ms')):
            groups = {role: [] for role in ('previous', 'candidate', 'python')}
            for run in runs:
                value = run[key] if key in ('startup_wall_ms', 'post_drop_exit_ms') else next(p['internal_ms'] for p in run['phases'] if p['phase'] == key)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                    raise ValueError('invalid lifecycle timing')
                groups[run['runtime']].append(value)
            print('| ' + label + ' | ' + ' | '.join(range_ms(groups[role]) for role in ('previous', 'candidate', 'python')) + ' |')
        print('\nHost memory is the maximum observed process HWM/RSS ratio from the '
              'three-process matrix, not the MLX allocator counter and not a '
              'separate physical-memory estimate.\n')
        lookup = {(c['cell'], c['comparator']): c for c in audit['cells']}
        print('| Workload | HWM / previous | HWM / Python | RSS / previous | RSS / Python |')
        print('| --- | ---: | ---: | ---: | ---: |')
        for workload in WORK:
            ratios = [lookup[(f'{workload}/{metric}', role)]['peak_observed_ratio']
                      for metric in ('VmHWM', 'VmRSS') for role in ('previous', 'python')]
            print('| ' + workload + ' | ' + ' | '.join(f'{x:.4f}' for x in ratios) + ' |')
        print()
    print('Reproduce with `python3 docs/results/2026-11-11-phase2-optimization/phase2-performance-summary-v1.py`. '
          'The two controlled audit files and referenced archives are inputs; '
          'the script neither collects new measurements nor adjudicates live memory.')

if __name__ == '__main__':
    main()
