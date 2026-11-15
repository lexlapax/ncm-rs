"""Verify current candidate source and archived 78-cell audit identities.

This is a scope/binding check only. It does not adjudicate live memory or close
Phase 2. It requires the repository's existing Git history and raw archives.
"""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[3]
CANDIDATE = '5200924'
BASELINE = '3307dff'
GUARDED = ['crates', 'vendor', 'Cargo.lock', 'upstream.toml', 'tools/reference', 'tests/fixtures',
           'tools/bench/audit_phase2_performance.py', 'tools/bench/phase2_summary.py',
           'tools/bench/phase2_mlx_summary.py', 'tools/bench/run_phase2_linux_comparison.py',
           'tools/bench/run_phase2_mlx_comparison.py']
BASELINE_PINS = ['upstream.toml', 'tests/fixtures',
                 'docs/results/2026-11-10-phase2-oracle/raw/sudoku-cuda.json',
                 'docs/results/2026-11-10-sudoku-artifacts/raw/artifact-manifest.json']
EVIDENCE = ROOT / 'docs/results/2026-11-11-phase2-optimization/raw'
AUDITS = {
    'linux': ('tiny-borrow-linux-gate-v1.tar.xz',
              'b16be8834fa12e30b5c70d9ac4073461d9caf1f7ab5f2fb0673b3ce2fe1aa87e',
              'reports/phase2-tiny-borrow-linux-controlled-audit-v1.json',
              '50960a48e6e026b377b547fcc3df39096a5408ed27c576114d3d79a9b52500e3'),
    'apple': ('apple-gate18-resource-memory-v1.tar.xz',
              '524293f724bb1c047af88d481db476056f365b0a01aa3af50ddd5bd3a27348b6',
              'reports/phase2-mlx-gate18-controlled-audit-v1.json',
              'a3ed81700b05f39a50a29a44340ffb00fb826fa738539989eb41e78556d58e4e'),
}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def changed(*args):
    return subprocess.check_output(['git', 'diff', '--name-only', *args], cwd=ROOT, text=True).splitlines()

def verify_quality(quality, runtime):
    policy = {'primitive': {'atol': 1e-5, 'rtol': 1e-4},
              'component': {'atol': 1e-4, 'rtol': 2e-4},
              'distance': {'atol': 1e-3, 'rtol': 2e-4},
              'accumulated': {'atol': 5e-4, 'rtol': 2e-4}}
    if (quality['runtime'] != runtime or quality['all_passed'] is not True
            or quality['all_trace_elements_pass'] is not True or quality['failures'] != []
            or quality['policy'] != policy
            or quality['left_trace_sha256'] != '89bee3830bf97485183a17d325f78183bcc6d5d573b2af312899b2ee9dfad7fa'):
        raise ValueError('quality policy, reference or status differs')
    traces = quality['traces']
    if (quality['trace_count'] != 642 or len(traces) != 642
            or len({row['name'] for row in traces}) != 642
            or any(row['elements'] <= 0 or row['mismatches'] != 0 for row in traces)):
        raise ValueError('quality trace inventory differs or contains failures')
    selections = quality['selections'].values()
    if (sum(row['decisions'] for row in selections) != 2500
            or sum(row['unstable'] for row in quality['selections'].values()) != 2
            or any(row['disagreements'] != 0 or row['stable_disagreements'] != 0
                   or row['candidate_selection_mismatches'] != 0 for row in quality['selections'].values())):
        raise ValueError('selection scope or decisions differ')

def verify_tokenizer_probe(probe):
    profiles = ('bert', 'gpt2', 'qwen3', 'tinystories')
    fixture = ROOT / 'tests/fixtures/tokenizers-v1.json'
    lock = json.loads((ROOT / 'tools/reference/tokenizers-v1.lock.json').read_bytes())
    if (probe['schema_version'] != 1 or probe['runtime'] != 'rust-tokenizers'
            or probe['fixture_sha256'] != sha(fixture.read_bytes())
            or probe['reference_versions'] != {'regex': '2025.9.18', 'tokenizers': '0.22.2', 'transformers': '4.57.6'}
            or set(probe['profiles']) != set(profiles)):
        raise ValueError('tokenizer fixture, version or inventory differs')
    for name in profiles:
        row = probe['profiles'][name]
        if (row['all_passed'] is not True or row['both_skip_modes'] is not True
                or row['cleanup'] is not False or row['encode_cases'] != 89
                or row['decode_cases_including_encoded'] < 89 or row['rejection_cases'] != 4
                or row['manifest_sha256'] != lock['profiles'][name]['manifest_sha256']):
            raise ValueError('tokenizer profile incomplete or differs: ' + name)
    sources = probe['source_sha256']
    if not isinstance(sources, dict) or not sources:
        raise ValueError('tokenizer source inventory missing')
    for path, digest in sources.items():
        if path.startswith('/') or '..' in Path(path).parts:
            raise ValueError('invalid tokenizer source path')
        source = subprocess.check_output(['git', 'show', f'{CANDIDATE}:{path}'], cwd=ROOT)
        if sha(source) != digest:
            raise ValueError('tokenizer source differs: ' + path)

def verify_coverage(audit, name):
    workloads = ('codec', 'bert', 'gpt2', 'qwen3', 'tinystories')
    codec = {f'codec/{batch}/{scope}' for batch in (1, 2, 4)
             for scope in ('encode', 'decode', 'reconstruct', 'resident_decoder_block')}
    tokenizer = {f'{profile}/{scope}' for profile in workloads[1:]
                 for scope in ('encode', 'decode', 'decode_skip_special')}
    timing = codec | tokenizer | {f'{work}/checked_load' for work in workloads}
    memory = {f'{work}/{metric}' for work in workloads for metric in ('VmHWM', 'VmRSS')}
    expected = {(cell, comparator) for cell in timing | memory for comparator in ('previous', 'python')}
    actual = [(c['cell'], c['comparator']) for c in audit['cells']]
    if len(expected) != 78 or len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('incomplete or duplicate matrix cells: ' + name)
    expected_repeats = 9 if name == 'linux' else 18
    if set(audit['controls']) != set(workloads) or any(audit['controls'][w]['processes_per_runtime'] != expected_repeats for w in workloads):
        raise ValueError('control workloads or repeat counts differ: ' + name)
    for cell in audit['cells']:
        if cell['cell'] in memory:
            ratio = cell['peak_observed_ratio']
            valid = isinstance(ratio, (float, int)) and not isinstance(ratio, bool) and math.isfinite(ratio) and 0 < ratio <= 1.05
        else:
            follow = cell['controlled_followup']
            interval = follow['paired_process_percentile_95_interval']
            values = (follow['median_ratio'], follow['p95_ratio'], interval['median_ratio'][1], interval['p95_ratio'][1])
            valid = all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) and v > 0 for v in values)
            valid = valid and values[0] <= 1.05 and values[1] <= 1.10 and values[2] <= 1.05 and values[3] <= 1.10
        if not valid:
            raise ValueError('cell does not independently satisfy its bound: ' + name + ' ' + cell['cell'])
    return expected_repeats

def main():
    if changed(CANDIDATE, 'HEAD', '--', *GUARDED):
        raise ValueError('committed guarded sources differ from the gated candidate')
    if changed('HEAD', '--', *GUARDED):
        raise ValueError('tracked guarded sources are modified or staged')
    if changed(BASELINE, 'HEAD', '--', *BASELINE_PINS):
        raise ValueError('committed baseline pins, oracle or fixtures differ')
    if changed('HEAD', '--', *BASELINE_PINS):
        raise ValueError('tracked baseline pins, oracle or fixtures are modified or staged')
    expected_manifest = 'ad6a99ad7d5ccc069e0fedf26a85f755e06cd76590463e1e04e17363e35d9832'
    for p in ('artifacts/canonical/sudoku-easy-qualified/manifest.json',
              'docs/results/2026-11-10-sudoku-artifacts/raw/artifact-manifest.json'):
        if sha((ROOT / p).read_bytes()) != expected_manifest:
            raise ValueError('artifact manifest mismatch: ' + p)
    audit_source = sha((ROOT / 'tools/bench/audit_phase2_performance.py').read_bytes())
    results = {}
    for name, (archive, archive_sha, member, audit_sha) in AUDITS.items():
        path = EVIDENCE / archive
        if sha(path.read_bytes()) != archive_sha:
            raise ValueError('archive hash mismatch: ' + archive)
        quality_member = ('reports/phase2-fused-integration-qualification-v1.json' if name == 'linux'
                          else '.reference/phase2-mlx-gate18-preflight-v1/qualification.json')
        with tarfile.open(path, 'r:xz') as stream:
            raw = stream.extractfile(member).read()
            quality_raw = stream.extractfile(quality_member).read()
        if sha(raw) != audit_sha:
            raise ValueError('audit hash mismatch: ' + name)
        published = ROOT / 'docs/results/2026-11-11-phase2-optimization' / ('linux-acceptance.json' if name == 'linux' else 'mlx-acceptance.json')
        if sha(published.read_bytes()) != audit_sha:
            raise ValueError('published matrix audit differs: ' + name)
        audit = json.loads(raw)
        if (audit['all_passed'] is not True or audit['quality_passed'] is not True
                or len(audit['cells']) != 78 or not all(c['passed'] is True for c in audit['cells'])
                or audit['limits'] != {'median': 1.05, 'p95': 1.10, 'comparable_host_memory': 1.05}
                or audit['audit_source_sha256'] != audit_source):
            raise ValueError('audit is incomplete, uses different criteria or source: ' + name)
        if audit['input_sha256'][quality_member] != sha(quality_raw):
            raise ValueError('quality evidence differs from audited input: ' + name)
        verify_quality(json.loads(quality_raw), 'rust-candle' if name == 'linux' else 'rust-mlx')
        expected_repeats = verify_coverage(audit, name)
        results[name] = {'matrix_audit_sha256': audit_sha, 'cells_passed': 78, 'timing_cells': 58,
                         'host_memory_cells': 20, 'control_processes_per_runtime': expected_repeats,
                         'quality_passed': True, 'qualified_traces': 642,
                         'selection_decisions': 2500, 'near_ties': 2}
    probes = {
        'linux': ('tiny-borrow-linux-gate-v1.tar.xz',
                  'b16be8834fa12e30b5c70d9ac4073461d9caf1f7ab5f2fb0673b3ce2fe1aa87e',
                  'reports/phase2-tiny-borrow-tokenizer-probe-v1.json'),
        'apple': ('tiny-borrow-apple-gate-v1.tar.xz',
                  '3448f9bd9510bce353970802b57fbd42023855e4779798d4f58d08de4a3031d9',
                  'reports/tokenizer-probe.json'),
    }
    for name, (archive, digest, member) in probes.items():
        path = EVIDENCE / archive
        if sha(path.read_bytes()) != digest:
            raise ValueError('tokenizer probe archive differs: ' + name)
        with tarfile.open(path, 'r:xz') as stream:
            raw = stream.extractfile(member).read()
        if sha(raw) != '6124d2dbd456dce9375e5d6bb17472875ae75674114eb51cd4f279046b08bee9':
            raise ValueError('tokenizer probe differs: ' + name)
        verify_tokenizer_probe(json.loads(raw))
        results[name]['tokenizer_profiles'] = 4
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    print(json.dumps({'schema_version': 1, 'scope': 'current source and historical audit binding; not Phase 2 acceptance',
                      'head': head, 'gated_candidate': CANDIDATE, 'baseline': BASELINE,
                      'manifest_sha256': expected_manifest, 'audits': results,
                      'separate_live_memory': 'unresolved', 'phase_2_complete': False}, indent=2))

if __name__ == '__main__':
    main()
