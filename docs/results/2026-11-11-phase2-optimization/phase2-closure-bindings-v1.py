"""Verify current candidate source and archived 78-cell audit identities.

This is a scope/binding check only. It does not adjudicate live memory or close
Phase 2. It requires the repository's existing Git history and raw archives.
"""
import hashlib
import json
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
        with tarfile.open(path, 'r:xz') as stream:
            raw = stream.extractfile(member).read()
        if sha(raw) != audit_sha:
            raise ValueError('audit hash mismatch: ' + name)
        audit = json.loads(raw)
        if (audit['all_passed'] is not True or audit['quality_passed'] is not True
                or len(audit['cells']) != 78 or not all(c['passed'] is True for c in audit['cells'])
                or audit['limits'] != {'median': 1.05, 'p95': 1.10, 'comparable_host_memory': 1.05}
                or audit['audit_source_sha256'] != audit_source):
            raise ValueError('audit is incomplete, uses different criteria or source: ' + name)
        results[name] = {'matrix_audit_sha256': audit_sha, 'cells_passed': 78, 'quality_passed': True}
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    print(json.dumps({'schema_version': 1, 'scope': 'current source and historical audit binding; not Phase 2 acceptance',
                      'head': head, 'gated_candidate': CANDIDATE, 'baseline': BASELINE,
                      'manifest_sha256': expected_manifest, 'audits': results,
                      'separate_live_memory': 'unresolved', 'phase_2_complete': False}, indent=2))

if __name__ == '__main__':
    main()
