"""Independently check the retained Phase 3 Metal quality and benchmark evidence."""
import argparse
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

from phase3_sudoku_score import score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'artifacts/canonical/sudoku-data-phase3'
REF = ROOT / 'reports/phase3-linux-qualified-v1'
REFERENCE_MANIFEST_SHA = '3fa4abca7c28c92085c3a3ac57656f4283433499187d4f5a58545ff4bb1ba82f'
GOLDEN_CODES = '0a9e9b6a4b2ed68e8a31ad1dfe8e1bbad910b39a2452975c059e429ab50ff245'
GOLDEN_IDS = 'a92e42db209774bd3a6b318e0e2f747040de73907098cbda4276ad209cd34a48'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def validate_sources(manifest):
    head = manifest['git_head']
    tracked = set(subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', head], cwd=ROOT, text=True).splitlines())
    prefix = next(name[:-len('/target/release/ncm')] for name in manifest['watched_sha256'] if name.endswith('/target/release/ncm'))
    checked = 0
    for name, expected in manifest['watched_sha256'].items():
        if not name.startswith(prefix + '/'):
            continue
        relative = name[len(prefix) + 1:]
        if relative in tracked:
            raw = subprocess.check_output(['git', 'show', f'{head}:{relative}'], cwd=ROOT)
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f'captured tracked source differs: {relative}')
            checked += 1
        elif relative.startswith(('artifacts/canonical/', 'reports/phase3-linux-qualified-v1/')):
            local = ROOT / relative
            if not local.is_file() or digest(local) != expected:
                raise ValueError(f'captured evaluation input differs: {relative}')
    if checked < 50:
        raise ValueError('insufficient tracked sources were checked')
    return checked


def check_quality(path):
    manifest = load(path / 'manifest.json')
    if manifest['git_status'] or not manifest['stable_inputs_and_sources'] or manifest['telemetry_errors']:
        raise ValueError('quality source or telemetry capture incomplete')
    if digest(REF / 'manifest.json') != REFERENCE_MANIFEST_SHA or manifest['reference_manifest_sha256'] != REFERENCE_MANIFEST_SHA:
        raise ValueError('Linux reference manifest differs')
    for name, expected in manifest['output_sha256'].items():
        if digest(path / name) != expected:
            raise ValueError(f'quality output changed: {name}')
    golden = (path / 'synthetic-golden.stdout').read_text()
    if 'metal_matches_pinned_python_synthetic_golden ... ok' not in golden or '1 passed' not in golden:
        raise ValueError('pinned golden did not run')
    test_path = 'crates/ncm-backend-mlx/tests/generator_golden.rs'
    test_source = subprocess.check_output(['git', 'show', f"{manifest['git_head']}:{test_path}"], cwd=ROOT, text=True)
    if GOLDEN_CODES not in test_source or GOLDEN_IDS not in test_source:
        raise ValueError('pinned golden hashes differ')
    expected_runs = ['synthetic-golden', 'easy-rust', 'medium-rust', 'hard-rust']
    if [r['name'] for r in manifest['runs']] != expected_runs:
        raise ValueError('incomplete quality run sequence')
    result = {}
    for difficulty in ('easy', 'medium', 'hard'):
        reference = load(REF / f'{difficulty}-python.json')
        dataset = load(DATA / f'{difficulty}.json')
        actual = load(path / f'{difficulty}-rust.stdout')
        recorded = load(path / f'{difficulty}-score.json')
        expected_input = (json.dumps(dataset['inputs'], separators=(',', ':')) + '\n').encode()
        if hashlib.sha256(expected_input).hexdigest() != reference['inputs_sha256']:
            raise ValueError('reference input differs')
        if not (len(dataset['inputs']) == len(dataset['references']) == len(actual['codes']) == len(actual['ids'])
                == len(reference['codes']) == len(reference['ids']) == 2000):
            raise ValueError('quality board count differs')
        actual_score = score(dataset['inputs'], dataset['references'], actual['ids'])
        reference_score = score(dataset['inputs'], dataset['references'], reference['ids'])
        code_differences = sum(a != b for a, b in zip(actual['codes'], reference['codes'], strict=True))
        id_differences = sum(a != b for a, b in zip(actual['ids'], reference['ids'], strict=True))
        if (recorded['rust'] != actual_score or recorded['python'] != reference_score
                or recorded['code_mismatched_boards'] != code_differences or recorded['id_mismatched_boards'] != id_differences):
            raise ValueError(f'{difficulty} quality summary differs from raw boards')
        result[difficulty] = {'rust': actual_score, 'python': reference_score,
                              'code_mismatched_boards': code_differences, 'id_mismatched_boards': id_differences}
    return manifest, result


def check_bench(path):
    manifest = load(path / 'manifest.json')
    if (manifest['git_status'] or not manifest['stable_inputs_and_sources'] or manifest['telemetry_errors']
            or manifest['batches'] != [1, 4, 32, 128] or manifest['warmups'] != 5
            or manifest['trials'] != 30 or manifest['repetitions'] != 3 or len(manifest['runs']) != 24):
        raise ValueError('benchmark capture incomplete')
    qualification = load(path / 'qualification.json')
    keys = {f'{runtime}_{field}_matching_rows' for runtime in ('rust', 'python') for field in ('codes', 'ids')}
    if set(qualification) != keys or any(v != 128 for v in qualification.values()):
        raise ValueError('128-board baseline qualification failed')
    recorded = load(path / 'summary.json'); result = {}
    for batch in (1, 4, 32, 128):
        values = {}; hashes = set()
        for runtime in ('rust', 'python'):
            reports = [load(path / f'{runtime}-b{batch}-r{i}.json') for i in range(3)]
            metas = [load(path / f'{runtime}-b{batch}-r{i}.meta.json') for i in range(3)]
            samples = sorted(t for report in reports for t in report['trials'])
            if len(samples) != 90 or any(not math.isfinite(x) or x <= 0 for x in samples):
                raise ValueError('bad timing samples')
            if any(r['batch'] != batch or r['warmups'] != 5 or len(r['trials']) != 30 for r in reports):
                raise ValueError('invalid measurement plan')
            if any(m['returncode'] != 0 or m['peak_rss_unit'] != 'KiB on Linux; bytes on macOS' for m in metas):
                raise ValueError('bad process result or RSS units')
            hashes.update(r['outputs_sha256_i64le'] for r in reports)
            native_memory = [r['mlx_memory'] if runtime == 'rust' else
                             {'peak': r['mlx_peak_bytes'], 'active': r['mlx_active_bytes'], 'cache': r['mlx_cache_bytes']} for r in reports]
            values[runtime] = {'median_ms': statistics.median(samples) * 1000,
                               'p95_ms_nearest_rank': samples[math.ceil(.95 * len(samples)) - 1] * 1000,
                               'boards_per_second_at_median': batch / statistics.median(samples),
                               'process_medians_ms': [statistics.median(r['trials']) * 1000 for r in reports],
                               'peak_rss_bytes': [m['child_peak_rss_platform_units'] for m in metas],
                               'mlx_memory': native_memory}
            if recorded[str(batch)][runtime] != values[runtime]:
                raise ValueError(f'{runtime} batch {batch} benchmark summary differs from raw samples')
        if len(hashes) != 1:
            raise ValueError(f'batch {batch} output hashes differ')
        ratio = values['rust']['median_ms'] / values['python']['median_ms']
        if not math.isclose(recorded[str(batch)]['rust_over_python_median'], ratio, rel_tol=1e-12):
            raise ValueError('benchmark ratio differs')
        values['rust_over_python_median'] = ratio
        values['outputs_sha256_i64le'] = hashes.pop()
        result[str(batch)] = values
    return manifest, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    args = parser.parse_args()
    quality_dir = args.capture / 'phase3-mlx-quality-v1'
    bench_dir = args.capture / 'phase3-mlx-bench-v1'
    quality_manifest, quality = check_quality(quality_dir)
    bench_manifest, bench = check_bench(bench_dir)
    check_log = (args.capture / 'phase3-mlx-check-v1.log').read_text()
    if 'Metal: 18 cases passed' not in check_log or 'Requested checks passed.' not in check_log:
        raise ValueError('required real-Metal operator checks did not pass')
    if quality_manifest['git_head'] != bench_manifest['git_head']:
        raise ValueError('quality and benchmark source revisions differ')
    quality_sources, bench_sources = validate_sources(quality_manifest), validate_sources(bench_manifest)
    activities = {}
    for name, path in (('quality', quality_dir), ('benchmark', bench_dir)):
        rows = [json.loads(line) for line in (path / 'activity.jsonl').read_text().splitlines()]
        if not rows or any('monotonic' not in row or 'top_cpu' not in row for row in rows):
            raise ValueError('activity samples incomplete')
        activities[name] = len(rows)
    checks = {}
    for batch, values in bench.items():
        rust, python = values['rust'], values['python']
        checks[batch] = {
            'median_latency_lower': rust['median_ms'] < python['median_ms'],
            'median_process_peak_rss_lower': statistics.median(rust['peak_rss_bytes']) < statistics.median(python['peak_rss_bytes']),
            'median_mlx_peak_lower': statistics.median(x['peak'] for x in rust['mlx_memory']) < statistics.median(x['peak'] for x in python['mlx_memory']),
        }
    print(json.dumps({'git_head': quality_manifest['git_head'], 'quality': quality, 'benchmarks': bench,
                      'tracked_sources_checked': {'quality': quality_sources, 'benchmark': bench_sources},
                      'activity_samples': activities, 'comparison_checks': checks,
                      'all_comparison_checks_passed': all(all(v.values()) for v in checks.values()),
                      'limitations': ['Activity samples are CPU-only and do not establish GPU idleness.',
                                      'RSS covers startup and loading; MLX counters do not establish total physical memory.',
                                      'Measurements are from one Apple system; compare actual samples and background activity.']},
                     indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
