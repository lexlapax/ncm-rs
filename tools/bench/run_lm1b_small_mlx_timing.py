"""Alternating fresh-process LM1B-small Metal timing: Rust versus the Python MLX adaptation.

For each batch, three repetitions per runtime run in alternating order. Each process loads the
model once, runs the warmups, then the measured requests. Peak process RSS comes from
`/usr/bin/time -l`. A cell passes when Rust is lower than Python in pooled median latency,
pooled p95 latency and median peak process RSS. MLX active/peak/cache counters are reported
separately and are not compared with RSS.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / '.reference/mlx-venv/bin/python'
RUST = ROOT / 'target/release/examples/lm1b_small_mlx_timing_probe'
ARTIFACT = ROOT / 'artifacts/canonical/lm1b-small-inference-v1'
PROBE = ROOT / 'tools/reference/lm1b_small_mlx_timing_probe.py'
BATCHES = (1, 4, 16)
RUNTIMES = ('python', 'rust')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def terminate_group(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def checked_result(result, runtime, batch, warmups, trials):
    values = result['seconds']
    if (result['runtime'] != runtime or result['batch'] != batch or result['warmups'] != warmups
            or result['trials'] != trials or len(values) != trials
            or any(not math.isfinite(x) or x <= 0 for x in values)
            or len(result['signatures']) != trials or any(len(x) != 64 for x in result['signatures'])
            or not math.isfinite(result['load_seconds']) or result['load_seconds'] <= 0):
        raise ValueError(f'invalid {runtime} probe result at batch {batch}')
    return result


def peak_rss_bytes(path):
    values = re.findall(r'^\s*(\d+)\s+maximum resident set size\s*$', path.read_text(), re.MULTILINE)
    if len(values) != 1 or int(values[0]) <= 0:
        raise ValueError(f'missing peak RSS in {path.name}')
    return int(values[0])


def summarize(rows, batches, trials):
    summary = {}
    for batch in batches:
        cell = {}
        for runtime in RUNTIMES:
            selected = [r for r in rows if r['runtime'] == runtime and r['batch'] == batch]
            if sorted(r['rep'] for r in selected) != [0, 1, 2]:
                raise ValueError(f'incomplete repetitions for {runtime} batch {batch}')
            values = sorted(x for r in selected for x in r['seconds'])
            if len(values) != 3 * trials:
                raise ValueError(f'unexpected sample count for {runtime} batch {batch}')
            cell[runtime] = {
                'median_seconds': statistics.median(values),
                'p95_seconds': values[math.ceil(len(values) * .95) - 1],
                'samples_per_second_at_median': batch / statistics.median(values),
                'median_peak_rss_bytes': statistics.median(r['peak_rss_bytes'] for r in selected),
                'median_load_seconds': statistics.median(r['load_seconds'] for r in selected),
                'mlx_peak_bytes': [r['mlx_memory']['peak'] for r in selected],
            }
        rust, python = cell['rust'], cell['python']
        cell['checks'] = {
            'median_lower': rust['median_seconds'] < python['median_seconds'],
            'p95_lower': rust['p95_seconds'] < python['p95_seconds'],
            'peak_rss_lower': rust['median_peak_rss_bytes'] < python['median_peak_rss_bytes'],
        }
        cell['passed'] = all(cell['checks'].values())
        summary[str(batch)] = cell
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--rotary', required=True, type=Path)
    p.add_argument('--batch', type=int, choices=BATCHES, action='append',
                   help='measure only these batch sizes (default: 1, 4 and 16)')
    p.add_argument('--smoke', action='store_true', help='one warmup and three trials; not evidence')
    args = p.parse_args()
    if sys.platform != 'darwin' or platform.machine() != 'arm64':
        raise RuntimeError('Metal timing requires Apple Silicon')
    batches = tuple(args.batch or BATCHES)
    warmups, trials = (1, 3) if args.smoke else (5, 30)
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('new output directory under reports required')
    if digest(args.rotary) != '9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc':
        raise ValueError('unexpected rotary tables')
    sources = [Path(__file__).resolve(), PROBE, ROOT / 'tools/reference/lm1b_small_mlx_reference.py',
               RUST, ARTIFACT / 'manifest.json',
               ROOT / 'Cargo.lock', ROOT / 'crates/ncm-backend-mlx/examples/lm1b_small_mlx_timing_probe.rs',
               args.rotary.resolve()]
    for directory in ('crates/ncm-core/src', 'crates/ncm-model/src', 'crates/ncm-backend-mlx/src'):
        sources.extend(sorted(path for path in (ROOT / directory).iterdir() if path.suffix == '.rs'))
    before = {str(path): digest(path) for path in sources}
    out.mkdir(parents=True)
    manifest = {
        'scope': 'LM1B-small Metal latency and memory, Rust versus Python MLX adaptation',
        'smoke': args.smoke, 'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'sources_sha256': before, 'batches': batches, 'warmups': warmups, 'trials': trials, 'repetitions': 3,
        'order': 'alternating runtime per repetition, fresh process each',
        'python_sampler': 'mx.random.categorical on device; reference forward pass and FP64 penalties',
        'rust_sampler': 'host FP64 inverse CDF (CLI default)',
        'host': {'machine': platform.machine(), 'mac_ver': platform.mac_ver()[0],
                 'loadavg_start': os.getloadavg()},
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    rows, process = [], None
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT / 'tools/reference'), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')
    try:
        for batch in batches:
            for rep in range(3):
                for runtime in (RUNTIMES if rep % 2 == 0 else RUNTIMES[::-1]):
                    stem = f'{runtime}-b{batch}-r{rep}'
                    if runtime == 'rust':
                        command = [str(RUST), str(ARTIFACT), str(args.rotary), str(batch), str(warmups), str(trials)]
                    else:
                        command = [str(PYTHON), str(PROBE), str(ARTIFACT), str(args.rotary), str(batch),
                                   str(warmups), str(trials), str(out / f'{stem}.json')]
                    with (out / f'{stem}.log').open('x') as log:
                        process = subprocess.Popen(['/usr/bin/time', '-l', '-o', str(out / f'{stem}.time'), *command],
                                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                   start_new_session=True)
                        started = time.monotonic()
                        try:
                            code = process.wait(timeout=1800)
                        except subprocess.TimeoutExpired:
                            terminate_group(process)
                            raise
                    if code:
                        raise RuntimeError(f'{stem} exited {code}; see retained log')
                    if runtime == 'rust':
                        result = json.loads((out / f'{stem}.log').read_text().splitlines()[-1])
                        (out / f'{stem}.json').write_text(json.dumps(result) + '\n')
                    else:
                        result = json.loads((out / f'{stem}.json').read_bytes())
                    checked_result(result, runtime, batch, warmups, trials)
                    rows.append({'runtime': runtime, 'batch': batch, 'rep': rep,
                                 'start_monotonic': started, 'end_monotonic': time.monotonic(),
                                 'seconds': result['seconds'], 'load_seconds': result['load_seconds'],
                                 'peak_rss_bytes': peak_rss_bytes(out / f'{stem}.time'),
                                 'mlx_memory': result['mlx_memory']})
                    print(stem, 'median', statistics.median(result['seconds']), flush=True)
    finally:
        if process is not None:
            terminate_group(process)
    if before != {str(path): digest(path) for path in sources}:
        raise ValueError('sources changed during measurement')
    summary = summarize(rows, batches, trials)
    result = {'scope': manifest['scope'], 'smoke': args.smoke, 'rows': rows, 'summary': summary,
              'loadavg_end': os.getloadavg(),
              'limits': 'separate random streams and samplers; desktop activity possible; '
                        'RSS is whole-process peak; MLX counters are allocator-scoped'}
    (out / 'result.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({b: {'passed': c['passed'], **c['checks']} for b, c in summary.items()}, sort_keys=True))


if __name__ == '__main__':
    main()
