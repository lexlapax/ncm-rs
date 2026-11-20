"""Alternate old-ID and unified ID/text/decode cold CUDA requests.

Fresh processes measure startup through output, not warm model performance. This
screen is not GPU-isolated, a Python comparison, or a release gate. Trusted stable
inputs and output parent are required.
"""
import argparse
import json
import math
import os
import signal
import statistics
import subprocess
import time
from pathlib import Path

from phase5_cli_cold_screen import digest
from phase5_text_generation_screen import BASELINE_SHA, BINARY_SHA, PROFILES

CONDITIONS = ('old_id', 'new_id', 'new_text', 'new_text_decode')
HASHES = {
    'xsum': ('87be072a79210d62215efaa82f801c64612f6868cf545a3fb888ca14aee6cdca',
             '6db35c54fece2e27ea0e96e6f4e29cebba23a42bd20e899401218658cd2a0328'),
    'wmt14': ('17257d936f55b159c8cdc8ce9a465bf9d0498a446d544215c171eae82f2a604a',
              'fd68c625bc98b4e7fccfde240db198c67ae8251e8607110b5319e8a617098d32'),
}


def order(group, ordinal):
    shift = ((group - 1) * 35 + ordinal) % len(CONDITIONS)
    return CONDITIONS[shift:] + CONDITIONS[:shift]


def summarize(records):
    expected = {(group, ordinal, condition) for group in range(1, 4) for ordinal in range(35) for condition in CONDITIONS}
    found = set()
    result = {}
    for row in records:
        key = (row['group'], row['ordinal'], row['condition'])
        if key not in expected or key in found or row['warmup'] != (row['ordinal'] < 5):
            raise ValueError('invalid trial inventory')
        if (not math.isfinite(row['elapsed_seconds']) or row['elapsed_seconds'] <= 0
                or type(row['max_rss_kib']) is not int or row['max_rss_kib'] <= 0):
            raise ValueError('invalid trial values')
        found.add(key)
    if found != expected:
        raise ValueError('missing trials')
    for condition in CONDITIONS:
        rows = [row for row in records if row['condition'] == condition and not row['warmup']]
        times = sorted(row['elapsed_seconds'] for row in rows)
        rss = [row['max_rss_kib'] for row in rows]
        median = statistics.median(times)
        result[condition] = {'samples': len(times), 'median_seconds': median,
                             'p95_seconds': times[math.ceil(.95 * len(times)) - 1],
                             'mad_seconds': statistics.median(abs(value - median) for value in times),
                             'median_rss_kib': statistics.median(rss), 'max_rss_kib': max(rss)}
    return result


def command(condition, profile, old, new, artifact, ids, text):
    common = [str(artifact), profile['device']]
    if condition == 'old_id':
        return [str(old), f'sample-{profile["name"]}', *common, str(ids)]
    argv = [str(new), 'sample', *common, str(text if condition != 'new_id' else ids)]
    if condition in ('new_text', 'new_text_decode'):
        argv.append('--text')
    if condition == 'new_text_decode':
        argv.append('--decode')
    if condition not in CONDITIONS:
        raise ValueError('unknown condition')
    return argv


def execute(output, condition, argv, group, ordinal, expected_sha):
    stem = f'{group}-{ordinal:02}-{condition}'
    stdout, stderr, rss = (output / f'{stem}.{suffix}' for suffix in ('stdout', 'stderr', 'rss'))
    wrapped = ['/usr/bin/time', '-f', '%M', '-o', str(rss), *argv]
    before = time.perf_counter_ns()
    with stdout.open('xb') as out, stderr.open('xb') as err:
        proc = subprocess.Popen(wrapped, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
        def timeout_handler(_signum, _frame):
            raise subprocess.TimeoutExpired(wrapped, 900)
        handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(900)
        try:
            code = proc.wait()  # blocking wait avoids polling-biased timings
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            raise
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, handler)
    elapsed = (time.perf_counter_ns() - before) / 1e9
    if code or digest(stdout) != expected_sha:
        raise ValueError(f'{stem}: exit or output differs')
    maximum = int(rss.read_text().strip())
    if maximum <= 0 or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError('invalid measurement')
    return {'group': group, 'ordinal': ordinal, 'condition': condition, 'warmup': ordinal < 5,
            'elapsed_seconds': elapsed, 'max_rss_kib': maximum, 'output_sha256': expected_sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=tuple(PROFILES), required=True)
    for name in ('baseline', 'candidate', 'artifact-root', 'reference', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--watch-file', type=Path, action='append', default=[])
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('output exists')
    if any(os.environ.get(key) for key in ('LD_PRELOAD', 'LD_AUDIT', 'CUDA_INJECTION64_PATH')):
        raise ValueError('unsupported environment')
    profile = {**PROFILES[args.profile], 'name': args.profile}
    old, new, reference = args.baseline.resolve(), args.candidate.resolve(), args.reference.resolve()
    artifact = (args.artifact_root / profile['artifact']).resolve()
    from shutil import which
    monitor = which('nvidia-smi')
    if not monitor or old == new:
        raise ValueError('monitor or distinct binaries required')
    watched = {old: BASELINE_SHA, new: BINARY_SHA, reference: profile['reference'],
               artifact / 'manifest.json': profile['manifest']}
    for path in [Path(__file__).resolve(), Path(__file__).with_name('phase5_cli_cold_screen.py').resolve(),
                 Path(__file__).with_name('phase5_text_generation_screen.py').resolve(),
                 Path('/usr/bin/time').resolve(), Path(monitor).resolve(), *(p.resolve() for p in args.watch_file)]:
        watched[path] = digest(path)
    if any(digest(path) != sha for path, sha in watched.items()):
        raise ValueError('watched identity differs')
    data = json.loads(reference.read_bytes())
    if len(data['ids']) != profile['rows'] or len(data['texts']) != profile['rows']:
        raise ValueError('reference row count differs')
    args.output.mkdir(parents=True, exist_ok=False)
    ids, text = args.output / 'ids.json', args.output / 'text.json'
    for path, value in ((ids, [data['ids'][0]]), (text, [data['texts'][0]])):
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=True, separators=(',', ':'))
            stream.write('\n')
    watched[ids], watched[text] = digest(ids), digest(text)
    records = []
    with (args.output / 'gpu.csv').open('xb') as gpu, (args.output / 'gpu.stderr').open('xb') as error:
        watcher = subprocess.Popen([monitor, '--query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used',
                                    '--format=csv,noheader,nounits', '-l', '1'], stdout=gpu, stderr=error, stdin=subprocess.DEVNULL)
        try:
            for group in range(1, 4):
                for ordinal in range(35):
                    for condition in order(group, ordinal):
                        argv = command(condition, profile, old, new, artifact, ids, text)
                        expected = HASHES[args.profile][condition == 'new_text_decode']
                        records.append(execute(args.output, condition, argv, group, ordinal, expected))
                        if watcher.poll() is not None:
                            raise RuntimeError('GPU monitor exited early')
                print(args.profile, 'group', group, '/ 3', flush=True)
        finally:
            early = watcher.poll() is not None
            if not early:
                watcher.terminate()
            try:
                watcher.wait(timeout=10)
            except subprocess.TimeoutExpired:
                watcher.kill(); watcher.wait(timeout=10)
    gpu = args.output / 'gpu.csv'
    if early or not gpu.stat().st_size or any(digest(path) != sha for path, sha in watched.items()):
        raise ValueError('monitor failed or watched inputs changed')
    report = {'profile': args.profile, 'scope': 'cold CLI requests, shared GPU; not release qualification',
              'gpu_csv_sha256': digest(gpu), 'watched_sha256': {str(path): sha for path, sha in watched.items()},
              'records': records, 'summary': summarize(records)}
    with (args.output / 'result.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
