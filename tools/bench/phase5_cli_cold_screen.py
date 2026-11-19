"""Alternate fresh-process CLI cold-request timings; not the Phase 5 performance gate.

Each invocation loads the model and performs one request. Warmups are separate
processes and do not establish warm in-process latency. Inputs, binaries and
artifacts must be trusted and stable. No system policy is modified.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import time
from pathlib import Path

ROUNDS = 3
WARMUPS = 5
TRIALS = 30


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def summarize(records, condition):
    rows = [r for r in records if r['condition'] == condition and not r['warmup']]
    if len(rows) != ROUNDS * TRIALS:
        raise ValueError('missing measured samples')
    values = sorted(r['elapsed_seconds'] for r in rows)
    rss = sorted(r['max_rss_kib'] for r in rows)
    median = statistics.median(values)
    return {'samples': len(rows), 'median_seconds': median,
            'p95_seconds_nearest_rank': values[math.ceil(0.95 * len(values)) - 1],
            'median_absolute_deviation_seconds': statistics.median(abs(x - median) for x in values),
            'min_seconds': values[0], 'max_seconds': values[-1],
            'median_max_rss_kib': statistics.median(rss), 'max_rss_kib': rss[-1]}


def execute(output, condition, binary, command, artifact, input_path, group, ordinal, warmup, expected,
            profile='sudoku', rope=None):
    stem = f'{group}-{ordinal:02d}-{condition}'
    stdout, stderr, rss = (output / f'{stem}.{suffix}' for suffix in ('stdout', 'stderr', 'rss'))
    device = 'cuda-cudnn' if profile in ('tinystories', 'xsum', 'lm1b') else 'cuda'
    if profile == 'tinystories':
        options = ['42'] if command == 'sample-tinystories' else ['--seed', '42']
    elif profile == 'lm1b':
        if rope is None:
            raise ValueError('LM1B requires a rotary table')
        options = ['42', str(rope)] if command == 'sample-lm1b' else ['--seed', '42', '--reference-rope', str(rope)]
    else:
        options = []
    argv = ['/usr/bin/time', '-f', '%M', '-o', str(rss), str(binary), command,
            str(artifact), device, str(input_path), *options]
    before = time.perf_counter_ns()
    with stdout.open('xb') as out, stderr.open('xb') as err:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                start_new_session=True)
        def timeout_handler(_signum, _frame):
            raise subprocess.TimeoutExpired(argv, 900)

        previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(900)
        try:
            # A timed Popen.wait polls in coarse intervals, biasing short timings.
            returncode = proc.wait()
        except BaseException:
            # GNU time owns the CLI child; terminate only the new process group.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            raise
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
    elapsed = (time.perf_counter_ns() - before) / 1e9
    if returncode != 0:
        raise RuntimeError(f'{stem}: exit {returncode}; see stderr')
    result_hash = digest(stdout)
    if result_hash != expected:
        raise ValueError(f'{stem}: output hash differs')
    max_rss = int(rss.read_text(encoding='utf-8').strip())
    if max_rss <= 0 or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError(f'{stem}: invalid measurement')
    return {'group': group, 'ordinal': ordinal, 'condition': condition, 'warmup': warmup,
            'elapsed_seconds': elapsed, 'max_rss_kib': max_rss, 'output_sha256': result_hash}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'artifact', 'input', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    for name in ('baseline', 'candidate', 'manifest', 'input', 'expected-output'):
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--profile', choices=('sudoku', 'tinystories', 'xsum', 'wmt14', 'lm1b'), default='sudoku')
    parser.add_argument('--rope', type=Path)
    parser.add_argument('--rope-sha256')
    parser.add_argument('--watch-file', type=Path, action='append', default=[],
                        help='additional native library to hash before and after measurement')
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('output already exists')
    if (args.profile == 'lm1b') != (args.rope is not None and args.rope_sha256 is not None):
        raise ValueError('LM1B requires both --rope and --rope-sha256, other profiles neither')
    if args.profile != 'lm1b' and (args.rope is not None or args.rope_sha256 is not None):
        raise ValueError('rotary options are only supported for LM1B')
    rope = args.rope.resolve() if args.rope is not None else None
    baseline, candidate, artifact, input_path = (
        getattr(args, name).resolve() for name in ('baseline', 'candidate', 'artifact', 'input'))
    output = args.output.absolute()
    for key in ('LD_PRELOAD', 'LD_AUDIT', 'CUDA_INJECTION64_PATH'):
        if os.environ.get(key):
            raise ValueError(f'unsupported environment variable: {key}')
    if baseline == candidate or args.baseline_sha256 == args.candidate_sha256:
        raise ValueError('baseline and candidate must differ')
    monitor_path = shutil.which('nvidia-smi')
    if not monitor_path:
        raise ValueError('nvidia-smi unavailable')
    watch_paths = [Path(__file__).resolve(), baseline, candidate, artifact / 'manifest.json',
                   input_path, Path('/usr/bin/time').resolve(), Path(monitor_path).resolve(),
                   *([rope] if rope is not None else []), *(path.resolve() for path in args.watch_file)]
    watched = {path: digest(path) for path in watch_paths}
    for path, expected in ((baseline, args.baseline_sha256), (candidate, args.candidate_sha256),
                           (artifact / 'manifest.json', args.manifest_sha256), (input_path, args.input_sha256)):
        if watched[path] != expected:
            raise ValueError(f'identity differs: {path}')
    if rope is not None and watched[rope] != args.rope_sha256:
        raise ValueError(f'identity differs: {rope}')
    output.mkdir(parents=True, exist_ok=False)
    records = []
    monitor_argv = [monitor_path, '--query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used',
                    '--format=csv,noheader,nounits', '-l', '1']
    monitor_exited_early = False
    with (output / 'gpu.csv').open('xb') as gpu, (output / 'gpu.stderr').open('xb') as gpu_error:
        monitor = subprocess.Popen(monitor_argv, stdin=subprocess.DEVNULL, stdout=gpu, stderr=gpu_error)
        try:
            for group in range(1, ROUNDS + 1):
                for ordinal in range(WARMUPS + TRIALS):
                    old_command = {'sudoku': 'generate', 'tinystories': 'sample-tinystories',
                                   'xsum': 'sample-xsum', 'wmt14': 'sample-wmt14',
                                   'lm1b': 'sample-lm1b'}[args.profile]
                    order = [('baseline', baseline, old_command), ('candidate', candidate, 'sample')]
                    if (group + ordinal) % 2:
                        order.reverse()
                    for condition, binary, command in order:
                        record = execute(output, condition, binary, command, artifact, input_path,
                                         group, ordinal, ordinal < WARMUPS, args.expected_output_sha256,
                                         args.profile, rope)
                        records.append(record)
                        if monitor.poll() is not None:
                            raise RuntimeError('GPU monitor exited during measurement')
                print(f'completed group {group}/{ROUNDS}', flush=True)
        finally:
            monitor_exited_early = monitor.poll() is not None
            if not monitor_exited_early:
                monitor.terminate()
            try:
                monitor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait(timeout=10)
    if monitor_exited_early or not (output / 'gpu.csv').stat().st_size:
        raise ValueError('GPU monitor exited early or produced no samples')
    if watched != {path: digest(path) for path in watched}:
        raise ValueError('watched input changed during measurement')
    result = {'scope': 'fresh CLI process including load, generation and stdout; no warm in-process timing',
              'quality_scope': 'one fixed-request output hash only; not full quality',
              'device_memory_scope': 'system-wide sampled GPU usage only; native allocation/reservation peaks not measured',
              'platform': 'Linux CUDA', 'profile': args.profile, 'rounds': ROUNDS,
              'warmups_per_group_per_condition': WARMUPS,
              'trials_per_group_per_condition': TRIALS, 'monitor_returncode': monitor.returncode,
              'gpu_csv_sha256': digest(output / 'gpu.csv'),
              'watched_sha256': {str(path): sha for path, sha in watched.items()},
              'records': records, 'summary': {key: summarize(records, key) for key in ('baseline', 'candidate')}}
    with (output / 'result.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
