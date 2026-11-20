"""Measure LM1B explicit/default rotary and empty-text cold CUDA requests.

Fresh-process diagnostic only; no GPU isolation, Python comparison or release
qualification. Inputs, artifacts, binaries and output parent must be trusted.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path
from shutil import which

import phase5_text_cold_screen as cold
from phase5_cli_cold_screen import digest
from phase5_lm1b_cli_cohort import OLD, NEW, ROPE, SIZES

HASHES = {
    'small': ('520ae6b5372b2013bda8235dfec40c85265716014ac4a9ba58fec03419be5d71',
              'cd87171d7f866ad664cedf252c8eee86e857d5c66a00855468da73e009a423c6'),
    'medium': ('d9c1c40663e66cf472846e3781ca4995bab998b5bc86ec0c675fe0b29c506488',
               'c2995c6aaaa74f764f6602d6105aa0f51c121a2f752371d5cd6653e36f5a143e'),
    'large': ('bbdc24f6d0bd2150c41af25fb4a37835e9a191fcce5fe181461863421b9bfbf9',
              '398b182c08dea307d3518c2382fd958d9e5b9bddf3870da33996e455e3bb44d1'),
}


def command(condition, old, new, artifact, rope, ids, text):
    if condition == 'old_id':
        return [str(old), 'sample-lm1b', str(artifact), 'cuda-cudnn', str(ids), '42', str(rope)]
    if condition not in cold.CONDITIONS:
        raise ValueError('unknown condition')
    argv = [str(new), 'sample', str(artifact), 'cuda-cudnn', str(ids if condition == 'new_id' else text), '--seed', '42']
    if condition in ('new_text', 'new_text_decode'):
        argv.append('--text')
    if condition == 'new_text_decode':
        argv.append('--decode')
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', choices=tuple(SIZES), required=True)
    for name in ('baseline', 'candidate', 'artifact-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--watch-file', type=Path, action='append', default=[])
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('output exists')
    if any(os.environ.get(key) for key in ('LD_PRELOAD', 'LD_AUDIT', 'CUDA_INJECTION64_PATH')):
        raise ValueError('unsupported environment')
    old, new = args.baseline.resolve(), args.candidate.resolve()
    root = args.artifact_root.resolve()
    artifact = root / f'lm1b-{args.size}-inference-v1'
    rope = root / 'lm1b-reference-rope-v1.safetensors'
    monitor = which('nvidia-smi')
    if not monitor or old == new:
        raise ValueError('monitor or distinct binaries required')
    watched = {old: OLD, new: NEW, artifact / 'manifest.json': SIZES[args.size][0], rope: ROPE}
    for path in [Path(__file__).resolve(), Path(cold.__file__).resolve(),
                 Path(__file__).with_name('phase5_cli_cold_screen.py').resolve(),
                 Path(__file__).with_name('phase5_text_generation_screen.py').resolve(),
                 Path(__file__).with_name('phase5_lm1b_cli_cohort.py').resolve(),
                 Path('/usr/bin/time').resolve(), Path(monitor).resolve(), *(p.resolve() for p in args.watch_file)]:
        watched[path] = digest(path)
    if any(digest(path) != sha for path, sha in watched.items()):
        raise ValueError('watched identity differs')
    args.output.mkdir(parents=True, exist_ok=False)
    ids, text = args.output / 'ids.json', args.output / 'text.json'
    for path, raw in ((ids, b'[[101]]\n'), (text, b'[""]\n')):
        with path.open('xb') as stream:
            stream.write(raw)
    watched[ids], watched[text] = digest(ids), digest(text)
    records = []
    with (args.output / 'gpu.csv').open('xb') as gpu, (args.output / 'gpu.stderr').open('xb') as error:
        watcher = subprocess.Popen([monitor, '--query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used',
                                    '--format=csv,noheader,nounits', '-l', '1'], stdout=gpu, stderr=error, stdin=subprocess.DEVNULL)
        try:
            for group in range(1, 4):
                for ordinal in range(35):
                    for condition in cold.order(group, ordinal):
                        argv = command(condition, old, new, artifact, rope, ids, text)
                        expected = HASHES[args.size][condition == 'new_text_decode']
                        records.append(cold.execute(args.output, condition, argv, group, ordinal, expected))
                        if watcher.poll() is not None:
                            raise RuntimeError('GPU monitor exited early')
                print(args.size, 'group', group, '/ 3', flush=True)
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
    with (args.output / 'result.json').open('x', encoding='utf-8') as stream:
        json.dump({'size': args.size, 'scope': 'cold LM1B CLI requests, shared GPU; not release qualification',
                   'gpu_csv_sha256': digest(gpu), 'watched_sha256': {str(path): sha for path, sha in watched.items()},
                   'records': records, 'summary': cold.summarize(records)}, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
