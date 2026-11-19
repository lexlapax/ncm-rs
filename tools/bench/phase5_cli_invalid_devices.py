"""Check unified CLI rejection of an invalid device on fixed qualified profiles.

This does not run inference or establish GPU quality/performance. Artifacts, binaries,
inputs and output parent directories must be trusted and stable during the run.
"""
import argparse
import json
import subprocess
from pathlib import Path

from phase5_cli_equivalence import PROFILES, digest

INVALID_DEVICE = 'phase5-invalid-device'


def check_case(binary, artifact, input_path, expected):
    argv = [str(binary), 'sample', str(artifact), INVALID_DEVICE, str(input_path)]
    process = subprocess.run(argv, capture_output=True, timeout=120, check=False)
    if process.returncode != 1 or process.stdout or expected not in process.stderr:
        raise ValueError(f'unexpected rejection result: {argv!r}, exit={process.returncode}, '
                         f'stdout={process.stdout!r}, stderr={process.stderr!r}')
    return {'exit': process.returncode, 'stdout_bytes': len(process.stdout),
            'stderr': process.stderr.decode('utf-8'), 'command': argv}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'artifacts', 'inputs', 'xsum-input', 'wmt14-input', 'tiny-input', 'lm1b-input', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--binary-sha256', required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('output already exists')
    binary, artifacts, inputs, xsum, wmt14, tiny, lm1b = (
        getattr(args, name).resolve() for name in
        ('binary', 'artifacts', 'inputs', 'xsum_input', 'wmt14_input', 'tiny_input', 'lm1b_input'))
    if digest(binary) != args.binary_sha256:
        raise ValueError('binary identity differs')
    paths = {name: inputs / (name.removeprefix('sudoku-') + '-two-inputs.json')
             for name in PROFILES if name.startswith('sudoku-')}
    paths.update({'tiny': tiny, 'xsum': xsum, 'wmt14': wmt14,
                  'lm1b-small': lm1b, 'lm1b-medium': lm1b, 'lm1b-large': lm1b})
    manifests = {name: artifacts / directory / 'manifest.json'
                 for name, (directory, _, _, _) in PROFILES.items()}
    for name, path in manifests.items():
        if digest(path) != PROFILES[name][1]:
            raise ValueError(f'{name}: manifest identity differs')
    watched = {path: digest(path) for path in
               [Path(__file__).resolve(), Path(__file__).with_name('phase5_cli_equivalence.py').resolve(),
                binary, *manifests.values(), *paths.values()]}
    results = {}
    for name, (directory, _, _, _) in PROFILES.items():
        if name.startswith('lm1b-'):
            expected = b'requires an explicit cuda-cudnn or mlx device'
        elif name == 'xsum':
            expected = b'XSum requested device is unavailable in this build; no fallback'
        elif name == 'wmt14':
            expected = b'WMT14 requested device is unavailable in this build'
        else:
            expected = b'requested device is unavailable in this build; no fallback'
        results[name] = check_case(binary, artifacts / directory, paths[name], expected)
        print(name, 'rejected', flush=True)
    if watched != {path: digest(path) for path in watched}:
        raise ValueError('watched input changed')
    result = {'scope': 'invalid device rejection only; no inference or quality',
              'device': INVALID_DEVICE, 'results': results,
              'watched_sha256': {str(path): sha for path, sha in watched.items()}}
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write('\n')


if __name__ == '__main__':
    main()
