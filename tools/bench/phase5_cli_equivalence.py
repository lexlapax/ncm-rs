"""Compare baseline, alias and unified CLI stdout on fixed Phase 5 requests.

This is a per-request output check, not a quality or performance benchmark.
Artifacts and binaries must be trusted and stable for the duration of the run.
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROPE_SHA = '9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc'
PROFILES = {
    'sudoku-easy': ('sudoku-easy-qualified', 'ad6a99ad7d5ccc069e0fedf26a85f755e06cd76590463e1e04e17363e35d9832', 322, 89),
    'sudoku-medium': ('sudoku-medium-phase3', 'facff42955d42294dc94f33d54688dc53046c1f9cc029e903f1ffe787ce45219', 322, 89),
    'sudoku-hard': ('sudoku-hard-phase3', '9ed3e86fdc3d193c88e1f4aae8f58a54019f985bdcf2acb9c851772c51dad86b', 322, 89),
    'tiny': ('tinystories-phase4-v1', '0337e4d2ab2e52d6808123847ad945749276a616d5865e3db5f9861f4ff48db1', 641, 128),
    'xsum': ('xsum-inference-rope-v2-final', '8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4', 200, 48),
    'wmt14': ('wmt14-inference-v1-final', '01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81', 313, 64),
    'lm1b-small': ('lm1b-small-inference-v1', '58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76', 641, 128),
    'lm1b-medium': ('lm1b-medium-inference-v1', '2fea4a252de0cecbe6ab12b2c4a0077cff66498e6344894694ba02876a38c30c', 641, 128),
    'lm1b-large': ('lm1b-large-inference-v1', '86676fed0f9813b3ec293aca87133eca61fc609d10d620978375d12c0b03698a', 641, 128),
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_output(raw, rows, code_length, id_length, seeded=False):
    result = json.loads(raw)
    expected = {'codes', 'ids', 'seed'} if seeded else {'codes', 'ids'}
    if not isinstance(result, dict) or set(result) != expected or (seeded and result['seed'] != 42):
        raise ValueError('unexpected output fields or seed')
    for key, length in (('codes', code_length), ('ids', id_length)):
        value = result[key]
        if (not isinstance(value, list) or len(value) != rows
                or any(not isinstance(row, list) or len(row) != length
                       or any(type(item) is not int or item < 0 for item in row) for row in value)):
            raise ValueError(f'invalid {key} output shape or values')


def compare_case(name, baseline, candidate, artifact, device, input_file, rope, output, rows):
    _, _, code_length, id_length = PROFILES[name]
    if name.startswith('sudoku-'):
        alias, old, new = 'generate', [], []
    elif name == 'tiny':
        alias, old, new = 'sample-tinystories', ['42'], ['--seed', '42']
    elif name == 'xsum':
        alias, old, new = 'sample-xsum', [], []
    elif name == 'wmt14':
        alias, old, new = 'sample-wmt14', [], []
    else:
        alias, old, new = 'sample-lm1b', ['42', str(rope)], ['--seed', '42', '--reference-rope', str(rope)]
    runs = [('baseline', baseline, alias, old), ('alias', candidate, alias, old),
            ('unified', candidate, 'sample', new)]
    outputs = []
    commands = []
    for label, binary, command, options in runs:
        argv = [str(binary), command, str(artifact), device, str(input_file), *options]
        process = subprocess.run(argv, capture_output=True, timeout=900, check=False)
        (output / f'{name}-{label}.stdout').write_bytes(process.stdout)
        (output / f'{name}-{label}.stderr').write_bytes(process.stderr)
        if process.returncode:
            raise RuntimeError(f'{name} {label} exited {process.returncode}; see captured stderr')
        validate_output(process.stdout, rows, code_length, id_length, name == 'tiny')
        outputs.append(process.stdout)
        commands.append(argv)
    if not outputs[0] == outputs[1] == outputs[2]:
        raise ValueError(f'{name}: stdout differs')
    return {'rows': rows, 'sha256': hashlib.sha256(outputs[0]).hexdigest(), 'commands': commands}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('baseline', 'candidate', 'artifacts', 'inputs', 'xsum-input', 'wmt14-input', 'rope', 'output'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--baseline-sha256', required=True)
    parser.add_argument('--candidate-sha256', required=True)
    parser.add_argument('--platform', choices=('cuda', 'mlx'), required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('output directory already exists')
    baseline, candidate, artifacts, inputs, xsum, wmt14, rope = (
        getattr(args, name).resolve() for name in
        ('baseline', 'candidate', 'artifacts', 'inputs', 'xsum_input', 'wmt14_input', 'rope'))
    # Do not resolve the final component: a dangling symlink must not become a destination.
    # As with the input artifacts, parent directories must be trusted and stable.
    output = args.output.absolute()
    if digest(baseline) != args.baseline_sha256 or digest(candidate) != args.candidate_sha256:
        raise ValueError('binary hash differs from requested identity')
    if baseline == candidate or args.baseline_sha256 == args.candidate_sha256:
        raise ValueError('baseline and candidate must be distinct binaries')
    if digest(rope) != ROPE_SHA:
        raise ValueError('reference rotary table hash differs')
    manifests = {}
    for name, (directory, expected, _, _) in PROFILES.items():
        path = artifacts / directory / 'manifest.json'
        if digest(path) != expected:
            raise ValueError(f'{name}: manifest hash differs')
        manifests[name] = path
    output.mkdir(parents=True, exist_ok=False)
    tiny, lm = output / 'tiny-input.json', output / 'lm1b-input.json'
    tiny.write_text('[[0]]\n', encoding='utf-8')
    lm.write_text('[[101]]\n', encoding='utf-8')
    paths = {name: inputs / (name.removeprefix('sudoku-') + '-two-inputs.json')
             for name in PROFILES if name.startswith('sudoku-')}
    paths.update({'tiny': tiny, 'xsum': xsum, 'wmt14': wmt14,
                  'lm1b-small': lm, 'lm1b-medium': lm, 'lm1b-large': lm})
    watched = {str(path): digest(path) for path in
               [Path(__file__).resolve(), baseline, candidate, rope, *manifests.values(), *paths.values()]}
    results = {}
    for name in PROFILES:
        rows = json.loads(paths[name].read_bytes())
        if not isinstance(rows, list) or not rows or any(not isinstance(row, list) for row in rows):
            raise ValueError(f'{name}: invalid input rows')
        device = ('mlx' if args.platform == 'mlx' else
                  'cuda' if name.startswith('sudoku-') or name == 'wmt14' else 'cuda-cudnn')
        results[name] = compare_case(name, baseline, candidate, artifacts / PROFILES[name][0],
                                     device, paths[name], rope, output, len(rows))
        print(name, results[name]['sha256'], flush=True)
    if watched != {path: digest(Path(path)) for path in watched}:
        raise ValueError('binary, manifest, rotary or input changed during run')
    report = {'scope': 'fixed-request byte equivalence only; not quality or performance',
              'platform': args.platform, 'watched_sha256': watched, 'results': results,
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    (output / 'result.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
