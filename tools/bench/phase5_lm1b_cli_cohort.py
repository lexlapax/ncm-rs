"""Compare retained CUDA LM1B cohort rows with old and unified CLI requests.

This is a one-seed routing retest, not the full 80-seed quality gate or a
performance measurement. Inputs, binaries and output parent must be trusted.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = 'f10abaaec672dd7c0c62034972fb4b15fc204f2b1f5693e4a09f5e9640767098'
NEW = '6194111fe3860584492676c6d14f50124a77f8dcf18f316a959b8abd16a20278'
ROPE = '9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc'
SIZES = {
    'small': ('58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76', '742d4f25e6c85b2037cfd621a86e48948b54bff137e639894330702feac99506'),
    'medium': ('2fea4a252de0cecbe6ab12b2c4a0077cff66498e6344894694ba02876a38c30c', '57f89148f93318ba7567bf32b7270d48867b8e719b8095855a24013bab1a20da'),
    'large': ('86676fed0f9813b3ec293aca87133eca61fc609d10d620978375d12c0b03698a', '166c8e49c3189595c84eb952fe99584e6515ef095e2dbfd720c3628924876a3f'),
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked(path, sha):
    raw = path.read_bytes()
    if digest(raw) != sha:
        raise ValueError(f'identity differs: {path}')
    return raw


def request_seed(seed, ordinal):
    if type(seed) is not int or not 2000 <= seed < 2080 or type(ordinal) is not int or not 0 <= ordinal < 32:
        raise ValueError('invalid cohort seed or batch')
    return (seed << 32) | ordinal


def validate_capture(capture, size, seed, manifest):
    if (capture['size'], capture['seed'], capture['count'], capture['batch_size'], capture['artifact_manifest_sha256']) != (size, seed, 256, 8, manifest):
        raise ValueError('capture metadata differs')
    for key, width, bound in (('codes', 641, 8192), ('raw_ids', 128, 30528)):
        rows = capture[key]
        if not isinstance(rows, list) or len(rows) != 256 or any(
            not isinstance(row, list) or len(row) != width or
            any(type(value) is not int or not 0 <= value < bound for value in row) for row in rows
        ):
            raise ValueError('invalid capture rows')


def compare(old, new, capture, ordinal):
    if old != new:
        raise ValueError('CLI output bytes differ')
    result = json.loads(old)
    expected = {'codes': capture['codes'][ordinal * 8:(ordinal + 1) * 8],
                'ids': capture['raw_ids'][ordinal * 8:(ordinal + 1) * 8]}
    if result != expected:
        raise ValueError('CLI output differs from cohort')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', choices=tuple(SIZES), required=True)
    parser.add_argument('--seed', type=int, default=2000)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--rope', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=32)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('output exists')
    if not 0 <= args.start < args.stop <= 32:
        raise ValueError('invalid batch range')
    request_seed(args.seed, args.start)
    old, new, rope = args.baseline.resolve(), args.candidate.resolve(), args.rope.resolve()
    if old == new:
        raise ValueError('binaries must be distinct')
    artifact = (args.artifacts / f'lm1b-{args.size}-inference-v1').resolve()
    manifest_sha, receipt_sha = SIZES[args.size]
    directory = ROOT / f'reports/phase5-lm1b-{args.size}-cuda-retest-v1/rust'
    receipt_path, capture_path = directory / 'receipt.json', directory / f'rust-seed-{args.seed}.json'
    receipt = json.loads(checked(receipt_path, receipt_sha))
    if receipt['runtime'] != 'rust' or set(receipt['captures']) != {str(seed) for seed in range(2000, 2080)}:
        raise ValueError('invalid cohort receipt')
    capture_sha = receipt['captures'][str(args.seed)]
    capture = json.loads(checked(capture_path, capture_sha))
    validate_capture(capture, args.size, args.seed, manifest_sha)
    watched = {old: OLD, new: NEW, rope: ROPE, artifact / 'manifest.json': manifest_sha,
               receipt_path: receipt_sha, capture_path: capture_sha,
               Path(__file__).resolve(): digest(Path(__file__).read_bytes())}
    for path, sha in watched.items():
        checked(path, sha)
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = args.output / 'inputs.json'
    inputs.write_text(json.dumps([[101]] * 8, separators=(',', ':')) + '\n', encoding='utf-8')
    input_sha = digest(inputs.read_bytes())
    records = []
    for ordinal in range(args.start, args.stop):
        seed = str(request_seed(args.seed, ordinal))
        commands = [[str(old), 'sample-lm1b', str(artifact), 'cuda-cudnn', str(inputs), seed, str(rope)],
                    [str(new), 'sample', str(artifact), 'cuda-cudnn', str(inputs), '--seed', seed, '--reference-rope', str(rope)]]
        outputs = []
        for label, command in zip(('old', 'new'), commands, strict=True):
            process = subprocess.run(command, capture_output=True, timeout=900, check=False)
            (args.output / f'{ordinal:02}-{label}.json').write_bytes(process.stdout)
            (args.output / f'{ordinal:02}-{label}.stderr').write_bytes(process.stderr)
            if process.returncode:
                raise RuntimeError(f'{label} batch {ordinal} exited {process.returncode}')
            outputs.append(process.stdout)
        compare(*outputs, capture, ordinal)
        records.append({'ordinal': ordinal, 'request_seed': int(seed), 'output_sha256': digest(outputs[0])})
        if (ordinal + 1) % 8 == 0 or ordinal + 1 == args.stop:
            print(args.size, ordinal + 1, '/', args.stop, flush=True)
    checked(inputs, input_sha)
    for path, sha in watched.items():
        checked(path, sha)
    with (args.output / 'summary.json').open('x', encoding='utf-8') as stream:
        json.dump({'size': args.size, 'seed': args.seed, 'start': args.start, 'stop': args.stop,
                   'rows': (args.stop - args.start) * 8, 'input_sha256': input_sha,
                   'scope': 'one-seed CUDA cohort CLI routing comparison, not quality or performance',
                   'watched_sha256': {str(path): sha for path, sha in watched.items()}, 'batches': records},
                  stream, indent=2, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    main()
