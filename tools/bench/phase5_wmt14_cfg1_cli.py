"""Compare the unified CUDA CLI against the frozen WMT14 CFG=1 corpus.

Nonblind output equivalence, not a performance or new held-out experiment.
Inputs and output parent must be trusted and stable.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFIX_SHA = '22c76d597ae51dffadf61661d6b7c4d4290fd14b8809bfa6f1959b5c668df93f'
REFERENCE_SHA = '78f24bba0fecc8ce7c06abf22508516142a5331cc6df5572efd533f9211e9256'
MANIFEST_SHA = '01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def expected_batch(prefixes, reference, start):
    if len(prefixes) != 3003 or reference.get('count') != 3003 or reference.get('start') != 0 or reference.get('batch_size') != 4 or reference.get('cfg_scale') != 1.0:
        raise ValueError('invalid frozen corpus envelope')
    outputs = reference.get('outputs')
    if not isinstance(outputs, list) or len(outputs) != 3003 or start not in range(0, 3003, 4):
        raise ValueError('invalid frozen corpus rows or batch')
    end = min(start + 4, 3003)
    rows, inputs = outputs[start:end], prefixes[start:end]
    if any(not isinstance(row, list) or len(row) != 96 or any(type(v) is not int or not 0 <= v < 151680 for v in row) for row in inputs):
        raise ValueError('invalid prefix')
    if any(not isinstance(row, dict) or row.get('row') != i for i, row in enumerate(rows, start)):
        raise ValueError('invalid output row order')
    for row in rows:
        for key, width in (('codes', 313), ('raw_ids', 64)):
            values = row.get(key)
            if not isinstance(values, list) or len(values) != width or any(type(v) is not int or v < 0 for v in values):
                raise ValueError('invalid reference output')
    return inputs, {'codes': [row['codes'] for row in rows], 'ids': [row['raw_ids'] for row in rows]}


def check_output(raw, expected):
    actual = json.loads(raw)
    if not isinstance(actual, dict) or set(actual) != {'codes', 'ids'}:
        raise ValueError('invalid CLI output')
    for key in ('codes', 'ids'):
        if not isinstance(actual[key], list) or any(
            not isinstance(row, list) or any(type(value) is not int for value in row)
            for row in actual[key]
        ):
            raise ValueError('invalid CLI output rows')
    if actual != expected:
        raise ValueError('CLI output differs from frozen reference')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--binary-sha256', required=True)
    parser.add_argument('--prefixes', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{64}', args.binary_sha256):
        raise ValueError('invalid binary hash')
    output = args.output.absolute()
    if output.exists() or output.is_symlink() or not output.resolve().is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('new output directory under reports required')
    binary, prefixes, reference, artifact = (p.resolve() for p in (args.binary, args.prefixes, args.reference, args.artifact))
    watched = {binary: args.binary_sha256, prefixes: PREFIX_SHA, reference: REFERENCE_SHA,
               artifact / 'manifest.json': MANIFEST_SHA, Path(__file__).resolve(): digest(Path(__file__).resolve())}
    for path, sha in watched.items():
        if digest(path) != sha:
            raise ValueError(f'input identity differs: {path}')
    prefix_data = json.loads(prefixes.read_bytes())['prefixes']
    reference_data = json.loads(reference.read_bytes())
    # Validate all batches before starting the GPU workload.
    plan = [(start, *expected_batch(prefix_data, reference_data, start)) for start in range(0, 3003, 4)]
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for start, inputs, expected in plan:
        input_path = output / f'input-{start:04}.json'
        with input_path.open('x') as stream:
            json.dump(inputs, stream, separators=(',', ':'))
            stream.write('\n')
        command = [str(binary), 'sample', str(artifact), 'cuda', str(input_path), '--cfg', '1']
        result = subprocess.run(command, capture_output=True, timeout=900, check=False)
        (output / f'output-{start:04}.json').write_bytes(result.stdout)
        (output / f'error-{start:04}.txt').write_bytes(result.stderr)
        if result.returncode:
            raise RuntimeError(f'CLI exited {result.returncode} at row {start}')
        check_output(result.stdout, expected)
        records.append({'start': start, 'rows': len(inputs), 'stdout_sha256': hashlib.sha256(result.stdout).hexdigest()})
        if len(records) % 50 == 0 or start == 3000:
            print(f'{start + len(inputs)}/3003', flush=True)
    for path, sha in watched.items():
        if digest(path) != sha:
            raise ValueError(f'input changed during capture: {path}')
    with (output / 'summary.json').open('x') as stream:
        json.dump({'scope': 'nonblind unified CUDA CFG=1 CLI output equivalence, not performance',
                   'rows': sum(record['rows'] for record in records), 'batches': records,
                   'watched_sha256': {str(path): sha for path, sha in watched.items()}}, stream, indent=2, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    main()
