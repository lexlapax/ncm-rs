"""Compare a CUDA CLI route against the frozen WMT14 CFG=1 corpus.

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
TEXT_SHA = '4c1d6b68e89b46800219e9099116b1a50d51b2c96a51bdc8abe6dec6a98add9e'
DECODE_SHA = '43392bd5c8011956f514a5f6bf3f8090d72a09dd3cd08b3027fc02d5c3e3b0d2'
DECODE_CAPTURE_SHA = 'ff04571bfc72be02aee45ef7f64b6b4226431dca2264a7fd02d30be44294ef0f'


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
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise ValueError('invalid CLI output')
    for key in ('codes', 'ids'):
        if not isinstance(actual[key], list) or any(
            not isinstance(row, list) or any(type(value) is not int for value in row)
            for row in actual[key]
        ):
            raise ValueError('invalid CLI output rows')
    if actual != expected:
        raise ValueError('CLI output differs from frozen reference')


def validate_text_reference(reference, prefixes):
    if (not isinstance(reference, dict) or set(reference) != {'texts', 'ids'}
            or reference['ids'] != prefixes or not isinstance(reference['texts'], list)
            or len(reference['texts']) != 3003 or any(not isinstance(s, str) for s in reference['texts'])):
        raise ValueError('text reference differs from frozen prefixes')
    return reference['texts']


def validate_decoded_reference(reference):
    if (not isinstance(reference, dict) or reference.get('capture_sha256') != DECODE_CAPTURE_SHA
            or reference.get('start') != 0 or reference.get('count') != 3003
            or not isinstance(reference.get('hypotheses'), list) or len(reference['hypotheses']) != 3003
            or any(not isinstance(s, str) for s in reference['hypotheses'])):
        raise ValueError('invalid pinned decoded reference')
    return reference['hypotheses']


def command_for(route, binary, artifact, input_path, *, text=False, decode=False):
    if route == 'unified':
        return [str(binary), 'sample', str(artifact), 'cuda', str(input_path), '--cfg', '1',
                *(['--text'] if text else []), *(['--decode'] if decode else [])]
    if route == 'alias' and not text and not decode:
        return [str(binary), 'sample-wmt14', str(artifact), 'cuda', str(input_path), '1']
    raise ValueError('unsupported route')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route', choices=('unified', 'alias'), default='unified')
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--binary-sha256', required=True)
    parser.add_argument('--prefixes', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--text-reference', type=Path, help='pinned WMT14 text/ID corpus, unified route only')
    parser.add_argument('--decode-reference', type=Path, help='pinned CFG=1 evaluator decoding, unified route only')
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.text_reference or args.decode_reference) and args.route != 'unified':
        raise ValueError('text or decode requires the unified route')
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
    text_data = None
    if args.text_reference:
        text_reference = args.text_reference.resolve()
        watched[text_reference] = TEXT_SHA
        if digest(text_reference) != TEXT_SHA:
            raise ValueError('text reference identity differs')
        text_data = validate_text_reference(json.loads(text_reference.read_bytes()), prefix_data)
    decoded = None
    if args.decode_reference:
        decode_reference = args.decode_reference.resolve()
        watched[decode_reference] = DECODE_SHA
        if digest(decode_reference) != DECODE_SHA:
            raise ValueError('decoded reference identity differs')
        decoded = validate_decoded_reference(json.loads(decode_reference.read_bytes()))
    reference_data = json.loads(reference.read_bytes())
    # Validate all batches before starting the GPU workload.
    plan = [(start, *expected_batch(prefix_data, reference_data, start)) for start in range(0, 3003, 4)]
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for start, inputs, expected in plan:
        input_path = output / f'input-{start:04}.json'
        with input_path.open('x') as stream:
            json.dump(text_data[start:start + len(inputs)] if text_data is not None else inputs,
                      stream, ensure_ascii=True, separators=(',', ':'))
            stream.write('\n')
        if decoded is not None:
            expected = {**expected, 'texts': decoded[start:start + len(inputs)]}
        command = command_for(args.route, binary, artifact, input_path,
                              text=text_data is not None, decode=decoded is not None)
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
        json.dump({'scope': 'nonblind CUDA CFG=1 CLI output equivalence, not performance',
                   'route': args.route, 'input_mode': 'text' if text_data is not None else 'ids',
                   'decode': decoded is not None,
                   'rows': sum(record['rows'] for record in records), 'batches': records,
                   'watched_sha256': {str(path): sha for path, sha in watched.items()}}, stream, indent=2, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    main()
