"""Compare CUDA --text generation with retained ID-input quality captures.

This is an output-equivalence retest on observed data, not a quality experiment or
performance benchmark. All inputs and the output parent must be trusted and stable.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BINARY_SHA = '6194111fe3860584492676c6d14f50124a77f8dcf18f316a959b8abd16a20278'
PROFILES = {
    'xsum': {'rows': 11334, 'width': 512, 'code_width': 200, 'id_width': 48,
             'reference_sha': 'f51a1232bcd47169cd970edd61b597b5aebef12f0f202e6d51f41be6642faa2f',
             'artifact': 'xsum-inference-rope-v2-final', 'device': 'cuda-cudnn',
             'manifest_sha': '8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4'},
    'wmt14': {'rows': 3003, 'width': 96, 'code_width': 313, 'id_width': 64,
              'reference_sha': '4c1d6b68e89b46800219e9099116b1a50d51b2c96a51bdc8abe6dec6a98add9e',
              'artifact': 'wmt14-inference-v1-final', 'device': 'cuda',
              'manifest_sha': '01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81'},
}
XSUM_SUMMARY_SHA = '6f202f51f1d6ea090af7de883f222e325a8badeffb0a23fa99e797b93cfdfa93'
WMT_CAPTURE_SHA = 'd26b8a3b16a4ad07565f52d3e61f22b30e2512ea10348e674a24e9ed52fecc79'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_checked(path, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError(f'identity differs: {path}')
    return raw


def batches(name):
    if name == 'xsum':
        return [(shard, ordinal, list(range(shard + ordinal * 64, min(11334, shard + (ordinal + 1) * 64), 2)))
                for shard in (0, 1) for ordinal in range(178)]
    if name == 'wmt14':
        return [(0, start // 4, list(range(start, min(start + 4, 3003)))) for start in range(0, 3003, 4)]
    raise ValueError('unsupported profile')


def validate_reference(reference, profile):
    if not isinstance(reference, dict) or set(reference) != {'texts', 'ids'}:
        raise ValueError('invalid reference')
    texts, ids = reference['texts'], reference['ids']
    if not isinstance(texts, list) or not isinstance(ids, list) or len(texts) != profile['rows'] or len(ids) != len(texts):
        raise ValueError('invalid reference count')
    if any(not isinstance(text, str) for text in texts) or any(
        not isinstance(row, list) or len(row) != profile['width'] or
        any(type(value) is not int or value < 0 for value in row) for row in ids
    ):
        raise ValueError('invalid reference rows')


def validate_output(output, expected, count, profile):
    if not isinstance(output, dict) or set(output) != {'codes', 'ids'} or output != expected:
        raise ValueError('generation differs from retained ID output')
    for key, width in (('codes', profile['code_width']), ('ids', profile['id_width'])):
        rows = output[key]
        if not isinstance(rows, list) or len(rows) != count or any(
            not isinstance(row, list) or len(row) != width or
            any(type(value) is not int or value < 0 for value in row) for row in rows
        ):
            raise ValueError(f'invalid {key} output')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=tuple(PROFILES), required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('output exists')
    profile = PROFILES[args.profile]
    plan = batches(args.profile)
    stop = len(plan) if args.stop is None else args.stop
    if not 0 <= args.start < stop <= len(plan):
        raise ValueError('invalid batch range')
    candidate, reference = args.candidate.resolve(), args.reference.resolve()
    artifact = (args.artifacts / profile['artifact']).resolve()
    watched = {candidate: BINARY_SHA, reference: profile['reference_sha'],
               artifact / 'manifest.json': profile['manifest_sha'], Path(__file__).resolve(): digest(Path(__file__).read_bytes())}
    ref = json.loads(read_checked(reference, profile['reference_sha']))
    validate_reference(ref, profile)
    expected_batches = []
    if args.profile == 'xsum':
        summary_path = ROOT / 'reports/phase5-xsum-cli-corpus-cuda-v1/summary.json'
        summary = json.loads(read_checked(summary_path, XSUM_SUMMARY_SHA))
        if not summary['codes_and_ids_match'] or summary['rows'] != profile['rows'] or summary['identities']['binary'] != BINARY_SHA:
            raise ValueError('invalid prior CLI comparison')
        watched[summary_path] = XSUM_SUMMARY_SHA
        for shard, ordinal, indices in plan[args.start:stop]:
            path = ROOT / f'reports/phase5-xsum-rust-quality-v1-s{shard}/batch-{ordinal:05}.json'
            sha = summary['shard_file_sha256'][str(path)]
            envelope = json.loads(read_checked(path, sha))
            record = envelope['record']
            if record['ordinal'] != ordinal or record['indices'] != indices:
                raise ValueError('XSum capture batch identity differs')
            expected_batches.append({'codes': record['codes'], 'ids': record['raw_ids']})
            watched[path] = sha
    else:
        path = ROOT / 'reports/phase5-wmt14-fast-cuda-full-v1/rust.json'
        capture = json.loads(read_checked(path, WMT_CAPTURE_SHA))
        if capture['count'] != profile['rows'] or len(capture['outputs']) != profile['rows']:
            raise ValueError('WMT14 capture count differs')
        for _, _, indices in plan[args.start:stop]:
            rows = [capture['outputs'][i] for i in indices]
            if [row['row'] for row in rows] != indices:
                raise ValueError('WMT14 row identity differs')
            expected_batches.append({'codes': [row['codes'] for row in rows], 'ids': [row['raw_ids'] for row in rows]})
        watched[path] = WMT_CAPTURE_SHA
    for path, sha in watched.items():
        read_checked(path, sha)
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for offset, ((shard, ordinal, indices), expected) in enumerate(zip(plan[args.start:stop], expected_batches, strict=True)):
        number = args.start + offset
        input_path = args.output / f'input-{number:04}.json'
        with input_path.open('x', encoding='utf-8') as stream:
            json.dump([ref['texts'][i] for i in indices], stream, ensure_ascii=True, separators=(',', ':'))
            stream.write('\n')
        command = [str(candidate), 'sample', str(artifact), profile['device'], str(input_path), '--text']
        process = subprocess.run(command, capture_output=True, timeout=900, check=False)
        (args.output / f'output-{number:04}.json').write_bytes(process.stdout)
        (args.output / f'error-{number:04}.txt').write_bytes(process.stderr)
        if process.returncode:
            raise RuntimeError(f'batch {number} exited {process.returncode}')
        validate_output(json.loads(process.stdout), expected, len(indices), profile)
        results.append({'batch': number, 'shard': shard, 'ordinal': ordinal, 'rows': len(indices),
                        'output_sha256': digest(process.stdout)})
        if (offset + 1) % 25 == 0 or number + 1 == stop:
            print(args.profile, number + 1, '/', stop, flush=True)
    for path, sha in watched.items():
        read_checked(path, sha)
    with (args.output / 'summary.json').open('x', encoding='utf-8') as stream:
        json.dump({'profile': args.profile, 'start': args.start, 'stop': stop,
                   'total_batches': len(plan), 'rows': sum(row['rows'] for row in results),
                   'scope': 'retained ID capture vs CUDA text generation; not quality or performance',
                   'watched_sha256': {str(path): sha for path, sha in watched.items()}, 'batches': results},
                  stream, indent=2, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    main()
