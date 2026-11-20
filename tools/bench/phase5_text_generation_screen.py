"""Compare selected CUDA text requests with equivalent pinned IDs and the old CLI.

This is a bounded output screen, not corpus-wide generation, quality or performance.
Inputs, artifacts, binaries and parent output directory must be trusted and stable.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BINARY_SHA = '6194111fe3860584492676c6d14f50124a77f8dcf18f316a959b8abd16a20278'
BASELINE_SHA = 'f10abaaec672dd7c0c62034972fb4b15fc204f2b1f5693e4a09f5e9640767098'
PROFILES = {
    'xsum': {'reference': 'f51a1232bcd47169cd970edd61b597b5aebef12f0f202e6d51f41be6642faa2f',
             'rows': 11334, 'width': 512, 'codes': 200, 'ids': 48, 'device': 'cuda-cudnn',
             'artifact': 'xsum-inference-rope-v2-final',
             'manifest': '8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4'},
    'wmt14': {'reference': '4c1d6b68e89b46800219e9099116b1a50d51b2c96a51bdc8abe6dec6a98add9e',
              'rows': 3003, 'width': 96, 'codes': 313, 'ids': 64, 'device': 'cuda',
              'artifact': 'wmt14-inference-v1-final',
              'manifest': '01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81'},
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked(path, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError(f'identity differs: {path}')
    return raw


def select(reference, profile):
    if not isinstance(reference, dict) or set(reference) != {'texts', 'ids'}:
        raise ValueError('invalid reference fields')
    texts, ids = reference['texts'], reference['ids']
    if (not isinstance(texts, list) or not isinstance(ids, list)
            or len(texts) != profile['rows'] or len(ids) != len(texts)):
        raise ValueError('invalid reference length')
    if any(not isinstance(text, str) for text in texts):
        raise ValueError('invalid text')
    if any(not isinstance(row, list) or len(row) != profile['width']
           or any(type(value) is not int or value < 0 for value in row) for row in ids):
        raise ValueError('invalid IDs')
    return [list(range(0, 1)), list(range(1234, 1238)), list(range(len(ids) - 32, len(ids)))]


def validate_pair(baseline, id_decoded, text_decoded, count, profile):
    raw, decoded, text = (json.loads(value) for value in (baseline, id_decoded, text_decoded))
    if set(raw) != {'codes', 'ids'} or set(decoded) != {'codes', 'ids', 'texts'}:
        raise ValueError('unexpected output fields')
    if id_decoded != text_decoded or decoded['codes'] != raw['codes'] or decoded['ids'] != raw['ids']:
        raise ValueError('outputs differ')
    for key in ('codes', 'ids'):
        rows = raw[key]
        if (not isinstance(rows, list) or len(rows) != count
                or any(not isinstance(row, list) or len(row) != profile[key]
                       or any(type(item) is not int or item < 0 for item in row) for row in rows)):
            raise ValueError(f'invalid {key} shape')
    if not isinstance(decoded['texts'], list) or len(decoded['texts']) != count or any(
        not isinstance(value, str) for value in decoded['texts']
    ):
        raise ValueError('invalid decoded text')
    if decoded != text:
        raise ValueError('decoded outputs differ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('output exists')
    baseline, candidate, refs, artifacts = (p.resolve() for p in
                                             (args.baseline, args.candidate, args.references, args.artifacts))
    watched = {baseline: BASELINE_SHA, candidate: BINARY_SHA, Path(__file__).resolve(): digest(Path(__file__).read_bytes())}
    if baseline == candidate:
        raise ValueError('binaries must be distinct')
    for name, profile in PROFILES.items():
        watched[refs / f'{name}.json'] = profile['reference']
        watched[artifacts / profile['artifact'] / 'manifest.json'] = profile['manifest']
    for path, sha in watched.items():
        checked(path, sha)
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    for name, profile in PROFILES.items():
        ref = json.loads(checked(refs / f'{name}.json', profile['reference']))
        results[name] = []
        for case, indices in enumerate(select(ref, profile)):
            inputs = {}
            for kind in ('ids', 'texts'):
                path = args.output / f'{name}-{case}-{kind}.json'
                raw = (json.dumps([ref[kind][i] for i in indices], ensure_ascii=True, separators=(',', ':')) + '\n').encode()
                with path.open('xb') as stream:
                    stream.write(raw)
                inputs[kind] = path
            art = artifacts / profile['artifact']
            common = [str(art), profile['device']]
            commands = [
                [str(baseline), f'sample-{name}', *common, str(inputs['ids'])],
                [str(candidate), 'sample', *common, str(inputs['ids']), '--decode'],
                [str(candidate), 'sample', *common, str(inputs['texts']), '--text', '--decode'],
            ]
            outputs = []
            for label, command in zip(('old', 'ids', 'text'), commands, strict=True):
                process = subprocess.run(command, capture_output=True, timeout=900, check=False)
                (args.output / f'{name}-{case}-{label}.stdout').write_bytes(process.stdout)
                (args.output / f'{name}-{case}-{label}.stderr').write_bytes(process.stderr)
                if process.returncode:
                    raise RuntimeError(f'{name} case {case} {label} exited {process.returncode}')
                outputs.append(process.stdout)
            validate_pair(*outputs, len(indices), profile)
            results[name].append({'indices': indices, 'old_sha256': digest(outputs[0]),
                                  'decoded_sha256': digest(outputs[1])})
            print(name, case, len(indices), flush=True)
    for path, sha in watched.items():
        checked(path, sha)
    with (args.output / 'result.json').open('x', encoding='utf-8') as stream:
        json.dump({'scope': 'selected CUDA text/ID fixed requests, not quality or performance',
                   'watched_sha256': {str(path): sha for path, sha in watched.items()}, 'results': results},
                  stream, indent=2, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    main()
