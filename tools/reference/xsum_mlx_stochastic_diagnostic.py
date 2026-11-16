"""Describe independent-stream Metal/Python XSum differences; not a quality test."""
import argparse
import json
from pathlib import Path

from safetensors.torch import load_file
from xsum_artifact import digest_file
from xsum_mlx_stochastic_score import DATA, DATA_SHA, ROOT, read_metal
from xsum_stochastic_followup_capture import validate_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metal', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--python-score', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    published_path = ROOT / 'docs/results/2026-11-15-phase4-xsum/stochastic-followup-result-v1.json'
    if digest_file(published_path) != '4c788e0509a40523a4e1a3b789843f817ee16465470014598150b4a3cb9c79eb':
        raise ValueError('published follow-up changed')
    published = json.loads(published_path.read_bytes())
    if digest_file(DATA) != DATA_SHA:
        raise ValueError('dataset changed')
    data = json.loads(DATA.read_bytes())
    manifest, metal_ids, files = read_metal(args.metal, data)
    seed = manifest['seed']
    if digest_file(args.python_score) != published['python_score_sha256'][seed - 4]:
        raise ValueError('pinned Python score differs')
    score = json.loads(args.python_score.read_bytes())
    if digest_file(args.python / 'manifest.json') != score['manifest_sha256']:
        raise ValueError('Python capture manifest differs')
    differences = 0
    differing_rows = []
    for batch in range(64):
        start = 512 + batch * 32
        prefixes = [(row[:512] + [50257] * 512)[:512] for row in data['inputs'][start:start + 32]]
        validate_batch(args.python, 'python', batch, seed * 1000 + batch, prefixes)
        trace = load_file(args.python / f'reference-b{batch}' / 'trace.safetensors')
        for offset, row in enumerate(trace['generator.penalized_ids'].tolist()):
            index = batch * 32 + offset
            count = sum(a != b for a, b in zip(metal_ids[index], row))
            differences += count
            if count:
                differing_rows.append(start + offset)
    if digest_file(DATA) != DATA_SHA or any(digest_file(Path(p)) != h for p, h in files.items()):
        raise ValueError('capture or dataset changed during diagnostic')
    result = {'seed': seed, 'rows': 2048, 'different_ids': differences,
              'different_rows': len(differing_rows), 'python_score_sha256': digest_file(args.python_score),
              'metal_manifest_sha256': digest_file(args.metal / 'manifest.json'),
              'scope': 'independent stochastic streams; differences are not a quality or parity failure'}
    with args.output.open('x') as out:
        json.dump(result, out, indent=2, sort_keys=True); out.write('\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
