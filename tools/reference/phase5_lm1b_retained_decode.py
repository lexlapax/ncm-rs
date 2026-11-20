"""Decode one retained LM1B CUDA cohort with the pinned BERT evaluator."""
import argparse
import importlib.metadata
import json
from pathlib import Path

from convert_tokenizers import VERSIONS
ROOT = Path(__file__).resolve().parents[2]

MANIFEST = '8042d9e1f380867272ee1b9d296ed060f19cacfecf249239c5332cda9c934efd'
RECEIPTS = {
    'small': '742d4f25e6c85b2037cfd621a86e48948b54bff137e639894330702feac99506',
    'medium': '57f89148f93318ba7567bf32b7270d48867b8e719b8095855a24013bab1a20da',
    'large': '166c8e49c3189595c84eb952fe99584e6515ef095e2dbfd720c3628924876a3f',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('size', choices=tuple(RECEIPTS))
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.resolve().is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('output must be new and under reports')
    from wmt14_tokenizer_inputs import digest_file, verify_input
    manifest_path = ROOT / 'artifacts/canonical/tokenizers-qualified/bert/manifest.json'
    source = ROOT / 'artifacts/upstream/tokenizers-v1/bert'
    if digest_file(manifest_path) != MANIFEST:
        raise ValueError('tokenizer manifest differs')
    manifest = json.loads(manifest_path.read_bytes())
    if manifest['profile'] != 'bert' or manifest['reference_versions'] != VERSIONS:
        raise ValueError('unexpected tokenizer manifest')
    for name, version in VERSIONS.items():
        if importlib.metadata.version(name) != version:
            raise ValueError('tokenizer dependency version differs')
    directory = ROOT / f'reports/phase5-lm1b-{args.size}-cuda-retest-v1/rust'
    receipt_path = directory / 'receipt.json'
    if digest_file(receipt_path) != RECEIPTS[args.size]:
        raise ValueError('capture receipt differs')
    receipt = json.loads(receipt_path.read_bytes())
    if receipt['runtime'] != 'rust' or set(receipt['captures']) != {str(n) for n in range(2000, 2080)}:
        raise ValueError('invalid capture receipt')
    watched = [Path(__file__).resolve(), manifest_path, receipt_path]
    for name, item in manifest['source']['files'].items():
        path = source / name
        verify_input(path, item['bytes'], item['sha256'])
        watched.append(path)
    captures = []
    for seed in range(2000, 2080):
        path = directory / f'rust-seed-{seed}.json'
        if digest_file(path) != receipt['captures'][str(seed)]:
            raise ValueError(f'capture identity differs for seed {seed}')
        capture = json.loads(path.read_bytes())
        if (capture['size'], capture['seed'], capture['count'], capture['batch_size']) != (args.size, seed, 256, 8):
            raise ValueError('capture metadata differs')
        rows = capture['raw_ids']
        if len(rows) != 256 or any(len(row) != 128 or any(type(x) is not int or not 0 <= x < 30528 for x in row) for row in rows):
            raise ValueError('invalid model output')
        captures.append(rows)
        watched.append(path)
    before = {str(path): digest_file(path) for path in watched}
    if before[str(manifest_path)] != MANIFEST or any(
        before[str(source / name)] != item['sha256']
        for name, item in manifest['source']['files'].items()
    ):
        raise ValueError('tokenizer changed during preflight')
    if before[str(receipt_path)] != RECEIPTS[args.size] or any(
        before[str(directory / f'rust-seed-{seed}.json')] != receipt['captures'][str(seed)]
        for seed in range(2000, 2080)
    ):
        raise ValueError('capture changed during preflight')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    if len(tokenizer) != 30522 or tokenizer.clean_up_tokenization_spaces:
        raise ValueError('unexpected tokenizer behavior')
    texts = [tokenizer.batch_decode(rows) for rows in captures]
    if before != {str(path): digest_file(path) for path in watched}:
        raise ValueError('inputs changed during decoding')
    result = {'size': args.size, 'tokenizer_manifest_sha256': MANIFEST,
              'receipt_sha256': RECEIPTS[args.size], 'seeds': list(range(2000, 2080)), 'texts': texts}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'size': args.size, 'rows': sum(map(len, texts)), 'sha256': digest_file(args.output)}))


if __name__ == '__main__':
    main()
