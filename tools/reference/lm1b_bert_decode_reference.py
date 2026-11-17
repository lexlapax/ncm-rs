"""Capture pinned BERT decoding for LM1B FLM scoring development cases."""
import argparse
import importlib.metadata
import json
from pathlib import Path

from transformers import AutoTokenizer
from convert_tokenizers import VERSIONS
from wmt14_tokenizer_inputs import ROOT, digest_file, verify_input

MANIFEST_SHA = '8042d9e1f380867272ee1b9d296ed060f19cacfecf249239c5332cda9c934efd'
CAPTURES = {
    11: '123ce052e32ebac1981321a72d728c7b16c876be5c8a118dd4668b0dc917760a',
    12: '72f802a08efcae3512b6e9fdc9a70c2044c9092b71397ad0a0e265dd440e88c3',
    13: '4a3683e8c99c459d5a02f468c91c0121067b12ce85efa96a86453958708ffc36',
    14: 'c0c438eaf647d7e9ca9ee9cf70b2d04b2ee9fae8faafeadfc3d7a1cbc04caa85',
    15: '3a07c39863262f745ad61d4cbb170ec081f93c7988d960afbc352eadd202cb26',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.resolve().is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('output must be new and under reports')
    manifest_path = ROOT / 'artifacts/canonical/tokenizers-qualified/bert/manifest.json'
    source = ROOT / 'artifacts/upstream/tokenizers-v1/bert'
    if digest_file(manifest_path) != MANIFEST_SHA:
        raise ValueError('tokenizer manifest differs')
    manifest = json.loads(manifest_path.read_bytes())
    if manifest['profile'] != 'bert' or manifest['reference_versions'] != VERSIONS:
        raise ValueError('unexpected tokenizer manifest')
    for name, version in VERSIONS.items():
        if importlib.metadata.version(name) != version:
            raise ValueError('tokenizer dependency version differs')
    watched = [Path(__file__).resolve(), Path(__file__).with_name('lm1b_dev_protocol.py').resolve(), manifest_path]
    for name, item in manifest['source']['files'].items():
        path = source / name
        verify_input(path, item['bytes'], item['sha256'])
        watched.append(path)
    rows = []
    for seed, sha in CAPTURES.items():
        path = ROOT / f'reports/phase4-lm1b-small-dev-seed-{seed}-v1.json'
        if digest_file(path) != sha:
            raise ValueError('capture identity differs')
        capture = json.loads(path.read_bytes())
        if (capture['size'], capture['seed'], capture['count'], capture['batch_size']) != ('small', seed, 256, 8):
            raise ValueError('capture metadata differs')
        for ids in capture['raw_ids']:
            if len(ids) != 128 or any(type(x) is not int or not 0 <= x < 30528 for x in ids):
                raise ValueError('invalid model output')
            rows.append({'kind': f'development-{seed}', 'ids': ids})
        if len(capture['raw_ids']) != 256:
            raise ValueError('invalid capture count')
        watched.append(path)
    synthetic = [[], [0, 100, 101, 102, 103], [101, 30522, 102],
                 list(range(30522, 30528)), [30521, 30522, 30527],
                 [101, 0, 101, 102, 0, 103], [7592, 1010, 2088, 999],
                 [101, 7592, 30525, 102, 0, 100, 103, 30527]]
    rows.extend({'kind': 'synthetic', 'ids': ids} for ids in synthetic)
    before = {str(path): digest_file(path) for path in watched}
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    if len(tokenizer) != 30522 or tokenizer.clean_up_tokenization_spaces:
        raise ValueError('unexpected tokenizer behavior')
    decoded = tokenizer.batch_decode([row['ids'] for row in rows])
    for row, text in zip(rows, decoded, strict=True):
        row['text'] = text
    if before != {str(path): digest_file(path) for path in watched}:
        raise ValueError('inputs changed during decoding')
    result = {'scope': 'LM1B FLM scorer BERT batch_decode; development and synthetic only',
              'manifest_sha256': MANIFEST_SHA, 'capture_sha256': CAPTURES, 'rows': rows}
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'rows': len(rows), 'reserved_development_ids': sum(x >= 30522 for row in rows[:1280] for x in row['ids']),
                      'sha256': digest_file(args.output)}, sort_keys=True))


if __name__ == '__main__':
    main()
