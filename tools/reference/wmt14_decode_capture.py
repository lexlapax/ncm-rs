"""Decode bounded WMT14 output captures with the pinned upstream tokenizer."""
import argparse
import importlib.metadata
import json
from pathlib import Path
from transformers import AutoTokenizer
from convert_tokenizers import VERSIONS
from wmt14_tokenizer_inputs import DATA_SHA, MANIFEST_SHA, ROOT, digest_file, verify_input

EOS, PAD = 151645, 151643


def validate_outputs(outputs, start, count):
    if type(start) is not int or type(count) is not int or start < 0 or count < 1 or start + count > 3003 or len(outputs) != count:
        raise ValueError('invalid capture range')
    for index, row in enumerate(outputs, start):
        if (type(row) is not dict or set(row) != {'row', 'codes', 'raw_ids'} or row['row'] != index or
                type(row['codes']) is not list or len(row['codes']) != 313 or
                any(type(x) is not int or not 0 <= x < 4096 for x in row['codes']) or
                type(row['raw_ids']) is not list or len(row['raw_ids']) != 64 or
                any(type(x) is not int or not 0 <= x < 151680 for x in row['raw_ids'])):
            raise ValueError('invalid captured output row')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--capture-sha256', required=True)
    args = p.parse_args()
    data_path = ROOT / 'artifacts/canonical/text-phase4-v1/wmt14.json'
    manifest_path = ROOT / 'artifacts/canonical/tokenizers-qualified/qwen3/manifest.json'
    source = ROOT / 'artifacts/upstream/tokenizers-v1/qwen3'
    if args.output.exists() or args.output.is_symlink() or digest_file(args.capture) != args.capture_sha256:
        raise ValueError('output exists or capture identity differs')
    if digest_file(data_path) != DATA_SHA or digest_file(manifest_path) != MANIFEST_SHA:
        raise ValueError('dataset or tokenizer identity differs')
    manifest = json.loads(manifest_path.read_bytes())
    if manifest['profile'] != 'qwen3' or manifest['reference_versions'] != VERSIONS:
        raise ValueError('unexpected tokenizer manifest')
    for name, version in VERSIONS.items():
        if importlib.metadata.version(name) != version:
            raise ValueError('tokenizer dependency version differs')
    for name, record in manifest['source']['files'].items():
        verify_input(source / name, record['bytes'], record['sha256'])
    capture = json.loads(args.capture.read_bytes())
    start, count = capture['start'], capture['count']
    validate_outputs(capture['outputs'], start, count)
    data = json.loads(data_path.read_bytes())
    if data['profile'] != 'wmt14' or len(data['references']) != 3003 or any(type(x) is not str for x in data['references']):
        raise ValueError('unexpected reference dataset')
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    if tokenizer.eos_token_id != EOS or tokenizer.pad_token_id != PAD:
        raise ValueError('unexpected tokenizer special IDs')
    hypotheses = []
    for row in capture['outputs']:
        ids = row['raw_ids']
        cut = ids.index(EOS) if EOS in ids else len(ids)
        hypotheses.append(tokenizer.decode([x for x in ids[:cut] if x not in (EOS, PAD)], skip_special_tokens=True))
    for name, record in manifest['source']['files'].items():
        verify_input(source / name, record['bytes'], record['sha256'])
    if digest_file(args.capture) != args.capture_sha256 or digest_file(data_path) != DATA_SHA or digest_file(manifest_path) != MANIFEST_SHA:
        raise ValueError('inputs changed during decoding')
    result = {'scope': 'pinned upstream WMT14 decode; scoring is separate', 'start': start, 'count': count,
              'capture_sha256': args.capture_sha256, 'dataset_sha256': DATA_SHA, 'tokenizer_manifest_sha256': MANIFEST_SHA,
              'hypotheses': hypotheses, 'references': data['references'][start:start + count]}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(f'decoded {count} rows; sha256 {digest_file(args.output)}')


if __name__ == '__main__':
    main()
