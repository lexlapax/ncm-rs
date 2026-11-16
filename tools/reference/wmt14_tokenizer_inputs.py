"""Capture pinned upstream Qwen source-tokenizer IDs for WMT14 evaluation inputs."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from transformers import AutoTokenizer
from convert_tokenizers import VERSIONS

ROOT = Path(__file__).resolve().parents[2]
DATA_SHA = '36035390ce542834f8dfeeac004c1b532aff247a4ff39910f4a16efd009ca47c'
MANIFEST_SHA = 'e1f881e225d651271d01d4067b3308578eb836703fdd25cfb17e7a1b95fcd69a'


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_input(path, size, expected):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != size or digest_file(path) != expected:
        raise ValueError(f'input type, size or SHA-256 differs: {path.name}')


def capture(data_path, source_dir, manifest_path):
    if digest_file(data_path) != DATA_SHA or digest_file(manifest_path) != MANIFEST_SHA:
        raise ValueError('dataset or tokenizer manifest differs')
    manifest = json.loads(manifest_path.read_bytes())
    if manifest['profile'] != 'qwen3' or manifest['reference_versions'] != VERSIONS:
        raise ValueError('unexpected tokenizer identity')
    for name, version in VERSIONS.items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'unexpected {name} version')
    for name, record in manifest['source']['files'].items():
        verify_input(source_dir / name, record['bytes'], record['sha256'])
    data = json.loads(data_path.read_bytes())
    if data['profile'] != 'wmt14' or len(data['inputs']) != 3003 or not all(
        isinstance(row, str) for row in data['inputs']
    ):
        raise ValueError('unexpected evaluation inputs')
    tokenizer = AutoTokenizer.from_pretrained(source_dir, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id != 151643:
        raise ValueError('unexpected source padding token')
    ids = [tokenizer(text)['input_ids'] for text in data['inputs']]
    # Upstream eval_shard.py pads/truncates with the VQ tokenizer's PAD ID.
    prefixes = [(row[:96] + [151643] * 96)[:96] for row in ids]
    for name, record in manifest['source']['files'].items():
        verify_input(source_dir / name, record['bytes'], record['sha256'])
    if digest_file(data_path) != DATA_SHA or digest_file(manifest_path) != MANIFEST_SHA:
        raise ValueError('input changed during capture')
    return {'dataset_sha256': DATA_SHA, 'tokenizer_manifest_sha256': MANIFEST_SHA,
            'source': 'pinned Transformers AutoTokenizer, local files only',
            'rows': ids, 'prefixes': prefixes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = capture(ROOT / 'artifacts/canonical/text-phase4-v1/wmt14.json',
                     ROOT / 'artifacts/upstream/tokenizers-v1/qwen3',
                     ROOT / 'artifacts/canonical/tokenizers-qualified/qwen3/manifest.json')
    with args.output.open('x') as stream:
        json.dump(report, stream, separators=(',', ':'))
        stream.write('\n')


if __name__ == '__main__':
    main()
