"""Capture WMT14-style decoding of synthetic sequences built from verified source IDs."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from transformers import AutoTokenizer
from convert_tokenizers import VERSIONS
from wmt14_tokenizer_inputs import MANIFEST_SHA, ROOT, verify_input

REFERENCE_SHA = '22c76d597ae51dffadf61661d6b7c4d4290fd14b8809bfa6f1959b5c668df93f'
EOS, PAD, SPECIAL = 151645, 151643, 151644


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    reference = ROOT / 'reports/phase4-wmt14-tokenizer-reference-v1.json'
    manifest_path = ROOT / 'artifacts/canonical/tokenizers-qualified/qwen3/manifest.json'
    source = ROOT / 'artifacts/upstream/tokenizers-v1/qwen3'
    if digest(reference) != REFERENCE_SHA or digest(manifest_path) != MANIFEST_SHA:
        raise ValueError('reference identity differs')
    manifest = json.loads(manifest_path.read_bytes())
    if manifest['profile'] != 'qwen3' or manifest['reference_versions'] != VERSIONS:
        raise ValueError('tokenizer identity differs')
    for name, version in VERSIONS.items():
        if importlib.metadata.version(name) != version:
            raise ValueError('tokenizer version differs')
    for name, record in manifest['source']['files'].items():
        verify_input(source / name, record['bytes'], record['sha256'])
    watched = [Path(__file__).resolve(), Path(__file__).with_name('wmt14_tokenizer_inputs.py').resolve(),
               Path(__file__).with_name('convert_tokenizers.py').resolve(), reference, manifest_path]
    before = {str(p): digest(p) for p in watched}
    rows = json.loads(reference.read_bytes())['rows']
    if len(rows) != 3003:
        raise ValueError('unexpected source rows')
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    if tokenizer.eos_token_id != EOS or tokenizer.pad_token_id != PAD or SPECIAL not in tokenizer.all_special_ids:
        raise ValueError('unexpected special tokens')
    records = []
    for row in rows:
        body = row[:64]
        midpoint = len(body) // 2
        for ids in (body, [PAD] + body[:midpoint] + [SPECIAL] + body[midpoint:] + [EOS, 1]):
            cut = ids.index(EOS) if EOS in ids else len(ids)
            decoded = tokenizer.decode([x for x in ids[:cut] if x not in (EOS, PAD)], skip_special_tokens=True)
            records.append({'ids': ids, 'text': decoded})
    for name, record in manifest['source']['files'].items():
        verify_input(source / name, record['bytes'], record['sha256'])
    if before != {str(p): digest(p) for p in watched}:
        raise ValueError('reference sources changed')
    with args.output.open('x') as stream:
        json.dump({'scope': 'synthetic source-derived sequences, not WMT14 model output',
                   'reference_sha256': REFERENCE_SHA, 'manifest_sha256': MANIFEST_SHA,
                   'source_sha256': before, 'rows': records}, stream, ensure_ascii=False, separators=(',', ':'))
        stream.write('\n')
    print(json.dumps({'rows': len(records), 'sha256': digest(args.output)}, sort_keys=True))


if __name__ == '__main__':
    main()
