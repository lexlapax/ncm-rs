"""Make pinned Python references for the Phase 5 corpus text-input gate."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from transformers import AutoTokenizer

from convert_tokenizers import VERSIONS

ROOT = Path(__file__).resolve().parents[2]
PINS = {
    'xsum': ('cba412c100c1264a7a107b2df3bacd20bff5846325ce48fd5aa353b4726daa6a',
             'c98e2a229ee71ff5031914d69fa1f9d093e60f62cc60f5ff1d8f12a6d2a0786b',
             'xsum-inference-rope-v2-final', 'gpt2', 11334),
    'wmt14': ('36035390ce542834f8dfeeac004c1b532aff247a4ff39910f4a16efd009ca47c',
              'e1f881e225d651271d01d4067b3308578eb836703fdd25cfb17e7a1b95fcd69a',
              'wmt14-inference-v1-final', 'qwen3', 3003),
}


def verified(path, digest, size=None):
    if path.is_symlink() or not path.is_file() or (size is not None and path.stat().st_size != size):
        raise ValueError(f'invalid input: {path}')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f'wrong SHA-256: {path}')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', choices=PINS)
    parser.add_argument('output', type=Path, help='new JSON output file')
    args = parser.parse_args()
    data_hash, manifest_hash, artifact, source_name, count = PINS[args.profile]
    for package, version in VERSIONS.items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f'wrong {package} version')
    data_path = ROOT / 'artifacts/canonical/text-phase4-v1' / (args.profile + '.json')
    manifest_path = ROOT / 'artifacts/canonical' / artifact / 'tokenizer-manifest.json'
    manifest = json.loads(verified(manifest_path, manifest_hash))
    source = ROOT / 'artifacts/upstream/tokenizers-v2' / source_name
    for name, record in manifest['source']['files'].items():
        verified(source / name, record['sha256'], record['bytes'])
    data = json.loads(verified(data_path, data_hash))
    inputs = data['inputs']
    if data['profile'] != args.profile or len(inputs) != count:
        raise ValueError('unexpected corpus')
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    if args.profile == 'xsum':
        texts, ids = [], []
        for row in inputs:
            if len(row) != 560:
                raise ValueError('unexpected XSum row length')
            expected = row[:512]
            unpadded = expected[:]
            while unpadded and unpadded[-1] == 50257:
                unpadded.pop()
            texts.append(tokenizer.decode(unpadded, skip_special_tokens=False,
                                          clean_up_tokenization_spaces=False))
            ids.append(expected)
    else:
        texts = inputs
        ids = []
        for text in texts:
            row = tokenizer(text)['input_ids']
            ids.append((row[:96] + [151643] * 96)[:96])
    for name, record in manifest['source']['files'].items():
        verified(source / name, record['sha256'], record['bytes'])
    verified(manifest_path, manifest_hash)
    verified(data_path, data_hash)
    raw = (json.dumps({'texts': texts, 'ids': ids}, ensure_ascii=True, separators=(',', ':')) + '\n').encode()
    with args.output.open('xb') as output:
        output.write(raw)
    print(json.dumps({'rows': len(texts), 'sha256': hashlib.sha256(raw).hexdigest()}, sort_keys=True))


if __name__ == '__main__':
    main()
