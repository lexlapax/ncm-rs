"""Create a pinned TinyStories text-input differential reference, without weights."""
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from convert_tokenizers import VERSIONS

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = 'c70274273f489abc73832b2b98584a8483b6b203'
SOURCE_HASHES = {
    'tokenization/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    'tokenization/tokenizer.py': 'e527d47e4ed8634c673d48a872a07c9cc8828d30988331de2d4bd3b7c21ddcb7',
    'tokenization/bpe_tokenizer.py': '5df2bf286b36cdc3c428aaf347c56a98c0abb3da65c8ff734582b1d59959ba2c',
}
PAYLOAD_SHA = 'fdf961a269fa4506a72d12056277450aaed13d31392b39fd640ff6db546516b4'
MANIFEST_SHA = '0f9da8645224160f1fdadeffb14cccede95a89948db12599d24865f1a859c6bb'


def checked(path, expected):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'not a regular input: {path}')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f'input hash differs: {path}')
    return data


def preflight(upstream, payload, manifest):
    if subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != UPSTREAM:
        raise ValueError('upstream revision differs')
    if subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip():
        raise ValueError('upstream checkout is not clean')
    for name, digest in SOURCE_HASHES.items():
        checked(upstream / name, digest)
    for name, version in VERSIONS.items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'wrong {name} version')
    checked(manifest, MANIFEST_SHA)
    return checked(payload, PAYLOAD_SHA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--tokenizer-asset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    payload = args.tokenizer_asset / 'tokenizer.json'
    manifest = args.tokenizer_asset / 'manifest.json'
    raw = preflight(upstream, payload, manifest)
    data = json.loads(raw)
    if len(data['vocab_hex']) != 8192:
        raise ValueError('unexpected vocabulary')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(upstream))
    from tokenization.bpe_tokenizer import BPETokenizer  # pylint: disable=import-outside-toplevel
    tokenizer = BPETokenizer({i: bytes.fromhex(value) for i, value in enumerate(data['vocab_hex'])},
                             [tuple(bytes.fromhex(value) for value in pair) for pair in data['merges_hex']],
                             {'eos_token': '<|endoftext|>', 'pad_token': '<|pad|>'})
    texts = [tokenizer.decode([i], skip_special_tokens=False) for i in range(8192)]
    texts += ['', 'hello world', '😀', 'a', 'hello', '<|endoftext|>', '<|pad|>']
    rows = [{'text': text, 'ids': tokenizer.encode(text)} for text in texts]
    if preflight(upstream, payload, manifest) != raw:
        raise ValueError('inputs changed during capture')
    result = {'scope': 'decoded-vocabulary and hand-picked texts; not generation quality',
              'upstream': UPSTREAM, 'payload_sha256': PAYLOAD_SHA, 'rows': rows}
    encoded = (json.dumps(result, ensure_ascii=True, separators=(',', ':')) + '\n').encode()
    with args.output.open('xb') as output:
        output.write(encoded)
    accepted = sum(len(row['ids']) == 1 and row['ids'][0] != 1 and row['ids'][0] < 8192 for row in rows)
    print(json.dumps({'rows': len(rows), 'accepted': accepted,
                      'sha256': hashlib.sha256(encoded).hexdigest()}, sort_keys=True))


if __name__ == '__main__':
    main()
