"""Generate pinned Python references for synthetic XSum/WMT14 input boundaries."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

# Deliberately include empty, special-token, Unicode and substantially overlength inputs.
TEXTS = ('', 'hello', 'hello ' * 1100, ' \n' * 700,
         'café 東京 😀', '<|endoftext|>', '<|im_end|>')
LENGTHS = {'xsum': (512, 50257), 'wmt14': (96, 151643)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', choices=LENGTHS)
    parser.add_argument('output', type=Path, help='new JSON output file')
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('output already exists')
    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel
    from convert_tokenizers import VERSIONS  # pylint: disable=import-outside-toplevel
    from phase5_text_inputs import PINS, ROOT, verified  # pylint: disable=import-outside-toplevel

    for package, version in VERSIONS.items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f'wrong {package} version')
    _, manifest_hash, artifact, source_name, _ = PINS[args.profile]
    manifest_path = ROOT / 'artifacts/canonical' / artifact / 'tokenizer-manifest.json'
    manifest = json.loads(verified(manifest_path, manifest_hash))
    source = ROOT / 'artifacts/upstream/tokenizers-v2' / source_name
    for name, record in manifest['source']['files'].items():
        verified(source / name, record['sha256'], record['bytes'])
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    length, pad = LENGTHS[args.profile]
    unpadded = [tokenizer(text)['input_ids'] for text in TEXTS]
    if not any(len(row) > length for row in unpadded):
        raise ValueError('reference did not exercise truncation')
    if not any(len(row) < length for row in unpadded):
        raise ValueError('reference did not exercise padding')
    rows = [(row[:length] + [pad] * length)[:length] for row in unpadded]
    for name, record in manifest['source']['files'].items():
        verified(source / name, record['sha256'], record['bytes'])
    verified(manifest_path, manifest_hash)
    raw = (json.dumps({'profile': args.profile, 'texts': TEXTS, 'ids': rows,
                       'original_lengths': [len(row) for row in unpadded]},
                      ensure_ascii=True, separators=(',', ':')) + '\n').encode()
    with args.output.open('xb') as output:
        output.write(raw)
    print(json.dumps({'rows': len(rows), 'sha256': hashlib.sha256(raw).hexdigest()}, sort_keys=True))


if __name__ == '__main__':
    main()
