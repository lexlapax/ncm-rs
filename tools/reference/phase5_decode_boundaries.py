"""Generate pinned Python XSum/WMT14 decoder references for synthetic ID rows."""
import argparse
import hashlib
import importlib.metadata
import json
import random
from pathlib import Path

POLICIES = {'xsum': (50256, 50257, 50304), 'wmt14': (151645, 151643, 151680)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', choices=POLICIES)
    parser.add_argument('output', type=Path)
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
    eos, pad, model_vocab = POLICIES[args.profile]
    rng = random.Random(0x5A17)
    rows = [[], [eos], [pad], [pad, 42, eos, 43], [model_vocab - 1, 42],
            [42, eos, model_vocab - 1], [42, pad, 43]]
    rows += [[rng.randrange(model_vocab) for _ in range(32)] for _ in range(128)]
    texts = []
    for row in rows:
        end = row.index(eos) if eos in row else len(row)
        texts.append(tokenizer.decode([item for item in row[:end] if item not in (eos, pad)],
                                      skip_special_tokens=True))
    for name, record in manifest['source']['files'].items():
        verified(source / name, record['sha256'], record['bytes'])
    verified(manifest_path, manifest_hash)
    raw = (json.dumps({'profile': args.profile, 'ids': rows, 'texts': texts},
                      ensure_ascii=True, separators=(',', ':')) + '\n').encode()
    with args.output.open('xb') as output:
        output.write(raw)
    print(json.dumps({'rows': len(rows), 'sha256': hashlib.sha256(raw).hexdigest()}, sort_keys=True))


if __name__ == '__main__':
    main()
