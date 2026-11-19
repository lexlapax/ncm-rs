"""Generate pinned upstream TinyStories references for synthetic ID sequences."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--tokenizer-asset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('output already exists')
    from phase5_tinystories_text_reference import preflight  # pylint: disable=import-outside-toplevel

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
    vocab = {i: bytes.fromhex(value) for i, value in enumerate(data['vocab_hex'])}
    tokenizer = BPETokenizer(vocab,
                             [tuple(bytes.fromhex(value) for value in pair) for pair in data['merges_hex']],
                             {'eos_token': '<|endoftext|>', 'pad_token': '<|pad|>'})
    single_bytes = {value[0]: key for key, value in vocab.items() if len(value) == 1}
    emoji = [single_bytes[value] for value in (0xf0, 0x9f, 0x98, 0x80)]
    rng = random.Random(0x5A17)
    rows = [[], [0], [1], [1, 99, 0, 100], emoji, emoji[:2], emoji[2:],
            [emoji[0], 1, *emoji[1:]], [emoji[0], 0, *emoji[1:]]]
    rows += [[rng.randrange(8192) for _ in range(128)] for _ in range(128)]
    texts = []
    for row in rows:
        end = row.index(0) if 0 in row else len(row)
        texts.append(tokenizer.decode([item for item in row[:end] if item not in (0, 1)],
                                      skip_special_tokens=True))
    if preflight(upstream, payload, manifest) != raw:
        raise ValueError('inputs changed during capture')
    encoded = (json.dumps({'ids': rows, 'texts': texts}, ensure_ascii=True,
                          separators=(',', ':')) + '\n').encode()
    with args.output.open('xb') as output:
        output.write(encoded)
    print(json.dumps({'rows': len(rows), 'sha256': hashlib.sha256(encoded).hexdigest()}, sort_keys=True))


if __name__ == '__main__':
    main()
