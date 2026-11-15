"""Convert pinned XSum/WMT14 evaluation pickles to checked non-executable JSON."""
import argparse
import hashlib
import io
import json
import pickle
import pickletools
from pathlib import Path

REVISION = '6580ba41a4b0a976dde876fbe84d25327617b0e6'
INPUTS = {
    'xsum': {
        'passages.pickle': (18506302, '39b820f0a71ddd6ad178b2700eeae6d035729053b1b2f115d3a92810c6ebf4f8'),
        'reference_summaries.pickle': (1456199, 'db624e065b6685dfea25db9106752e837f06f26b799763593759efd150fe554a'),
    },
    'wmt14': {
        'sources.pickle': (406100, '62254653c32ce9e8b93576797597381babaab301b10ee0a2a3babec89576ea43'),
        'reference_translations.pickle': (366263, '3dcef81d8aeded13f79917cd6b794d2ca9f33223bf35c1fff1c5643eae0df52b'),
    },
}
ALLOWED = {'PROTO', 'FRAME', 'EMPTY_LIST', 'MEMOIZE', 'MARK', 'BININT1', 'BININT2',
           'APPENDS', 'STOP', 'SHORT_BINUNICODE', 'BINUNICODE'}


class Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError('global references are forbidden')

    def persistent_load(self, pid):
        raise pickle.UnpicklingError('persistent references are forbidden')


def read(path, expected_size, expected_hash):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != expected_size:
        raise ValueError('input must be a regular file of the pinned size')
    raw = path.read_bytes()
    if len(raw) != expected_size or hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError('input hash differs')
    if any(op.name not in ALLOWED for op, _, _ in pickletools.genops(raw)):
        raise ValueError('unexpected pickle opcode')
    source = io.BytesIO(raw)
    value = Restricted(source).load()
    if source.tell() != len(raw) or type(value) is not list:
        raise ValueError('unexpected pickle structure or trailing bytes')
    return value


def convert(root, output, profile):
    files = INPUTS[profile]
    loaded = [read(root / profile / name, *record) for name, record in files.items()]
    inputs, references = loaded
    count = 11334 if profile == 'xsum' else 3003
    if len(inputs) != count or len(references) != count:
        raise ValueError('unexpected evaluation row count')
    if any(type(value) is not str or not 1 <= len(value) <= 4096 for value in references):
        raise ValueError('invalid reference text')
    if profile == 'xsum':
        if any(type(row) is not list or len(row) != 560
               or any(type(value) is not int or not 0 <= value <= 50257 for value in row)
               for row in inputs):
            raise ValueError('invalid XSum token rows')
    elif any(type(value) is not str or not 1 <= len(value) <= 4096 for value in inputs):
        raise ValueError('invalid WMT14 source text')
    result = {'schema_version': 1, 'dataset_revision': REVISION, 'profile': profile,
              'source_sha256': {name: record[1] for name, record in files.items()},
              'inputs': inputs, 'references': references}
    with output.open('x') as file:
        json.dump(result, file, ensure_ascii=True, separators=(',', ':'))
        file.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('profile', choices=tuple(INPUTS))
    args = parser.parse_args()
    convert(args.source, args.output, args.profile)
