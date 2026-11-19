"""Export pinned upstream LM1B-small rotary tables for an explicit diagnostic.

This is an offline probe input, not a runtime artifact or replacement oracle.
"""
import argparse
import hashlib
import sys
from pathlib import Path

UPSTREAM_COMMON_SHA256 = '41b7c2c653d7ba312d583caca878ea3708d83da1a770b0b39651dbfd19754b73'


def write_exclusive(path, payload):
    # Exclusive creation refuses existing files and symlinks, even after preflight.
    with path.open('xb') as output:
        output.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('output already exists')
    common = args.upstream / 'models' / 'common.py'
    if hashlib.sha256(common.read_bytes()).hexdigest() != UPSTREAM_COMMON_SHA256:
        raise ValueError('unexpected upstream rotary implementation')
    import torch  # pylint: disable=import-outside-toplevel
    from safetensors.torch import save  # pylint: disable=import-outside-toplevel

    sys.path.insert(0, str(args.upstream))
    from models.common import precompute_rope_cos_sin  # pylint: disable=import-outside-toplevel

    torch.set_num_threads(1)
    cosine, sine = precompute_rope_cos_sin(torch.arange(128), 64, 10000)
    write_exclusive(args.output, save({'cosine': cosine[:, :32].contiguous(),
                                       'sine': sine[:, :32].contiguous()}))
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
