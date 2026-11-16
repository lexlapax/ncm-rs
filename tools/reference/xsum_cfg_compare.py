"""Compare native XSum guided output with a pinned Python trace."""
import argparse
import json
from pathlib import Path

from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    args = parser.parse_args()
    reference = load_file(args.reference)
    native = json.loads(args.native.read_text())
    results = {}
    for name, key, width in (('codes', 'generator.selected_codes', 200),
                             ('ids', 'generator.penalized_ids', 48)):
        expected = reference[key].tolist()
        actual = native[name]
        if (not expected or len(actual) != len(expected)
                or any(len(row) != width for row in expected + actual)):
            raise ValueError(f'invalid {name} shape')
        differences = sum(a != b for row, ref in zip(actual, expected)
                          for a, b in zip(row, ref))
        results[name] = {'elements': len(expected) * width, 'differences': differences}
    print(json.dumps(results, sort_keys=True))
    if any(result['differences'] for result in results.values()):
        raise SystemExit('guided output differs from Python')


if __name__ == '__main__':
    main()
