"""Score Phase 3 Sudoku outputs and compare two complete raw-output files."""
import argparse
import hashlib
import json
from pathlib import Path


def text(ids):
    table = '_123456789|'
    if any(type(x) is not int or not 0 <= x <= 11 for x in ids):
        raise ValueError('invalid token id')
    return ''.join(table[x] for x in ids if x != 11)

def valid(grid):
    s = grid.replace('|', '')
    if len(s) != 81 or any(c not in '123456789' for c in s): return False
    rows = [s[i:i + 9] for i in range(0, 81, 9)]
    columns = [''.join(row[i] for row in rows) for i in range(9)]
    boxes = [''.join(rows[r][c] for r in range(br, br + 3) for c in range(bc, bc + 3)) for br in (0, 3, 6) for bc in (0, 3, 6)]
    return all(sorted(group) == list('123456789') for group in rows + columns + boxes)

def score(inputs, references, ids):
    if not (len(inputs) == len(references) == len(ids)) or not ids:
        raise ValueError('board counts differ or are empty')
    solved = valid_count = malformed = clues = 0
    cells = []
    for prefix, target, prediction in zip(inputs, references, ids, strict=True):
        if len(prefix) != 91 or len(target) != 89 or len(prediction) != 89:
            raise ValueError('token tensor shape mismatch')
        output, expected = text(prediction), text(target)
        solved += output == expected
        valid_count += valid(output)
        expected_format = len(output) == 89 and all(output[i] == '|' for i in range(9, 89, 10)) and all(c in '123456789' for c in output.replace('|', ''))
        malformed += not expected_format
        clues += expected_format and all(p == 0 or p == prediction[i] for i, p in enumerate(prefix[1:-1]))
        a, b = output.replace('|', ''), expected.replace('|', '')
        pairs = list(zip(a, b))
        if not pairs: raise ValueError('upstream cell metric undefined for empty output')
        cells.append(sum(x == y for x, y in pairs) / len(pairs))
    n = len(ids)
    return {'boards': n, 'solved': solved, 'solved_rate': solved / n,
            'cell_accuracy': sum(cells) / n, 'valid_grids': valid_count, 'valid_rate': valid_count / n,
            'malformed': malformed, 'clues_preserved': clues, 'clue_preservation_rate': clues / n}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset', type=Path); p.add_argument('rust', type=Path); p.add_argument('python', type=Path)
    a = p.parse_args()
    data = json.loads(a.dataset.read_bytes()); rust = json.loads(a.rust.read_bytes()); ref = json.loads(a.python.read_bytes())
    inputs, refs = data['inputs'], data['references']
    expected_input = (json.dumps(inputs, separators=(',', ':')) + '\n').encode()
    if ref.get('inputs_sha256') != hashlib.sha256(expected_input).hexdigest():
        raise ValueError('Python result input hash differs from dataset')
    if ref.get('difficulty') != data.get('difficulty'):
        raise ValueError('Python result difficulty differs from dataset')
    if not (len(inputs) == len(refs) == len(rust['ids']) == len(ref['ids']) == len(rust['codes']) == len(ref['codes']) == 2000):
        raise ValueError('full 2,000-board evaluation required')
    if any(len(row) != 322 for rows in (rust['codes'], ref['codes']) for row in rows): raise ValueError('invalid code length')
    result = {'difficulty': data['difficulty'], 'dataset_revision': data['dataset_revision'],
              'rust': score(inputs, refs, rust['ids']), 'python': score(inputs, refs, ref['ids']),
              'code_mismatched_boards': sum(x != y for x, y in zip(rust['codes'], ref['codes'], strict=True)),
              'id_mismatched_boards': sum(x != y for x, y in zip(rust['ids'], ref['ids'], strict=True)),
              'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (a.dataset, a.rust, a.python)}}
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == '__main__': main()
