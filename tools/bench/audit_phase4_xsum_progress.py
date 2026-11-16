"""Read-only progress comparison; final scoring uses the committed comparator."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, 'tools/reference')
from xsum_quality_generate import read_record, shard_batches
from xsum_quality_rust import read_batch

ap = argparse.ArgumentParser()
ap.add_argument('--shard', type=int, choices=(0, 1), required=True)
args = ap.parse_args()
reference = Path('reports/phase4-xsum-quality-python-smoke-s0' if args.shard == 0
                 else 'reports/phase4-xsum-quality-python-dev-s1')
candidate = Path(f'reports/phase4-xsum-quality-rust-dev-v2-s{args.shard}')
paths = sorted(candidate.glob('batch-*.json'))
if [p.name for p in paths] != [f'batch-{i:05}.json' for i in range(len(paths))]:
    raise ValueError('candidate batch sequence has gaps')
batches, _ = shard_batches(args.shard, 11334)
rows = codes = ids = 0
first = []
for ordinal, path in enumerate(paths):
    indices = batches[ordinal]
    p = read_record(reference / path.name, indices, ordinal)
    r = read_batch(path, indices, ordinal)
    for index, pc, rc, pi, ri in zip(indices, p['codes'], r['codes'], p['raw_ids'], r['raw_ids']):
        dc = sum(a != b for a, b in zip(pc, rc))
        di = sum(a != b for a, b in zip(pi, ri))
        codes += dc; ids += di; rows += 1
        if (dc or di) and len(first) < 8: first.append({'index': index, 'codes': dc, 'ids': di})
print({'shard': args.shard, 'batches': len(paths), 'rows': rows,
       'code_differences': codes, 'id_differences': ids, 'first': first})
