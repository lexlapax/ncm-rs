"""Export sanitized measured LM1B timings, bound to the published aggregate.

No inputs, outputs, PIDs or paths are exported. The historical shared-GPU runs
remain provisional and are not release performance qualification.
"""
import argparse
import json
import math
import statistics
from pathlib import Path

from export_phase5_tinystories_trials import sha

SUMMARY_SHA = '257ecb87e33299188270110574f666a7d363f58dc4b5cb8a175e803902f21545'
EXPECTED = {'small': (1, 4, 16), 'medium': (1, 4, 16), 'large': (1, 4, 8)}


def extract(summary, sources):
    runs = summary.get('runs')
    if not isinstance(runs, list) or len(runs) != len(sources) or len(runs) != 5:
        raise ValueError('expected five source runs')
    result = {size: {} for size in EXPECTED}
    for run, (manifest_bytes, result_bytes) in zip(runs, sources, strict=True):
        identities = run['input_sha256']
        if sha(manifest_bytes) != identities['manifest.json'] or sha(result_bytes) != identities['result.json']:
            raise ValueError('source differs from published input identity')
        manifest, capture = json.loads(manifest_bytes), json.loads(result_bytes)
        size, batches = manifest['size'], manifest['batches']
        if (size not in EXPECTED or run['size'] != size or not batches or set(map(str, batches)) != set(run['batches'])
                or len(set(batches)) != len(batches) or any(type(b) is not int or b not in EXPECTED[size] for b in batches)
                or manifest['warmups'] != 5 or manifest['trials'] != 30 or manifest['repetitions'] != 3):
            raise ValueError('unexpected run protocol')
        rows = capture['rows']
        if not isinstance(rows, list) or len(rows) != len(batches) * 6:
            raise ValueError('unexpected row count')
        for batch in batches:
            key = str(batch)
            if key in result[size]:
                raise ValueError('duplicate size/batch')
            result[size][key] = {}
            for runtime in ('rust', 'python'):
                selected = sorted((r for r in rows if r.get('batch') == batch and r.get('runtime') == runtime), key=lambda r: r['rep'])
                if len(selected) != 3 or [r['rep'] for r in selected] != [0, 1, 2]:
                    raise ValueError('missing or duplicated process')
                medians = run['batches'][key][runtime]['process_medians_ms']
                if len(medians) != 3:
                    raise ValueError('invalid process medians')
                records = []
                for row, expected in zip(selected, medians, strict=True):
                    values = row['seconds']
                    if (not isinstance(values, list) or len(values) != 30
                            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values)
                            or not math.isclose(statistics.median(values) * 1000, expected, rel_tol=1e-12, abs_tol=1e-10)):
                        raise ValueError('invalid timing or median')
                    records.append(values)
                result[size][key][runtime] = records
    if any(set(result[size]) != {str(batch) for batch in batches} for size, batches in EXPECTED.items()):
        raise ValueError('incomplete coverage')
    return {'schema_version': 1, 'scope': 'shared-GPU synchronized model timing; excludes load, not release qualification',
            'summary_sha256': SUMMARY_SHA, 'warmups_per_process': 5, 'trials_per_process': 30,
            'processes_per_runtime_batch': 3, 'sizes': result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--runs', type=Path, nargs=5, required=True, help='small, medium, then large batch 1, 4, 8')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('new output file required')
    summary = args.summary.read_bytes()
    if sha(summary) != SUMMARY_SHA:
        raise ValueError('published summary differs')
    paths = [(directory / 'manifest.json', directory / 'result.json') for directory in args.runs]
    sources = [(a.read_bytes(), b.read_bytes()) for a, b in paths]
    value = extract(json.loads(summary), sources)
    if args.summary.read_bytes() != summary or any(a.read_bytes() != raw_a or b.read_bytes() != raw_b
                                                   for (a, b), (raw_a, raw_b) in zip(paths, sources, strict=True)):
        raise ValueError('inputs changed during export')
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(payload)
    print(sha(payload.encode()))


if __name__ == '__main__':
    main()
