"""Export sanitized TinyStories CUDA timing trials from the retained nonblind screen.

This preserves latency observations only, not outputs, paths, device peaks or
release qualification. Inputs and destination parents must be trusted/stable.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_SHA = '3aeb0f7d1ae9002fc15c959dd50417b14834629399715a4e6d77688ce09a375d'
SUMMARY_SHA = '0c0b360b4fb4129955762a0351c4d15f71874ec29c39e646c88457bbaca35ac3'
BATCHES = (1, 4, 32, 128)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_trials(doc, batch):
    values = doc.get('trials_seconds')
    if (doc.get('batch') != batch or doc.get('warmups') != 5 or not isinstance(values, list) or len(values) != 30
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values)):
        raise ValueError('invalid timing record')
    return values


def extract(manifest, summary, read_record):
    if manifest.get('batches') != list(BATCHES) or manifest.get('processes_per_runtime_batch') != 3:
        raise ValueError('invalid manifest')
    result = {}
    for batch in BATCHES:
        row = {}
        for runtime in ('python', 'rust'):
            records = []
            expected = summary['batches'][str(batch)][runtime]['process_medians_ms']
            if not isinstance(expected, list) or len(expected) != 3:
                raise ValueError('invalid summary')
            for rep in range(3):
                raw = read_record(runtime, batch, rep)
                values = validate_trials(json.loads(raw), batch)
                if not math.isclose(statistics.median(values) * 1000, expected[rep], rel_tol=1e-12, abs_tol=1e-10):
                    raise ValueError('record median differs from published summary')
                records.append({'trials_seconds': values, 'source_sha256': sha(raw)})
            row[runtime] = records
        result[str(batch)] = row
    return {'schema_version': 1, 'scope': 'shared-GPU synchronized model timing; excludes load, not release qualification',
            'warmups_per_process': 5, 'trials_per_process': 30, 'processes_per_runtime_batch': 3,
            'manifest_sha256': MANIFEST_SHA, 'summary_sha256': SUMMARY_SHA, 'batches': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('new output file required')
    manifest_bytes = (args.run / 'manifest.json').read_bytes()
    summary_bytes = args.summary.read_bytes()
    if sha(manifest_bytes) != MANIFEST_SHA or sha(summary_bytes) != SUMMARY_SHA:
        raise ValueError('manifest or published summary identity differs')
    seen = {}
    def read_record(runtime, batch, rep):
        path = args.run / f'{runtime}-b{batch}-r{rep}.json'
        raw = path.read_bytes()
        seen[path] = sha(raw)
        return raw
    result = extract(json.loads(manifest_bytes), json.loads(summary_bytes), read_record)
    if ((args.run / 'manifest.json').read_bytes() != manifest_bytes or args.summary.read_bytes() != summary_bytes
            or any(sha(path.read_bytes()) != value for path, value in seen.items())):
        raise ValueError('inputs changed during export')
    payload = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(payload)
    print(sha(payload.encode()))


if __name__ == '__main__':
    main()
