"""Publish sanitized XSum/WMT14 CUDA model latency trials, not inference outputs.

Existing nonblind, shared-GPU screens only; inputs and destination parents must
be trusted and stable. Summary agreement cannot authenticate individual trials.
"""
import argparse
import json
import math
import statistics
from pathlib import Path

from export_phase5_tinystories_trials import sha, validate_trials

PROFILES = {
    'xsum': {'batches': (1, 4, 32),
             'manifest_sha': '4395af5fcf84d629ff3a5eb7e785b8ebc4aa93e819d5583340471775d6ce61fe',
             'summary_sha': '1223fba4ca6ccff50c20f1243d293be10652f75d297f8dc01de8528bc7229725'},
    'wmt14': {'batches': (1, 4, 16, 32),
              'manifest_sha': 'b581a18e6f5b0a29ff96210e76205a25d0e5570707f149be0e40c43810b3a8bf',
              'summary_sha': '25516fe1c6c5195c16a0960f87144308e7d95795c2859445ad4ba984fd992229'},
}


def extract(profile_name, manifest, summary, read_record):
    profile = PROFILES[profile_name]
    batches = profile['batches']
    # Manifest differs in shape between these two benchmark supervisors.
    if manifest.get('batches') != list(batches):
        raise ValueError('unexpected batches')
    if profile_name == 'wmt14':
        valid_manifest = (manifest.get('repetitions') == 3 and manifest.get('warmups') == 5
                          and manifest.get('trials') == 30 and manifest.get('modes') == [0])
    else:
        valid_manifest = (manifest.get('processes_per_runtime_batch') == 3
                          and manifest.get('warmup_per_process') == 5 and manifest.get('trials_per_process') == 30)
    if not valid_manifest:
        raise ValueError('unexpected capture protocol')
    result = {}
    for batch in batches:
        result[str(batch)] = {}
        for runtime in ('python', 'rust'):
            expected = summary['batches'][str(batch)][runtime]['process_medians_ms']
            if not isinstance(expected, list) or len(expected) != 3:
                raise ValueError('invalid aggregate')
            records = []
            for rep in range(3):
                raw = read_record(runtime, batch, rep)
                doc = json.loads(raw)
                values = validate_trials(doc, batch)
                if ((profile_name == 'wmt14' and (type(doc.get('cfg')) is not int or doc['cfg'] != 0))
                        or (profile_name == 'xsum' and (type(doc.get('cfg_scale')) is not float or doc['cfg_scale'] != 0.0))):
                    raise ValueError('unexpected CFG mode')
                if not math.isclose(statistics.median(values) * 1000, expected[rep], rel_tol=1e-12, abs_tol=1e-10):
                    raise ValueError('process median differs from published aggregate')
                records.append({'source_sha256': sha(raw), 'trials_seconds': values})
            result[str(batch)][runtime] = records
    return {'schema_version': 1, 'profile': profile_name,
            'scope': 'shared-GPU synchronized model timing; excludes load, not release qualification',
            'warmups_per_process': 5, 'trials_per_process': 30, 'processes_per_runtime_batch': 3,
            'manifest_sha256': profile['manifest_sha'], 'summary_sha256': profile['summary_sha'], 'batches': result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=tuple(PROFILES), required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('new output file required')
    manifest_path = args.run / 'manifest.json'
    manifest, summary = manifest_path.read_bytes(), args.summary.read_bytes()
    profile = PROFILES[args.profile]
    if sha(manifest) != profile['manifest_sha'] or sha(summary) != profile['summary_sha']:
        raise ValueError('manifest or aggregate identity differs')
    seen = {}
    def read_record(runtime, batch, rep):
        stem = f'{runtime}-cfg0-b{batch}-r{rep}' if args.profile == 'wmt14' else f'{runtime}-b{batch}-r{rep}'
        path = args.run / f'{stem}.json'
        raw = path.read_bytes()
        seen[path] = sha(raw)
        return raw
    result = extract(args.profile, json.loads(manifest), json.loads(summary), read_record)
    if (manifest_path.read_bytes() != manifest or args.summary.read_bytes() != summary
            or any(sha(path.read_bytes()) != expected for path, expected in seen.items())):
        raise ValueError('inputs changed during export')
    payload = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(payload)
    print(sha(payload.encode()))


if __name__ == '__main__':
    main()
