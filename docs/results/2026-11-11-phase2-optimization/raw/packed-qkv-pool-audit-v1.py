"""Summarize retained CUDA pool counters; no hardware access or performance gate."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
ARCHIVES = {
    'isolated': ('packed-qkv-v1.tar.xz', 'cfd2de5b6f10989d587d39cfa07eefa3a93e3dbb6cc2735cddb34dc9111c695a'),
    'combined': ('combined-linux-gate-v1.tar.xz', 'cb98da607fdd11155b0801ddb376e43516afb9716484f2df38cc54a9fbca0ecb'),
}
KEYS = ('used_current', 'used_peak', 'reserved_current', 'reserved_peak')
SCOPES = ('encode', 'decode', 'reconstruct', 'resident_decoder_block')

def read(tar, name):
    member = tar.extractfile(name)
    if member is None:
        raise ValueError('missing archive member')
    return json.load(member)

def main():
    result = {'schema_version': 1, 'scope': 'Offline CUDA pool counters only; not a new run or acceptance gate.',
              'archives_sha256': {k: v[1] for k, v in ARCHIVES.items()}, 'comparisons': {}}
    for kind, (filename, expected) in ARCHIVES.items():
        path = ROOT / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('archive digest differs')
        with tarfile.open(path, 'r:xz') as tar:
            if kind == 'combined':
                env = read(tar, 'reports/phase2-combined-linux-threeway-v1/environment.json')
                if env['binary_sha256']['candidate']['codec_bench'] != 'e4c13a76e7bb4f2d6d7876e5b66a4d2a8cd43d2846120ddb945db3946f509004':
                    raise ValueError('candidate benchmark identity differs')
            rows = {}
            for role in ('previous', 'candidate'):
                for i in range(1, 4):
                    name = (f'ncm-packed-qkv-v1/{role}-{i}.json' if kind == 'isolated' else
                            f'reports/phase2-combined-linux-threeway-v1/{role}/codec-rust-{i}.json')
                    report = read(tar, name)
                    rows[role, i] = {(m['batch'], m['scope']): m['pool_bytes'] for m in report['measurements']}
            comparison = []
            for batch in (1, 2, 4):
                for scope in SCOPES:
                    values = {}
                    for role in ('previous', 'candidate'):
                        samples = [{key: rows[role, i][batch, scope][key] for key in KEYS} for i in range(1, 4)]
                        if not all(sample == samples[0] for sample in samples):
                            raise ValueError('pool counters differ across retained processes')
                        values[role] = samples[0]
                    comparison.append({'batch': batch, 'scope': scope, **values})
            result['comparisons'][kind] = comparison
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
