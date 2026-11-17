"""Compare five checked LM1B CPU-penalty and device-penalty development captures."""
import argparse
import json
from pathlib import Path
from lm1b_device_development_compare import DEVICE as RUST_CAPTURES
from tinystories_artifact import ROOT, digest_file

DEVICE = {
    11: '8ac975723efe98e3ba5d7fd277f9f73cc1faa8144f3da9f55c02369145b046d1',
    12: 'bb525e2a5781ef5b9dff05570b59ed0b767b496af41e7b7e507ff475a70eacfc',
    13: '937746f36bda59f1141a6d73b3efedcf1ab2e941596e3146785b439ee7068c3c',
    14: '6a0a08961a1f57c4b6d72559f8c88ef35468cea39c5a1414f9dc178be7ec9d4c',
    15: '06011ff91b10431ae4088b9bd9d25f093fc3bb99556a67a8751966d144546d33',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.resolve().is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('output must be new and under reports')
    results = []
    watched = [Path(__file__).resolve(), Path(__file__).with_name('lm1b_device_development_compare.py').resolve()]
    for seed in range(11, 16):
        cpu_path = ROOT / f'reports/phase4-lm1b-small-device-dev-seed-{seed}-v1.json'
        device_path = ROOT / f'reports/phase4-lm1b-small-full-device-dev-seed-{seed}-v1.json'
        if digest_file(cpu_path) != RUST_CAPTURES[seed] or digest_file(device_path) != DEVICE[seed]:
            raise ValueError('capture hash differs')
        cpu, device = json.loads(cpu_path.read_bytes()), json.loads(device_path.read_bytes())
        for item in (cpu, device):
            if (item['size'], item['seed'], item['count'], item['batch_size'], item['artifact_manifest_sha256']) != (
                    'small', seed, 256, 8, '58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76'):
                raise ValueError('unexpected capture metadata')
            for name, width, upper in [('codes', 641, 8192), ('raw_ids', 128, 30528)]:
                rows = item[name]
                if len(rows) != 256 or any(len(row) != width or any(type(x) is not int or not 0 <= x < upper for x in row) for row in rows):
                    raise ValueError('invalid captured values')
        if device.get('penalty') != 'CUDA FP64':
            raise ValueError('device penalty mode missing')
        counts = {key: sum(a != b for x, y in zip(cpu[key], device[key], strict=True) for a, b in zip(x, y, strict=True)) for key in ('codes', 'raw_ids')}
        results.append({'seed': seed, 'cpu_sha256': RUST_CAPTURES[seed], 'device_sha256': DEVICE[seed], 'differences': counts})
        watched.extend((cpu_path, device_path))
    before = {str(path): digest_file(path) for path in watched}
    result = {'scope': 'five non-held-out development seeds; exact outputs only, not a performance or quality gate',
              'samples': 1280, 'results': results}
    if before != {str(path): digest_file(path) for path in watched}:
        raise ValueError('inputs changed during comparison')
    for seed in range(11, 16):
        if (digest_file(ROOT / f'reports/phase4-lm1b-small-device-dev-seed-{seed}-v1.json') != RUST_CAPTURES[seed]
                or digest_file(ROOT / f'reports/phase4-lm1b-small-full-device-dev-seed-{seed}-v1.json') != DEVICE[seed]):
            raise ValueError('capture identity changed during comparison')
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps({'samples': result['samples'], 'differences': [x['differences'] for x in results], 'sha256': digest_file(args.output)}, sort_keys=True))
    if any(any(x['differences'].values()) for x in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
