"""Compare five checked LM1B CPU-penalty and device-penalty development captures."""
import argparse
import json
from pathlib import Path

from tinystories_artifact import ROOT, digest_file
RUST_CAPTURES = {11: '7d7aff0b110d1d00022631d779319dfaca47d16342e91d5177f0b13a3007aca0', 12: 'bbc628c6cda1fd933a9cf379057ba18f3d4f66d09be7a51a3b4d95de51101531', 13: 'a7d840f8228f4055cd01046da9ae33609ee7f90aea7e5d5bd2458ab73627a042', 14: '3a59d5f3019b3d86f3ced0bb9d4c22e54e3fc0d136f9d43a11ec67db84127db8', 15: '82163f29433ff47efa8593554ee6090b1f3b9ba3f3e81633f05108de8334a20e'}

DEVICE = {11: 'ad884b258b000f8d0f61a6e0a6408305f29711fafc03e454928935f655540060', 12: 'f8116ba24c2f941ec8c2d61008b464e17833905917984349968bed2d034f81d6', 13: '5241de5c05f5492503adba7d488e86de3f2930a54e09ed68b46f79fba6c3ddcd', 14: 'e8e75a981070b0b72db79c7f06518849b59a4f5cec9db1160b28c06eefc1ac43', 15: 'ee364d1a80b67cff489893ae111cb6706d5b01d183ffa6fad2285f7d11feaf5f'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.resolve().is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('output must be new and under reports')
    results = []
    watched = [Path(__file__).resolve(), (ROOT / 'tools/reference/lm1b_reference_rope_diagnostic.py').resolve()]
    for seed in range(11, 16):
        cpu_path = ROOT / f'reports/phase4-lm1b-reference-rope-development-seed{seed}-v1.json'
        device_path = ROOT / f'reports/phase4-lm1b-reference-rope-device-seed{seed}-v1.json'
        if digest_file(cpu_path) != RUST_CAPTURES[seed] or digest_file(device_path) != DEVICE[seed]:
            raise ValueError('capture hash differs')
        cpu, device = json.loads(cpu_path.read_bytes()), json.loads(device_path.read_bytes())
        for item in (cpu, device):
            if (item['size'], item['seed'], item['count'], item['batch_size'], item['artifact_manifest_sha256']) != (
                    'small', seed, 256, 8, '58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76'):
                raise ValueError('unexpected capture metadata')
            if item.get('diagnostic') != {'rotary_tables_sha256': '9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc', 'backend': 'reference operators with 1 MiB QKV preference'}:
                raise ValueError('candidate identity differs')
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
        if (digest_file(ROOT / f'reports/phase4-lm1b-reference-rope-development-seed{seed}-v1.json') != RUST_CAPTURES[seed]
                or digest_file(ROOT / f'reports/phase4-lm1b-reference-rope-device-seed{seed}-v1.json') != DEVICE[seed]):
            raise ValueError('capture identity changed during comparison')
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps({'samples': result['samples'], 'differences': [x['differences'] for x in results], 'sha256': digest_file(args.output)}, sort_keys=True))
    if any(any(x['differences'].values()) for x in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
