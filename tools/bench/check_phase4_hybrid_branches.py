"""Compare isolated hybrid generator branches with their retained Metal references."""
import argparse
import json
import subprocess
from pathlib import Path

from run_phase3_mlx_bench import digest

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'fused', 'hybrid', 'artifact', 'baseline-reference', 'fused-reference', 'candidate-source', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    binary = {name: getattr(args, name).resolve() for name in ('baseline', 'fused', 'hybrid')}
    artifact, out = args.artifact.resolve(), args.output.resolve()
    refs = {'baseline': args.baseline_reference.resolve(), 'fused': args.fused_reference.resolve()}
    source = args.candidate_source.resolve()
    if out.exists() or len(set(binary.values())) != 3:
        raise ValueError('need three distinct binaries and a new output directory')
    watched = list(binary.values()) + list(refs.values()) + [source, Path(__file__).resolve(),
              ROOT / '.reference/mlx-native/lib/libmlx.dylib', ROOT / '.reference/mlx-native/lib/mlx.metallib']
    watched += [p for p in artifact.rglob('*') if p.is_file()]
    before = {str(p): digest(p) for p in watched}
    out.mkdir(parents=True)
    results = {}
    commands = []
    try:
        for batch, reference in ((1, 'baseline'), (4, 'baseline'), (15, 'baseline'), (16, 'fused'), (128, 'fused')):
            for kind in (reference, 'hybrid'):
                name = f'{kind}-b{batch}'
                command = [str(binary[kind]), str(artifact), str(batch), str(batch), '42']
                commands.append(command)
                process = subprocess.run(command, capture_output=True, timeout=90, check=False)
                (out / f'{name}.json').write_bytes(process.stdout)
                (out / f'{name}.stderr').write_bytes(process.stderr)
                if process.returncode:
                    raise RuntimeError(f'{name} failed: {process.returncode}')
                data = json.loads(process.stdout)
                if data['seed'] != 42 or data['batch'] != batch or any(len(data[k]) != batch for k in ('codes', 'raw_ids', 'texts', 'toks')):
                    raise ValueError(f'invalid sample structure: {name}')
                results[name] = data
            for key in ('codes', 'raw_ids', 'texts', 'toks'):
                if results[f'{reference}-b{batch}'][key] != results[f'hybrid-b{batch}'][key]:
                    raise AssertionError(f'{key} differ at batch {batch}')
            print(f'batch {batch} matches {reference}', flush=True)
        for kind in ('baseline', 'fused'):
            if kind == 'baseline':
                command = [str(binary[kind]), str(artifact), '128', '128', '42']
                commands.append(command)
                process = subprocess.run(command, capture_output=True, timeout=90, check=False)
                (out / 'baseline-b128.json').write_bytes(process.stdout)
                (out / 'baseline-b128.stderr').write_bytes(process.stderr)
                if process.returncode:
                    raise RuntimeError(f'baseline-b128 failed: {process.returncode}')
                results['baseline-b128'] = json.loads(process.stdout)
            retained = json.loads(refs[kind].read_bytes())
            if retained['seed'] != 42 or retained['batch'] != 128:
                raise ValueError('invalid retained reference metadata')
            for key in ('codes', 'raw_ids', 'texts', 'toks'):
                if results[f'{kind}-b128'][key] != retained[key][:128]:
                    raise AssertionError(f'{kind} differs from retained {key}')
        status = 'complete'
    finally:
        after = {str(p): digest(p) for p in watched}
        manifest = {'status': locals().get('status', 'failed'), 'commands': commands,
                    'watched_sha256': before, 'stable_inputs_and_sources': before == after,
                    'scope': 'single-batch branch identity; no performance or exhaustive-parity claim'}
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if not manifest['stable_inputs_and_sources']:
        raise RuntimeError('inputs or sources changed')
    print('branch identity checks complete', flush=True)


if __name__ == '__main__':
    main()
