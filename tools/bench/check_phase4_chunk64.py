"""Compare seeded MLX outputs around the isolated 64-row chunk boundary."""
import argparse
import json
import subprocess
from pathlib import Path

from run_phase3_mlx_bench import digest

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'artifact', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--candidate-source', type=Path, action='append', required=True)
    args = parser.parse_args()
    old, new, artifact, out = (getattr(args, name).resolve()
                               for name in ('baseline', 'candidate', 'artifact', 'output'))
    if out.exists() or not old.is_file() or not new.is_file() or old == new:
        raise ValueError('need distinct binaries and a new output directory')
    watched = [Path(__file__).resolve(), ROOT / 'tools/bench/run_phase3_mlx_bench.py',
               ROOT / '.reference/mlx-native/lib/libmlx.dylib',
               ROOT / '.reference/mlx-native/lib/mlx.metallib', old, new]
    watched += [source.resolve() for source in args.candidate_source]
    watched += [path for path in artifact.rglob('*') if path.is_file()]
    before = {str(path): digest(path) for path in watched}
    out.mkdir(parents=True)
    manifest = {'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'watched_sha256': before, 'commands': [], 'comparisons': {},
                'scope': 'one seed and one batch at each boundary; no quality or performance claim'}
    try:
        for batch in (64, 65, 127, 128):
            outputs = {}
            for label, binary in (('baseline', old), ('candidate', new)):
                command = [str(binary), str(artifact), str(batch), str(batch), '42']
                manifest['commands'].append(command)
                result = subprocess.run(command, capture_output=True, timeout=180, check=False, cwd=ROOT)
                (out / f'{label}-b{batch}.json').write_bytes(result.stdout)
                (out / f'{label}-b{batch}.stderr').write_bytes(result.stderr)
                if result.returncode:
                    raise RuntimeError(f'{label} batch {batch} exited {result.returncode}')
                data = json.loads(result.stdout)
                if data['seed'] != 42 or data['batch'] != batch:
                    raise ValueError('unexpected sample metadata')
                if any(len(data[key]) != batch for key in ('codes', 'raw_ids', 'texts', 'toks')):
                    raise ValueError('unexpected sample count')
                outputs[label] = data
            manifest['comparisons'][str(batch)] = {
                key: sum(a != b for a, b in zip(outputs['baseline'][key], outputs['candidate'][key]))
                for key in ('codes', 'raw_ids', 'texts', 'toks')}
            print(f"batch {batch}: {manifest['comparisons'][str(batch)]}", flush=True)
        manifest['status'] = 'complete'
    finally:
        manifest['stable_inputs_and_sources'] = before == {str(path): digest(path) for path in watched}
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if not manifest['stable_inputs_and_sources']:
        raise RuntimeError('watched inputs or sources changed')


if __name__ == '__main__':
    main()
