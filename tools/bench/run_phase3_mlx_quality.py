"""Capture the pinned golden and all three 2,000-board Sudoku evaluations on Metal."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
from pathlib import Path

from run_phase3_mlx_bench import ROOT, digest, monitor, run

DATA = ROOT / 'artifacts/canonical/sudoku-data-phase3'
ARTIFACTS = {'easy': ROOT / 'artifacts/canonical/sudoku-easy-qualified',
             **{d: ROOT / f'artifacts/canonical/sudoku-{d}-phase3' for d in ('medium', 'hard')}}
MANIFESTS = {'easy': 'ad6a99ad7d5ccc069e0fedf26a85f755e06cd76590463e1e04e17363e35d9832',
             'medium': 'facff42955d42294dc94f33d54688dc53046c1f9cc029e903f1ffe787ce45219',
             'hard': '9ed3e86fdc3d193c88e1f4aae8f58a54019f985bdcf2acb9c851772c51dad86b'}
UPSTREAM = 'c70274273f489abc73832b2b98584a8483b6b203'


def preflight(reference_dir, reference_manifest_sha256):
    manifest_path = reference_dir / 'manifest.json'
    if digest(manifest_path) != reference_manifest_sha256:
        raise ValueError('Linux reference manifest hash differs')
    manifest = json.loads(manifest_path.read_bytes())
    if not manifest.get('stable_inputs_and_sources') or manifest.get('upstream_head') != UPSTREAM:
        raise ValueError('Linux reference was not captured from stable pinned sources')
    def matches_linux_capture(path):
        suffix = '/' + path.relative_to(ROOT).as_posix()
        hashes = [value for name, value in manifest['watched_sha256'].items() if name.endswith(suffix)]
        return len(hashes) == 1 and digest(path) == hashes[0]
    watched = [manifest_path]
    for difficulty, artifact in ARTIFACTS.items():
        artifact_manifest = artifact / 'manifest.json'
        if digest(artifact_manifest) != MANIFESTS[difficulty]:
            raise ValueError(f'{difficulty} artifact manifest differs')
        artifact_data = json.loads(artifact_manifest.read_bytes())
        watched.append(artifact_manifest)
        for name, record in artifact_data['files'].items():
            path = artifact / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
                raise ValueError(f'{difficulty} artifact file differs: {name}')
            watched.append(path)
        reference = reference_dir / f'{difficulty}-python.json'
        if digest(reference) != manifest['output_sha256'][reference.name]:
            raise ValueError(f'{difficulty} Python reference differs')
        result = json.loads(reference.read_bytes())
        dataset_path, input_path = DATA / f'{difficulty}.json', DATA / f'{difficulty}-2000-inputs.json'
        if not matches_linux_capture(dataset_path) or not matches_linux_capture(input_path):
            raise ValueError(f'{difficulty} dataset differs from the Linux capture')
        dataset, raw_inputs = json.loads(dataset_path.read_bytes()), input_path.read_bytes()
        expected = (json.dumps(dataset['inputs'], separators=(',', ':')) + '\n').encode()
        if (dataset['difficulty'] != difficulty or result.get('difficulty') != difficulty
                or result.get('upstream') != UPSTREAM or len(dataset['inputs']) != 2000
                or raw_inputs != expected or result.get('inputs_sha256') != hashlib.sha256(raw_inputs).hexdigest()
                or len(result['codes']) != 2000 or len(result['ids']) != 2000):
            raise ValueError(f'{difficulty} evaluation inputs or reference metadata differ')
        watched.extend([reference, dataset_path, input_path])
    return watched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--reference-manifest-sha256', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--preflight-only', action='store_true', help='check artifacts and reference inputs without GPU execution')
    args = parser.parse_args()
    if any(os.environ.get(k) for k in ('DYLD_INSERT_LIBRARIES', 'MallocStackLogging', 'MallocStackLoggingNoCompact')):
        raise SystemExit('instrumented environment is not permitted')
    references = args.reference_dir.resolve()
    watched = preflight(references, args.reference_manifest_sha256)
    if args.preflight_only:
        print('Pinned artifacts, datasets and Linux reference outputs verified; no GPU work ran.')
        return
    if args.output is None or args.output.exists():
        raise SystemExit('a new output directory is required')
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise SystemExit('Apple Silicon required')
    cli = ROOT / 'target/release/ncm'
    watched += [cli, Path(__file__).resolve(), ROOT / 'tools/bench/run_phase3_mlx_bench.py',
                ROOT / 'tools/bench/phase3_process.py', ROOT / 'tools/bench/phase3_sudoku_score.py',
                ROOT / 'Cargo.toml', ROOT / 'Cargo.lock',
                ROOT / '.reference/mlx-native/lib/libmlx.dylib', ROOT / '.reference/mlx-native/lib/mlx.metallib']
    watched += list((ROOT / 'crates').rglob('*.rs')) + list((ROOT / 'crates').rglob('Cargo.toml'))
    before = {str(p): digest(p) for p in watched}
    out = args.output.resolve(); out.mkdir(parents=True)
    manifest = {'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                'reference_manifest_sha256': args.reference_manifest_sha256, 'watched_sha256': before,
                'scope': 'pinned synthetic golden followed by full 2,000-board MLX CLI runs',
                'limitations': ['CPU activity sampled every two seconds; GPU activity is not measured.',
                                'No exclusive reservation, CPU affinity or GPU clock/power changes.',
                                'Peak RSS includes process startup and model loading.'], 'runs': []}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    stop = threading.Event(); errors = []
    thread = threading.Thread(target=monitor, args=(stop, out / 'activity.jsonl', errors)); thread.start()
    try:
        os.environ['NCM_PHASE3_ARTIFACT'] = str(ARTIFACTS['easy'])
        golden = ['cargo', 'test', '-p', 'ncm-backend-mlx', '--features', 'mlx', '--test', 'generator_golden', '--', '--ignored']
        manifest['runs'].append({'name': 'synthetic-golden', 'command': golden})
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        run(golden, 'synthetic-golden', out)
        result = (out / 'synthetic-golden.stdout').read_text()
        if 'metal_matches_pinned_python_synthetic_golden ... ok' not in result or '1 passed' not in result:
            raise ValueError('synthetic golden did not run and pass')
        print('synthetic-golden complete', flush=True)
        for difficulty, artifact in ARTIFACTS.items():
            stem = f'{difficulty}-rust'
            command = [str(cli), 'generate', str(artifact), 'mlx', str(DATA / f'{difficulty}-2000-inputs.json'), '128']
            manifest['runs'].append({'name': stem, 'command': command})
            (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            run(command, stem, out)
            score = subprocess.check_output([sys.executable, str(ROOT / 'tools/bench/phase3_sudoku_score.py'),
                                             str(DATA / f'{difficulty}.json'), str(out / f'{stem}.stdout'),
                                             str(references / f'{difficulty}-python.json')], cwd=ROOT)
            (out / f'{difficulty}-score.json').write_bytes(score)
            print(stem, 'complete', flush=True)
    finally:
        stop.set(); thread.join(timeout=15)
        manifest['stable_inputs_and_sources'] = before == {str(p): digest(p) for p in watched}
        manifest['telemetry_errors'] = errors
        manifest['output_sha256'] = {p.name: digest(p) for p in out.glob('*') if p.is_file() and p.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if errors or thread.is_alive() or not manifest['stable_inputs_and_sources']:
        raise RuntimeError('incomplete telemetry or changed source')
    print('MLX quality capture complete; review score differences before acceptance.', flush=True)


if __name__ == '__main__':
    main()
