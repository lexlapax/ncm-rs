"""Capture full CUDA Sudoku comparisons with before/after input and source hashes."""
import argparse
import json
import os
import subprocess
from pathlib import Path

from run_phase3_linux_bench import ROOT, SOURCES, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    forbidden = ('LD_PRELOAD', 'LD_AUDIT', 'CUDA_INJECTION64_PATH', 'CUDA_VISIBLE_DEVICES',
                 'CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG', 'NCM_PHASE3_LT_PROFILE')
    if any(os.environ.get(name) for name in forbidden):
        raise SystemExit('profiling, injection or device-override environment is not permitted')
    out = args.output.resolve()
    if out.exists():
        raise SystemExit('output exists; choose a new run identity')
    upstream = ROOT / '.reference/upstream-ncm'
    if subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip():
        raise SystemExit('upstream checkout is dirty')
    cli = ROOT / 'target/release/ncm'
    reference = ROOT / '.reference/venv/bin/python'
    data = ROOT / 'artifacts/canonical/sudoku-data-phase3'
    artifacts = {'easy': ROOT / 'artifacts/canonical/sudoku-easy-qualified',
                 **{d: ROOT / f'artifacts/canonical/sudoku-{d}-phase3' for d in ('medium', 'hard')}}
    watched = [ROOT / p for p in SOURCES] + [Path(__file__).resolve(), cli,
               ROOT / 'tools/bench/phase3_sudoku_score.py']
    watched += list((ROOT / 'crates').rglob('*.rs'))
    watched += list((ROOT / 'crates').rglob('Cargo.toml'))
    watched += list(upstream.rglob('*.py'))
    for d, artifact in artifacts.items():
        watched += [p for p in artifact.rglob('*') if p.is_file()]
        watched += [data / f'{d}.json', data / f'{d}-2000-inputs.json']
        watched += [ROOT / f'artifacts/upstream/sudoku/{d}/{name}' for name in ('config.yaml', 'generator.pt', 'tokenizer.pt')]
    before = {str(p): digest(p) for p in watched}
    out.mkdir(parents=True)
    manifest = {'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'upstream_head': subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip(),
                'watched_sha256': before, 'runs': []}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    for difficulty, artifact in artifacts.items():
        inputs = data / f'{difficulty}-2000-inputs.json'
        reference_out = out / f'{difficulty}-python.json'
        rust_out = out / f'{difficulty}-rust.json'
        commands = {
            'python': [str(reference), str(ROOT / 'tools/reference/phase3_generator_reference.py'),
                       '--upstream', str(upstream), '--source', str(ROOT / f'artifacts/upstream/sudoku/{difficulty}'),
                       '--difficulty', difficulty, '--inputs', str(inputs), '--output', str(reference_out), '--batch-size', '128'],
            'rust': [str(cli), 'generate', str(artifact), 'cuda', str(inputs), '128'],
        }
        for runtime, command in commands.items():
            stem = f'{difficulty}-{runtime}'
            with (out / f'{stem}.stdout').open('xb') as stdout, (out / f'{stem}.stderr').open('xb') as stderr:
                completed = subprocess.run(['/usr/bin/time', '-v', '-o', str(out / f'{stem}.time'), *command],
                                           cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=False)
            manifest['runs'].append({'name': stem, 'command': command, 'returncode': completed.returncode})
            (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            if completed.returncode:
                raise RuntimeError(f'{stem} failed: {completed.returncode}')
            if runtime == 'rust':
                rust_out.write_bytes((out / f'{stem}.stdout').read_bytes())
            print(stem, 'complete', flush=True)
        score = subprocess.check_output(['python3', str(ROOT / 'tools/bench/phase3_sudoku_score.py'),
                                         str(data / f'{difficulty}.json'), str(rust_out), str(reference_out)], cwd=ROOT)
        (out / f'{difficulty}-score.json').write_bytes(score)
    after = {str(p): digest(p) for p in watched}
    manifest['stable_inputs_and_sources'] = before == after
    manifest['output_sha256'] = {p.name: digest(p) for p in out.glob('*') if p.is_file() and p.name != 'manifest.json'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if before != after or subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('source or input changed during capture')


if __name__ == '__main__':
    main()
