"""Capture official-seed TinyStories Metal samples for pinned external scoring."""
import argparse
import json
import os
import platform
import subprocess
import threading
from pathlib import Path

from run_phase3_mlx_bench import digest, monitor, run

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / 'artifacts/canonical/tinystories-phase4-v1'
BINARY = ROOT / 'target/release/examples/tinystories_samples'
SEEDS = (42, 0, 1, 2, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--qualification-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise ValueError('Apple Silicon required')
    if any(os.environ.get(k) for k in ('DYLD_INSERT_LIBRARIES', 'MallocStackLogging', 'MallocStackLoggingNoCompact')):
        raise ValueError('instrumented environment is not permitted')
    qualification = args.qualification.resolve()
    if digest(qualification) != args.qualification_sha256:
        raise ValueError('qualification hash differs')
    stats = json.loads(qualification.read_bytes())
    for level in range(1, 12):
        row = stats[f'generator.level_{level:02d}.logits']
        if row['argmax_differences'] or row['phase2_accumulated_band_atol_5e-4_rtol_2e-4_outside']:
            raise ValueError('direct MLX head comparison did not pass')
    if (stats['generator.final_logits']['phase2_accumulated_band_atol_5e-4_rtol_2e-4_outside']
            or stats['generator.output_ids']['differences']):
        raise ValueError('direct MLX decoder comparison did not pass')
    out = args.output.resolve()
    if out.exists():
        raise ValueError('output must be new')
    watched = [Path(__file__).resolve(), ROOT / 'tools/bench/run_phase3_mlx_bench.py',
               ROOT / 'tools/bench/phase3_process.py', BINARY, qualification,
               ROOT / '.reference/mlx-native/lib/libmlx.dylib', ROOT / '.reference/mlx-native/lib/mlx.metallib',
               ROOT / 'Cargo.toml', ROOT / 'Cargo.lock']
    watched += list((ROOT / 'crates').rglob('*.rs')) + list((ROOT / 'crates').rglob('Cargo.toml'))
    watched += [p for p in ARTIFACT.rglob('*') if p.is_file()]
    before = {str(p): digest(p) for p in watched}
    out.mkdir(parents=True)
    manifest = {'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                'watched_sha256': before, 'qualification_sha256': args.qualification_sha256,
                'seeds': SEEDS, 'samples_per_seed': 256, 'generation_batch': 128,
                'reference': 'pinned upstream Python CUDA; quality scored separately',
                'limitations': ['CPU activity sampled every two seconds; GPU activity is not measured.',
                                'No exclusive reservation or GPU clock/power changes.'], 'samples': []}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    stop = threading.Event()
    errors = []
    thread = threading.Thread(target=monitor, args=(stop, out / 'activity.jsonl', errors))
    thread.start()
    try:
        for seed in SEEDS:
            stem = f'rust-{seed}-b128'
            run([str(BINARY), str(ARTIFACT), '256', '128', str(seed)], stem, out)
            raw = (out / f'{stem}.stdout').read_bytes()
            result = json.loads(raw)
            if result['seed'] != seed or result['batch'] != 128:
                raise ValueError(f'invalid sample metadata: {stem}')
            for field in ('codes', 'raw_ids', 'texts', 'toks'):
                if len(result[field]) != 256:
                    raise ValueError(f'invalid sample count: {stem}/{field}')
            if (any(len(row) != 641 or any(type(x) is not int or not 0 <= x < 4096 for x in row)
                    for row in result['codes'])
                    or any(len(row) != 128 or any(type(x) is not int or not 0 <= x < 8192 for x in row)
                           for row in result['raw_ids'])):
                raise ValueError(f'invalid sampled IDs: {stem}')
            path = out / f'{stem}.json'
            path.write_bytes(raw)
            manifest['samples'].append({'seed': seed, 'file': path.name, 'sha256': digest(path)})
            (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            print(stem, 'complete', flush=True)
    finally:
        stop.set()
        thread.join(timeout=15)
        manifest['stable_inputs_and_sources'] = before == {str(p): digest(p) for p in watched}
        manifest['telemetry_errors'] = errors
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if errors or thread.is_alive() or not manifest['stable_inputs_and_sources']:
        raise RuntimeError('incomplete telemetry or changed source')
    print('sample capture complete; independent tokenizer and quality scoring remain required', flush=True)


if __name__ == '__main__':
    main()
