"""Pause only owned test processes at an RSS threshold for non-gating heap diagnostics."""
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

root = Path.cwd()
out = root / 'reports/phase4-xsum-mlx-rss-stop-v1'
if out.exists() or out.is_symlink():
    raise FileExistsError(out)
artifact = root / 'artifacts/canonical/xsum-inference-rope-v2-final'
inputs = root / 'reports/phase4-xsum-mlx-cli-inputs-v1/prefixes.json'
rust = root / 'target/release/examples/xsum_perf_screen'
python = root / '.reference/mlx-venv/bin/python'
adapter = root / 'tools/reference/phase4_xsum_mlx.py'
files = [artifact / 'manifest.json', inputs, rust, adapter, Path(__file__).resolve()]
def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
hashes = {str(p): digest(p) for p in files}
if hashes[str(artifact / 'manifest.json')] != '8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4' or hashes[str(inputs)] != 'a9c0da895804e5146191993ae27db4c921d1b08b21594cb6eaa63777748019cf':
    raise ValueError('unexpected input identity')
out.mkdir()
results = {}
for kind in ('rust', 'python'):
    result_path = out / f'{kind}.json'
    command = ([str(rust), str(artifact), str(inputs), '1'] if kind == 'rust' else
               [str(python), str(adapter), '--artifact', str(artifact), '--inputs', str(inputs),
                '--batch', '1', '--bench', '--output', str(result_path)])
    with (out / f'{kind}.stdout').open('xb') as stdout, (out / f'{kind}.stderr').open('xb') as stderr:
        start = time.monotonic()
        process = subprocess.Popen(command, cwd=root, stdout=stdout, stderr=stderr, start_new_session=True)
        paused = False
        captured = None
        try:
            while process.poll() is None:
                stat = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(process.pid)], capture_output=True, text=True, timeout=2)
                try:
                    rss = int(stat.stdout.strip())
                except ValueError:
                    rss = 0
                if captured is None and rss >= 1100 * 1024:
                    elapsed = time.monotonic() - start
                    os.kill(process.pid, signal.SIGSTOP); paused = True
                    for name, cmd in (('vmmap', ['/usr/bin/vmmap', '-summary', str(process.pid)]),
                                      ('heap', ['/usr/bin/heap', '--sortBySize', '--noContent', str(process.pid)])):
                        try:
                            r = subprocess.run(cmd, capture_output=True, timeout=30, check=False)
                            (out / f'{kind}-{name}.txt').write_bytes(r.stdout + r.stderr)
                            (out / f'{kind}-{name}.exit').write_text(str(r.returncode) + '\n')
                        except subprocess.TimeoutExpired as exc:
                            (out / f'{kind}-{name}.txt').write_bytes((exc.stdout or b'') + (exc.stderr or b''))
                            (out / f'{kind}-{name}.exit').write_text('timeout\n')
                    os.kill(process.pid, signal.SIGCONT); paused = False
                    captured = {'rss_kib': rss, 'seconds_before_pause': elapsed}
                time.sleep(0.02)
            code = process.wait()
        finally:
            if process.poll() is None:
                if paused:
                    os.kill(process.pid, signal.SIGCONT)
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
        if code or captured is None:
            raise RuntimeError(f'{kind} failed or threshold missed: {code}')
    if kind == 'rust':
        result_path.write_bytes((out / 'rust.stdout').read_bytes())
    result = json.loads(result_path.read_bytes())
    results[kind] = {'command': command, 'threshold_observation': captured,
                     'codes': result['codes'], 'ids': result['ids']}
def flat(values):
    return [item for row in values for item in row] if values and isinstance(values[0], list) else values
if any(flat(results['rust'][key]) != flat(results['python'][key]) for key in ('codes', 'ids')):
    raise ValueError('outputs differ')
if any(digest(Path(p)) != sha for p, sha in hashes.items()):
    raise ValueError('input changed')
(out / 'summary.json').write_text(json.dumps({'source_sha256': hashes, 'results': results,
    'scope': 'intrusive threshold snapshots, not timing or peak-RSS comparison'}, indent=2, sort_keys=True) + '\n')
print('complete')
