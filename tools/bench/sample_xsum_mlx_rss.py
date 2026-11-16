"""Non-gating fresh-process RSS sampling of the existing Metal baseline."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

root = Path.cwd()
out = root / 'reports/phase4-xsum-mlx-rss-samples-v1'
if out.exists() or out.is_symlink():
    raise FileExistsError(out)
artifact = root / 'artifacts/canonical/xsum-inference-rope-v2-final'
inputs = root / 'reports/phase4-xsum-mlx-cli-inputs-v1/prefixes.json'
rust = root / 'target/release/examples/xsum_perf_screen'
python = root / '.reference/mlx-venv/bin/python'
adapter = root / 'tools/reference/phase4_xsum_mlx.py'
files = [artifact / 'manifest.json', inputs, rust, adapter, Path(__file__).resolve()]
def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
hashes = {str(p): digest(p) for p in files}
if hashes[str(artifact / 'manifest.json')] != '8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4' or hashes[str(inputs)] != 'a9c0da895804e5146191993ae27db4c921d1b08b21594cb6eaa63777748019cf':
    raise ValueError('unexpected inputs')
out.mkdir()
results = {}
for kind in ('rust', 'python'):
    result_path = out / f'{kind}.json'
    command = ([str(rust), str(artifact), str(inputs), '1'] if kind == 'rust' else
               [str(python), str(adapter), '--artifact', str(artifact), '--inputs', str(inputs),
                '--batch', '1', '--bench', '--output', str(result_path)])
    with (out / f'{kind}.stdout').open('xb') as stdout, (out / f'{kind}.stderr').open('xb') as stderr:
        begin = time.monotonic()
        process = subprocess.Popen(command, cwd=root, stdout=stdout, stderr=stderr, start_new_session=True)
        samples = []
        try:
            while process.poll() is None:
                ps = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(process.pid)], capture_output=True, text=True, timeout=2)
                try:
                    rss_kib = int(ps.stdout.strip())
                    if rss_kib > 0:
                        samples.append({'seconds': time.monotonic() - begin, 'rss_kib': rss_kib})
                except ValueError:
                    pass
                time.sleep(0.05)
            code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate(); process.wait(timeout=10)
        if code or not samples:
            raise RuntimeError(f'{kind} failed or had no RSS samples: {code}')
    if kind == 'rust':
        result_path.write_bytes((out / 'rust.stdout').read_bytes())
    result = json.loads(result_path.read_bytes())
    results[kind] = {'command': command, 'samples': samples, 'elapsed_seconds': time.monotonic() - begin,
                     'maximum_sampled_rss_kib': max(s['rss_kib'] for s in samples),
                     'codes': result['codes'], 'ids': result['ids']}
def flat(values):
    return [item for row in values for item in row] if values and isinstance(values[0], list) else values
if (flat(results['rust']['codes']) != flat(results['python']['codes'])
        or flat(results['rust']['ids']) != flat(results['python']['ids'])):
    raise ValueError('baseline outputs differ')
if any(digest(Path(p)) != h for p, h in hashes.items()):
    raise ValueError('inputs changed during diagnostic')
summary = {'source_sha256': hashes, 'results': results,
           'scope': 'one process per runtime, sampled ps RSS; not a performance or peak-RSS gate'}
(out / 'summary.json').write_text(json.dumps(summary, sort_keys=True, indent=2) + '\n')
print('complete')
