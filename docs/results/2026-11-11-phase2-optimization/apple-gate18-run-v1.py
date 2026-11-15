"""One declared Apple gate: 3-process matrix, 18-process control, then audits."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
PRE = ROOT / '.reference/phase2-mlx-gate18-preflight-v1'
STATUS = PRE / 'gate-status.json'
IDENTITIES = PRE / 'identities-v1.json'
OWNED_STATUS = False
MATRIX = ROOT / 'reports/phase2-mlx-gate18-matrix-v1'
CONTROL = ROOT / 'reports/phase2-mlx-gate18-control-v1'
INITIAL = ROOT / 'reports/phase2-mlx-gate18-initial-audit-v1.json'
FINAL = ROOT / 'reports/phase2-mlx-gate18-controlled-audit-v1.json'
BASE = ROOT / '.reference/phase2-optimization-baseline'
OLD = ROOT / '.reference/phase2-optimization-mlx-bin'

def capture(args, *, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def status(stage, **fields):
    global OWNED_STATUS
    require(OWNED_STATUS or not STATUS.exists(), 'preexisting gate status')
    data = {'stage': stage, 'pid': os.getpid(), 'updated_utc': datetime.now(timezone.utc).isoformat(), **fields}
    tmp = STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(STATUS)
    OWNED_STATUS = True

def check():
    require(platform.system() == 'Darwin' and platform.machine() == 'arm64', 'Apple Silicon required')
    require(not capture(['git', 'status', '--porcelain']), 'candidate checkout dirty')
    require(not capture(['git', 'status', '--porcelain'], cwd=BASE), 'baseline checkout dirty')
    require(capture(['git', 'rev-parse', 'HEAD'], cwd=BASE) == '3307dff82a08f7251b64de0866a01d131528b74f', 'baseline commit changed')
    changed = set(capture(['git', 'diff', '--name-only', '5200924', 'HEAD', '--', 'crates', 'Cargo.lock', 'vendor', 'tools', 'tests']).splitlines())
    require(changed <= {'tools/bench/trace_mlx_tail_activity.py', 'tools/bench/test_trace_mlx_tail_activity.py'}, 'qualified execution sources changed')
    expected = {
        OLD/'mlx_codec_bench': '6483e9b6f24b2d6fad9b2f28487f99b3a69c0e289c4daf977e16733c277acfbd',
        OLD/'asset_bench': 'dcaed6168a6f0ed7dc53a4efbae45ebf1b6912434b46de701127ee40079c0a50',
        ROOT/'target/release/examples/mlx_codec_bench': '6bb8929ce0704a659423e8ef7082dfd4a4388bb309339924e0d943f7cd6ef46b',
        ROOT/'target/release/examples/asset_bench': 'd443d975604546f8a54255b96a52469c55791092f346d16d65516d5541c75dde',
        ROOT/'.reference/mlx-native/lib/libmlx.dylib': 'd24c7a9b9d55a76bfd3bbcd1d042251a185cbadcb6340c3244a6ffa3dcb7c7e8',
        ROOT/'.reference/mlx-native/lib/mlx.metallib': 'dc59d1cceb1a5c7e578232e6e41e28e2c73c9463ac6dbc3886c3ee17ffc270ed',
        ROOT/'tools/bench/run_phase2_mlx_comparison.py': '095f424b44368c66285b265a7e0d35192d59be22b771ca133650ea483a759ab3',
        ROOT/'artifacts/canonical/sudoku-easy-qualified/manifest.json': 'ad6a99ad7d5ccc069e0fedf26a85f755e06cd76590463e1e04e17363e35d9832',
        PRE/'qualification.json': '84fd4e747d650fb135947146c0a5e0f442dbc562298dc98ea33b7d309c39a160',
        ROOT/'reports/phase2-tiny-borrow-mlx-lifecycle-v1.json': '1c966a0c1e399793a36d04a0c4c0182c90a733eb5ef3611232bc43e3f404c14a',
    }
    for path, digest in expected.items():
        require(sha(path) == digest, f'identity changed: {path}')
    require(capture(['sw_vers', '-productVersion']) == '26.7.3' and capture(['sw_vers', '-buildVersion']) == '25G416', 'macOS changed')
    require(capture(['sysctl', '-n', 'machdep.cpu.brand_string']) == 'Apple M1 Ultra', 'CPU changed')
    require(capture(['rustc', '--version']) == 'rustc 1.99.1 (58cf8f9c3 2026-10-13) (Homebrew)', 'compiler changed')
    code = ('import hashlib,importlib.metadata as m,json; '
            'd={x.metadata["Name"]:x.version for x in m.distributions()}; '
            'print(hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":")).encode()).hexdigest())')
    for env, digest in [('mlx-venv', '5b46008dc81b4bcc33f8a1bc34c2ee5e40b445c74da7a87e7cb89c25360c0ae6'),
                        ('tokenizers-venv', 'e272cd1bd1344d42dded11e230bfb194cae14355534082a2cd031ae80b56297d')]:
        py = str(ROOT/'.reference'/env/'bin/python')
        require(capture([py, '--version']) == 'Python 3.14.10' and capture([py, '-c', code]) == digest, f'{env} changed')
    require(not any(os.environ.get(k) for k in ('DYLD_INSERT_LIBRARIES', 'MallocStackLogging', 'MallocStackLoggingNoCompact')), 'instrumented environment')
    require(not any(p.exists() for p in (STATUS, IDENTITIES, MATRIX, CONTROL, INITIAL, FINAL)), 'gate output already exists')
    sys.path.insert(0, str(ROOT/'tools/bench'))
    from audit_phase2_performance import quality
    quality(json.loads((PRE/'qualification.json').read_bytes()))
    return {str(p.relative_to(ROOT)): digest for p, digest in expected.items()}

def main():
    require(PRE.is_dir(), 'preflight directory missing')
    identities = check()
    head = capture(['git', 'rev-parse', 'HEAD'])
    with IDENTITIES.open('x') as file:
        json.dump({'head': head, 'sha256': identities}, file, indent=2)
        file.write('\n')
    status('preflight_complete', head=head)
    runner = ROOT/'tools/bench/run_phase2_mlx_comparison.py'
    for name, output, count in [('matrix', MATRIX, 3), ('control', CONTROL, 18)]:
        status(name, repeats=count, output=str(output))
        subprocess.run([sys.executable, str(runner), str(BASE), str(OLD), str(output), '--repeats', str(count)], check=True)
    statuses = {}
    for name, config, output in [
        ('initial', 'docs/results/2026-11-11-phase2-optimization/apple-gate18-initial-config-v1.json', INITIAL),
        ('controlled', 'docs/results/2026-11-11-phase2-optimization/apple-gate18-controlled-config-v1.json', FINAL),
    ]:
        status('audit_'+name)
        statuses[name] = subprocess.run([sys.executable, 'tools/bench/audit_phase2_performance.py', config, str(output)], check=False).returncode
    status('complete', audit_returncodes=statuses)
    print('Audit return codes:', statuses, flush=True)
    return 0 if statuses['controlled'] == 0 else 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        if PRE.is_dir() and (OWNED_STATUS or not STATUS.exists()):
            status('failed' if OWNED_STATUS else 'preflight_failed', error=str(exc))
        raise
