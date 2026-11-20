"""Investigate a CPU-only lifecycle fixture's fail-closed environment mismatch.

Only runs the existing fake owned-child fixture; no GPU or machine policy changes.
Mismatch diagnostics contain fixed-key booleans/counts, never environment values
or unexpected key names. A mismatch is never accepted or retried within a run.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'tools/bench'


def categories(actual, expected, parent):
    return {'same_keys': set(actual) == set(expected),
            'missing_count': len(set(expected) - set(actual)),
            'extra_count': len(set(actual) - set(expected)),
            'same_as_parent_proc': actual == parent,
            'known_key_values_match': {key.decode('ascii'): actual.get(key) == value
                                       for key, value in expected.items()}}


def install(output):
    """Instrument only the fake collector's environment parser, preserving failure."""
    import observe_readonly_process as observer
    original = observer.environment
    expected = {b'PATH': os.defpath.encode(), b'PYTHONNOUSERSITE': b'1', b'LANG': b'C', b'LC_ALL': b'C'}

    def capture(raw):
        actual = original(raw)
        if actual != expected:
            parent = original(Path('/proc/self/environ').read_bytes())
            with Path(output).open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(categories(actual, expected, parent), sort_keys=True) + '\n')
        return actual
    observer.environment = capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--attempts', type=int, default=30)
    args = parser.parse_args()
    if (args.output.exists() or args.output.is_symlink()
            or not args.output.resolve().is_relative_to((ROOT / 'reports').resolve())):
        raise ValueError('new output under reports required')
    if not 1 <= args.attempts <= 50:
        raise ValueError('attempts must be 1..=50')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = {'PATH': os.defpath, 'PYTHONNOUSERSITE': '1', 'LANG': 'C', 'LC_ALL': 'C'}
    results = []
    for index in range(args.attempts):
        attempt = out / f'{index:02}'
        attempt.mkdir()
        diag = attempt / 'environment-categories.jsonl'
        code = ('from phase5_lifecycle_env_probe import install; '
                f'install({str(diag)!r}); '
                'from test_readonly_trace_coordinator import collection_driver; '
                f'collection_driver({str(attempt)!r}, "lifecycle")')
        outer = ('import os,json; from pathlib import Path; from supervised_worker import supervise; '
                 f'argv=[{sys.executable!r},"-c",{code!r}]; '
                 f'result=supervise(argv,cwd=Path({str(HERE)!r}),env={{"PATH":os.defpath,"PYTHONNOUSERSITE":"1"}},'
                 f'output_dir=Path({str(attempt / "supervision")!r}),timeout_seconds=15,parent_pid=os.getppid()); '
                 'print(json.dumps(result))')
        with (attempt / 'outer.stdout').open('xb') as stdout, (attempt / 'outer.stderr').open('xb') as stderr:
            process = subprocess.run([sys.executable, '-c', outer], cwd=HERE, env=env,
                                     stdout=stdout, stderr=stderr, timeout=25, check=False)
        supervisor = json.loads((attempt / 'supervision/supervisor.json').read_bytes())
        failures = []
        for path in sorted((attempt / 'collection/sequence').glob('*.process.json')):
            record = json.loads(path.read_bytes())
            if record['observation_failures']:
                failures.append({'job': path.name, 'reasons': record['observation_failures'],
                                 'accepted_samples': len(record['samples'])})
        entry = {'attempt': index, 'exit': process.returncode, 'status': supervisor['status'],
                 'observation_failures': failures,
                 'environment_categories': [json.loads(line) for line in diag.read_bytes().splitlines()] if diag.exists() else []}
        results.append(entry)
        print(json.dumps(entry, sort_keys=True), flush=True)
        if process.returncode or supervisor['status'] != 'complete' or failures or entry['environment_categories']:
            break
    with (out / 'summary.json').open('x', encoding='utf-8') as stream:
        json.dump({'scope': 'CPU fake lifecycle fixture only; no observation acceptance or policy change',
                   'attempts': results}, stream, indent=2, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    main()
