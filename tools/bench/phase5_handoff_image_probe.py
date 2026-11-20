"""Classify executable identities during a CPU-only fake-handoff failure.

Test-only instrumentation; mismatches still fail closed. Only booleans for
fixed identities are recorded, never unexpected paths, inodes or argv.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'tools/bench'


def categories(images, collector, workload, launcher):
    def flags(image):
        return {'collector': image == collector, 'workload': image == workload,
                'launcher': image == launcher}
    return {'samples': [flags(image) for image in images]}


def install(destination):
    """Wrap only the fixture worker's observer; never accept a mismatch."""
    import observe_readonly_process as observer
    original = observer.ProcessObservation.observe
    stat = observer.os.stat
    collector_stat = stat('/proc/self/exe')
    collector = (collector_stat.st_dev, collector_stat.st_ino)

    def observe(self, pid):
        images = []
        watched = f'/proc/{pid}/exe'
        def stat_probe(path, *args, **kwargs):
            result = stat(path, *args, **kwargs)
            if os.fspath(path) == watched:
                images.append((result.st_dev, result.st_ino))
            return result
        observer.os.stat = stat_probe
        try:
            return original(self, pid)
        except observer.ObservationMismatch as error:
            if str(error) == 'unexpected live executable identity':
                launcher = observer.identity_key(self.launcher) if self.launcher else None
                payload = categories(images, collector, observer.identity_key(self.workload), launcher)
                with Path(destination).open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(payload, sort_keys=True) + '\n')
            raise
        finally:
            observer.os.stat = stat
    observer.ProcessObservation.observe = observe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--attempts', type=int, default=30)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.resolve().is_relative_to((ROOT / 'reports').resolve()):
        raise ValueError('new output under reports required')
    if not 1 <= args.attempts <= 50:
        raise ValueError('attempts must be 1..=50')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = {'PATH': os.defpath, 'PYTHONNOUSERSITE': '1', 'LANG': 'C', 'LC_ALL': 'C'}
    entries = []
    for attempt in range(args.attempts):
        diag = output / f'{attempt:02}-categories.jsonl'
        # Modify the fake fixture's worker string in the subprocess, not production code.
        injection = f'from phase5_handoff_image_probe import install\ninstall({str(diag)!r})\n'
        code = ("import unittest; import test_clock_worker_handoff as t; "
                "needle='root=Path(sys.argv[2]);parent=int(sys.argv[3]);mode=sys.argv[4]'; "
                f"inject={injection!r}+needle; "
                "assert t.WORKER.count(needle)==1; t.WORKER=t.WORKER.replace(needle,inject); "
                "suite=unittest.defaultTestLoader.loadTestsFromName('test_clock_worker_handoff.ClockHandoffTests.test_default_budget_handoff_preserves_deadline_and_all_eighteen_real_observations'); "
                "result=unittest.TextTestRunner(verbosity=1).run(suite); "
                "raise SystemExit(not (result.wasSuccessful() and result.testsRun==1 and not result.skipped))")
        with (output / f'{attempt:02}.stdout').open('xb') as stdout, (output / f'{attempt:02}.stderr').open('xb') as stderr:
            process = subprocess.run([sys.executable, '-c', code], cwd=HERE, env=env,
                                     stdout=stdout, stderr=stderr, timeout=30, check=False)
        samples = [json.loads(line) for line in diag.read_bytes().splitlines()] if diag.exists() else []
        entry = {'attempt': attempt, 'exit': process.returncode, 'identity_categories': samples}
        entries.append(entry)
        print(json.dumps(entry, sort_keys=True), flush=True)
        if process.returncode or samples:
            break
    with (output / 'summary.json').open('x', encoding='utf-8') as stream:
        json.dump({'scope': 'CPU-only fake handoff; no observer acceptance or policy change', 'attempts': entries},
                  stream, indent=2, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    main()
