"""Capture one bounded, marked TinyStories Metal diagnostic; raw trace stays local."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time

from run_phase3_mlx_bench import monitor


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--artifact', required=True, type=Path)
    parser.add_argument('--native-prefix', required=True, type=Path)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--marker-source', required=True, type=Path)
    parser.add_argument('--marker-object', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if os.uname().sysname != 'Darwin':
        raise ValueError('Mac required')
    binary, artifact, native, source_root, marker_source, marker_object, out = (p.resolve() for p in
        (args.binary, args.artifact, args.native_prefix, args.source_root, args.marker_source,
         args.marker_object, args.output))
    if out.exists() or not binary.is_file() or not artifact.is_dir() or not source_root.is_dir():
        raise ValueError('need existing inputs and a new output directory')
    root = Path(__file__).resolve().parents[2]
    watched = [Path(__file__).resolve(), root / 'tools/bench/phase4_tinystories_signposts.patch',
               binary, marker_source, marker_object, source_root / 'Cargo.toml', source_root / 'Cargo.lock',
               native / 'lib/libmlx.dylib', native / 'lib/mlx.metallib']
    watched += list((source_root / 'crates').rglob('*.rs')) + list((source_root / 'crates').rglob('Cargo.toml'))
    watched += [p for p in artifact.rglob('*') if p.is_file()]
    before = {str(p): digest(p) for p in watched}
    out.mkdir(parents=True)
    command = ['/usr/bin/xcrun', 'xctrace', 'record', '--template', 'Metal System Trace',
               '--instrument', 'Points of Interest', '--instrument', 'Thread Activity',
               '--all-processes', '--no-prompt', '--time-limit', '180s', '--output', str(out / 'system.trace')]
    target = [str(binary), str(artifact), '128', '4096']
    meta = {'git_head': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
            'trace_command': command, 'target_command': target, 'watched_sha256': before,
            'status': 'starting', 'trace_scope': 'system-wide; raw trace remains private',
            'limitations': ['Instrumentation perturbs timing; not acceptance evidence.',
                            'Trace may not contain a slow request; no repeat is implied.']}
    meta_path = out / 'metadata.json'

    def save():
        meta_path.write_text(json.dumps(meta, indent=2) + '\n')

    stop, errors = threading.Event(), []
    thread = threading.Thread(target=monitor, args=(stop, out / 'activity.jsonl', errors))
    recorder = None
    save()
    thread.start()
    try:
        with (out / 'recorder.log').open('xb', buffering=0) as log:
            recorder = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if recorder.poll() is not None:
                    raise RuntimeError(f'recorder exited before readiness: {recorder.returncode}')
                if b'Starting recording' in (out / 'recorder.log').read_bytes():
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError('recorder readiness not observed')
            meta['status'] = 'target_running'
            save()
            with (out / 'target.json').open('xb') as stdout, (out / 'target.stderr').open('xb') as stderr:
                target_result = subprocess.run(target, cwd=out, env={'PATH': '/usr/bin:/bin'},
                                               stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                               timeout=155, check=False)
            meta['target_returncode'] = target_result.returncode
            meta['status'] = 'saving'
            save()
            meta['recorder_returncode'] = recorder.wait(timeout=200)
        data = json.loads((out / 'target.json').read_bytes())
        samples = data['trials_seconds']
        if (target_result.returncode or meta['recorder_returncode'] or data['batch'] != 128
                or data['warmups'] != 5 or len(samples) != 30 or any(not math.isfinite(x) or x <= 0 for x in samples)):
            raise ValueError('incomplete target or recorder result')
        meta['status'] = 'complete'
    finally:
        if recorder is not None and recorder.poll() is None:
            recorder.terminate()
            try:
                recorder.wait(timeout=5)
            except subprocess.TimeoutExpired:
                recorder.kill()
                recorder.wait()
        stop.set()
        thread.join(timeout=15)
        meta['stable_inputs_and_sources'] = before == {str(p): digest(p) for p in watched}
        meta['telemetry_errors'] = errors
        if meta['status'] != 'complete' or errors or thread.is_alive() or not meta['stable_inputs_and_sources']:
            meta['status'] = 'failed'
        save()
    if meta['status'] != 'complete':
        raise RuntimeError('trace capture incomplete; preserve outputs')
    print('marked diagnostic complete; verify trace markers and GPU tables separately', flush=True)


if __name__ == '__main__':
    main()
