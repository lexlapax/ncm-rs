"""Run one Phase 3 process and retain stdout, stderr and lifecycle measurements."""
import argparse
import json
import resource
import subprocess
import time
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--stdout', required=True, type=Path)
p.add_argument('--stderr', required=True, type=Path)
p.add_argument('--metadata', required=True, type=Path)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
command = a.command[1:] if a.command[:1] == ['--'] else a.command
if not command or any(path.exists() for path in (a.stdout, a.stderr, a.metadata)):
    p.error('a command and new output paths are required')
start = time.monotonic()
with a.stdout.open('xb') as out, a.stderr.open('xb') as err:
    result = subprocess.run(command, stdout=out, stderr=err, check=False)
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
with a.metadata.open('x') as out:
    json.dump({'command': command, 'returncode': result.returncode,
               'wall_seconds_including_load': time.monotonic() - start,
               'child_user_seconds': usage.ru_utime, 'child_system_seconds': usage.ru_stime,
               'child_peak_rss_platform_units': usage.ru_maxrss,
               'peak_rss_unit': 'KiB on Linux; bytes on macOS'}, out, indent=2)
    out.write('\n')
raise SystemExit(result.returncode)
