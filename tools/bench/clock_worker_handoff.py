"""Validate the armed clock worker handoff without hashing frozen payloads/tools.

Import-only: caller supplies reviewed role commands/images/environment and private
paths. Consistency of declarations and live procfs samples is not independent
producer authentication, policy authorization or a complete launch entry.
"""
import math
import os
from pathlib import Path
import signal
import stat
import sys
import threading
import time

from analyze_readonly_trace import read_bounded
from finalize_readonly_trace import parsed
from observe_readonly_process import environment, process_stat, read_proc
from readonly_trace_preflight import require
from supervised_worker import direct_children, subreaper

OBLIGATION_BYTES = 1024 ** 2


def validate_handoff(path, *, parent_pid, parent_argv, commands, role_image,
                     taskset_path, cwd, env):
    """Return the persisted absolute work deadline, never a restarted allowance.

    All role argv refer to post-taskset interpreter commands except commands.worker,
    which must retain its CPU0 taskset prefix. The restorer's set/worker/reset argv
    are compared with reviewed declarations. No command is executed by this helper.
    The future entry must bind these declarations to frozen preparation and must
    turn any exception/failed sequence into a nonzero worker exit.
    """
    require(sys.platform == 'linux' and sys.flags.no_site == 1
            and threading.current_thread() is threading.main_thread() and threading.active_count() == 1
            and len(list(Path('/proc/self/task').iterdir())) == 1
            and signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL
            and not direct_children() and subreaper() == 0,
            'fresh -S Linux worker with exclusive reaping required')
    require(type(parent_pid) is int and parent_pid > 1 and os.getppid() == parent_pid,
            'expected live restoration parent required')
    require(isinstance(commands, dict) and set(commands) == {'set', 'worker', 'reset'},
            'reviewed restoration commands required')
    for argv in [parent_argv, *commands.values()]:
        require(isinstance(argv, list) and argv
                and all(isinstance(arg, str) and arg and '\0' not in arg for arg in argv)
                and Path(argv[0]).is_absolute(), 'absolute role command required')
    require(isinstance(taskset_path, str) and Path(taskset_path).is_absolute()
            and commands['worker'][:3] == [taskset_path, '-c', '0']
            and len(commands['worker']) >= 6, 'CPU0 worker launcher required')
    fixed_env = {'PATH': os.defpath, 'LANG': 'C', 'LC_ALL': 'C',
                 'PYTHONNOUSERSITE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
    require(isinstance(env, dict) and set(env) == set(fixed_env) | {'HOME'}
            and all(env[k] == v for k, v in fixed_env.items())
            and isinstance(env['HOME'], str) and '\0' not in env['HOME']
            and Path(env['HOME']).is_absolute(), 'isolated role environment required')
    require(dict(os.environ) == env and os.getcwd() == str(cwd) and os.sched_getaffinity(0) == {0},
            'worker environment/cwd/affinity mismatch')
    require(isinstance(role_image, dict) and type(role_image.get('device')) is int
            and type(role_image.get('inode')) is int and role_image['inode'] > 0,
            'reviewed role image required')
    expected_image = role_image['device'], role_image['inode']
    path = Path(path)
    require(path.is_absolute() and path.name == 'obligation.json', 'explicit obligation path required')
    directory = path.parent.lstat()
    require(stat.S_ISDIR(directory.st_mode) and stat.S_IMODE(directory.st_mode) == 0o700
            and directory.st_uid == os.geteuid(), 'private owned restoration directory required')
    record = parsed(read_bounded(path, OBLIGATION_BYTES))
    require(isinstance(record, dict) and type(record.get('schema_version')) is int and record['schema_version'] == 2
            and record.get('armed_before_set') is True
            and type(record.get('supervisor_pid')) is int and record['supervisor_pid'] == parent_pid
            and type(record.get('supervisor_parent_pid')) is int,
            'armed obligation parent binding mismatch')
    require(type(record.get('maximum_reset_attempts')) is int and record['maximum_reset_attempts'] == 2,
            'fixed reset attempt count required')
    for field, expected in (('lock_seconds', 600), ('command_seconds', 10),
                            ('cleanup_reserve_seconds', 10)):
        require(type(record.get(field)) in (int, float) and record[field] == expected,
                'fixed restoration budget mismatch')
    for name, argv in commands.items():
        require(record.get(name + '_argv') == argv, 'obligation command differs from reviewed role')
    work, reset = (record.get(name + '_deadline_monotonic_seconds') for name in ('work', 'reset'))
    require(all(type(value) in (int, float) and math.isfinite(value) for value in (work, reset))
            and work == reset - 10, 'absolute work/reset deadline mismatch')
    clock = record['clock']
    before, after = clock['monotonic_before_ns'], clock['monotonic_after_ns']
    require(type(before) is int and type(after) is int and 0 <= before <= after
            and reset - 600 <= before / 1e9, 'obligation clock predates armed deadline')
    parent_before = process_stat(read_proc(f'/proc/{parent_pid}/stat', 65536))
    require(parent_before[:2] == (parent_pid, record['supervisor_parent_pid']),
            'restoration parent ancestry changed')
    for pid, argv in ((parent_pid, parent_argv), (os.getpid(), commands['worker'][3:])):
        require(read_proc(f'/proc/{pid}/cmdline', 65536) == b'\0'.join(arg.encode() for arg in argv) + b'\0',
                'live role command mismatch')
        image = os.stat(f'/proc/{pid}/exe')
        require((image.st_dev, image.st_ino) == expected_image, 'live role image mismatch')
        require(os.sched_getaffinity(pid) == {0}, 'live role affinity mismatch')
        require(environment(read_proc(f'/proc/{pid}/environ', 65536))
                == {key.encode(): value.encode() for key, value in env.items()}, 'live role environment mismatch')
    require(read_proc(f'/proc/{parent_pid}/task/{parent_pid}/children', 65536).split() == [str(os.getpid()).encode()],
            'worker is not the sole restoration child')
    require(process_stat(read_proc(f'/proc/{parent_pid}/stat', 65536)) == parent_before
            and os.getppid() == parent_pid, 'restoration parent changed during handoff')
    now = time.monotonic()
    require(after / 1e9 <= now < work <= now + 590, 'expired or extended work deadline')
    return work
