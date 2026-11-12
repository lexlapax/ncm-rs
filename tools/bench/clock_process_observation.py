"""Construct fixed-job observation callbacks without file IO or hashing.

Import-only. The caller freezes/verifies inputs before telemetry and rechecks them
only after all collection/supervision ends. Supplied identities are declarations,
not independent provenance. No GPU queries, policy commands or launch entry here.
"""
from copy import deepcopy
import os
from pathlib import Path
import re

from observe_readonly_process import ProcessObservation
from readonly_trace_preflight import require

RUNTIMES = ('previous', 'python', 'candidate')
ORDER = ('previous', 'python', 'candidate', 'python', 'candidate', 'previous',
         'candidate', 'previous', 'python') * 2


def identity(value):
    require(isinstance(value, dict) and isinstance(value.get('path'), str)
            and '\0' not in value['path'] and Path(value['path']).is_absolute()
            and type(value.get('device')) is int and value['device'] >= 0
            and type(value.get('inode')) is int and value['inode'] > 0
            and type(value.get('bytes')) is int and value['bytes'] > 0
            and isinstance(value.get('sha256'), str)
            and re.fullmatch('[0-9a-f]{64}', value['sha256']) is not None,
            'complete prelaunch file identity required')


def process_factory(jobs, *, env, workloads, taskset, injection, libraries):
    """Prebuild all 18 independent observers; reject command/env drift on issuance.

    workloads maps each runtime to {command_path, image}. command_path preserves
    the declared argv spelling (including a reviewed Python-venv symlink); image
    is the separately verified ELF identity. libraries maps runtimes to nonempty
    prelaunch required-library lists. Lazy/unlisted mappings remain post-only.
    No filesystem reads, hashes, metadata subprocesses or path resolution occur.
    The trusted callback may issue each ordinal once, never reuse observer state.
    """
    jobs, env, workloads, taskset, injection, libraries = deepcopy(
        (jobs, env, workloads, taskset, injection, libraries))
    fixed_env = {'PATH': os.defpath, 'LANG': 'C', 'LC_ALL': 'C',
                 'PYTHONNOUSERSITE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
    require(isinstance(env, dict) and set(env) == set(fixed_env) | {'HOME'}
            and all(env[k] == v for k, v in fixed_env.items())
            and isinstance(env['HOME'], str) and '\0' not in env['HOME']
            and Path(env['HOME']).is_absolute(), 'isolated observation environment required')
    require(isinstance(jobs, list) and len(jobs) == 18, 'eighteen frozen jobs required')
    require(isinstance(workloads, dict) and set(workloads) == set(RUNTIMES)
            and isinstance(libraries, dict) and set(libraries) == set(RUNTIMES),
            'all runtime images and required libraries must be declared')
    identity(taskset)
    identity(injection)
    for runtime in RUNTIMES:
        value = workloads[runtime]
        require(isinstance(value, dict) and set(value) == {'command_path', 'image'}
                and isinstance(value['command_path'], str) and '\0' not in value['command_path']
                and Path(value['command_path']).is_absolute(), 'explicit runtime command path required')
        identity(value['image'])
        require(isinstance(libraries[runtime], list) and libraries[runtime],
                'nonempty required-library inventory for every runtime required')
        for library in libraries[runtime]:
            identity(library)
    observers = []
    for index, job in enumerate(jobs):
        runtime = ORDER[index]
        require(isinstance(job, dict) and job.get('runtime') == runtime
                and type(job.get('repeat')) is int and job['repeat'] == index // 3 + 1
                and job.get('workload') == 'codec' and isinstance(job.get('argv'), list)
                and len(job['argv']) >= 5
                and job['argv'][:4] == [taskset['path'], '-c', '2', workloads[runtime]['command_path']],
                'frozen codec order/launcher/runtime command mismatch')
        observers.append(ProcessObservation(
            workload=workloads[runtime]['image'], workload_argv=job['argv'][3:],
            expected_env=env, cpu=2, injection=injection, traced=False,
            required_libraries=libraries[runtime], launcher=taskset, launcher_argv=job['argv']))
    issued = set()

    def make(ordinal, job, actual_env):
        require(type(ordinal) is int and 1 <= ordinal <= 18, 'invalid observation ordinal')
        require(job == jobs[ordinal - 1] and actual_env == env,
                'observation job or environment differs from frozen declarations')
        require(ordinal not in issued, 'process observer already issued')
        issued.add(ordinal)
        return observers[ordinal - 1]

    return make
