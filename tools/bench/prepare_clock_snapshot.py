"""Assemble current file-only clock preparation; never launch or change clocks.

This is an intermediate preparation, not the frozen role entry, hardware readiness,
permission check, or collection authorization. It performs bounded Git/loader
metadata probes but does not import Torch or invoke any GPU utility.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from clock_preparation_inventory import compose
from clock_process_observation import ORDER
from clock_evidence import META_BYTES
from prepare_clock_libraries import inspect as inspect_libraries
from prepare_clock_preloads import inventory as inspect_preloads
from prepare_clock_python import prepare as inspect_python
from prepare_clock_upstream import inspect as inspect_upstream
from prepare_readonly_launch import timezone_identity, tool_inventory
from readonly_trace_preflight import file_identity, prepare as inspect_native, require
from readonly_trace_runtime import verify_worktrees
from run_phase2_linux_comparison import plan_jobs
from verify_clock_preparation_files import verify as verify_files

STARTUP = {'distutils-precedence.pth': '2638ce9e2500e572a5e0de7faed6661eb569d1b696fcba07b0dd223da5f5d224',
           '_distutils_hack/__init__.py': 'ced72e54431ecf2d8d7dbd8a1e27724f1b171af3ca1503c304bf11c59a7fe861'}
TORCH_SOURCES = {'__init__.py': '7ed81103f3d256de131b886e67be458b3205784e3b4af9c495041282391e19db',
                 '_utils_internal.py': 'eb77101e63128e55da13b29c7133e42b702a00e0b6eb1d44521ba1944178e794',
                 'version.py': '6feda231420514c053c7116e00e36d3313d6708aa1ab2a93dfa07ad4e5735184'}
UPSTREAM_CONFIG_SHA = 'e11456e4bc00390f8f42b98eac749b4e829771b7f59283376b33b8b9ca0d630c'


def validate_jobs(jobs, *, root, collection, taskset, contract):
    """Independently check the fixed 18 codec commands and output locations."""
    root, collection = Path(root), Path(collection)
    directory = collection/'sequence'
    require(isinstance(jobs, list) and len(jobs) == len(ORDER), 'exactly eighteen planned jobs required')
    for ordinal, (job, runtime) in enumerate(zip(jobs, ORDER)):
        repeat = ordinal//3 + 1
        kind = 'python' if runtime == 'python' else 'rust'
        report = str(directory/runtime/f'codec-{kind}-{repeat}.json')
        cwd = root/'.reference/phase2-optimization-baseline' if runtime == 'previous' else root
        argv = [str(taskset), '-c', '2', contract['workloads'][runtime]['command_path']]
        if runtime == 'python':
            argv += ['tools/reference/codec_bench.py', str(root/'artifacts/canonical/sudoku-easy-qualified'),
                     str(root/'.reference/upstream-ncm')]
        else:
            argv += [str(root/'artifacts/canonical/sudoku-easy-qualified')]
        argv += ['tests/fixtures/codec-bench-v1.json', report]
        copies = [str(directory/d/Path(report).name) for d in ('previous', 'candidate')] if runtime == 'python' else []
        require(job == {'runtime': runtime, 'repeat': repeat, 'workload': 'codec', 'argv': argv,
                        'cwd': str(cwd), 'report': report, 'python_copies': copies},
                'planned command/order/cwd/output differs from frozen codec protocol')


def prepare(root, collection, evidence, *, candidate_head, home, python, git, taskset, nvidia_smi):
    """Build one private metadata snapshot, but never claim or start collection.

    Existing helpers supply pinned input checks and bounded static loader/Git
    probes. All producer records are composed and their files rehashed last.
    A reviewed future entry must freshly repeat/bind these results, construct and
    authenticate actual role commands, and separately establish GPU identity,
    policy, permissions, restoration and an explicit quiet window.
    """
    require(sys.flags.no_site, 'preparation requires Python -S')
    root, collection, evidence, home, python, git, taskset, nvidia_smi = map(
        Path, (root, collection, evidence, home, python, git, taskset, nvidia_smi))
    require(all(p.is_absolute() for p in (root, collection, evidence, home, python, git, taskset, nvidia_smi))
            and root.resolve(strict=True) == root and home.resolve(strict=True) == home
            and collection.parent.resolve(strict=True) == collection.parent
            and evidence.parent.resolve(strict=True) == evidence.parent
            and not os.path.lexists(collection) and not os.path.lexists(evidence)
            and not evidence.is_relative_to(collection) and not collection.is_relative_to(evidence),
            'fresh disjoint canonical preparation/collection paths required')
    require(isinstance(candidate_head, str) and re.fullmatch('[0-9a-f]{40}', candidate_head),
            'full candidate commit required')
    # Prepare native declarations before any output creation. This validates CPU
    # topology and all archived native input pins, but does not launch a workload.
    native = inspect_native(root, collection/'sequence', home, taskset=taskset, reserved_parent=collection)
    env = dict(native['observer_env'])
    env['TOKENIZERS_PARALLELISM'] = 'false'
    evidence.mkdir(mode=0o700)
    worktrees = verify_worktrees(root, candidate_head, evidence/'worktrees', git=git)
    upstream = inspect_upstream(root, root/'.reference/upstream-ncm', evidence/'upstream',
                                git_path=git, home=home, config_sha=UPSTREAM_CONFIG_SHA)
    py = inspect_python(root, reviewed_startup=STARTUP)
    site = Path(py['site_packages'])
    preloads = inspect_preloads(site, reviewed_sources=TORCH_SOURCES)
    extensions = list((site/'torch').glob('_C*.so'))
    require(len(extensions) == 1 and extensions[0].resolve(strict=True) == extensions[0],
            'one canonical Torch extension required')
    native_by_path = {v['path']: v for v in native['file_identities']}
    previous = root/'.reference/phase2-optimization-baseline-bin/codec_bench'
    candidate = root/'target/release/examples/codec_bench'
    injection = root/'.reference/phase2-cupti/libcupti_trace_injection.so'
    for path in (previous, candidate, injection, taskset):
        require(str(path) in native_by_path, 'native preparation missing expected image')
    native_sonames = ['libc.so.6', 'libcuda.so.1', 'libcurand.so.10', 'libcublas.so.13']
    minor = '.'.join(py['venv_version_metadata'].split('.')[:2])
    targets = {
        'previous': {'image': native_by_path[str(previous)], 'required': native_sonames},
        'candidate': {'image': native_by_path[str(candidate)], 'required': native_sonames},
        'python': {'image': py['interpreter_image'], 'required': ['libc.so.6', f'libpython{minor}.so.1.0']},
        'torch_extension': {'image': file_identity(extensions[0]), 'required': ['libc.so.6', 'libtorch_python.so', 'libtorch_cpu.so', 'libtorch_cuda.so']},
        'torch_global_deps': {'image': file_identity(site/'torch/lib/libtorch_global_deps.so'), 'required': ['libc.so.6', 'libcudart.so.13']},
        'taskset': {'image': native_by_path[str(taskset)], 'required': ['libc.so.6']},
    }
    loader_path = Path('/usr/lib/ld-linux-x86-64.so.2').resolve(strict=True)
    roots = [Path('/usr/lib'), Path('/opt/cuda/targets/x86_64-linux/lib'), site]
    loader = inspect_libraries(targets, loader_identity=file_identity(loader_path),
                               environment=native['observer_env'], library_roots=roots,
                               output_dir=evidence/'libraries')
    utils = [file_identity(python), file_identity(nvidia_smi), worktrees['git_identity']]
    tz = timezone_identity()
    tools = tool_inventory(root)
    groups = {'tools': tools, 'utilities': utils, 'timezone': [tz['file_identity']]}
    aliases = [{'path': tz['localtime_path'], 'canonical_path': tz['file_identity']['path']}]
    inventory = compose(native, py, loader, preloads, upstream_record=upstream,
                        upstream_tree=root/'.reference/upstream-ncm', injection=native_by_path[str(injection)],
                        extra_groups=groups, path_bindings=aliases)
    jobs = plan_jobs(root/'.reference/phase2-optimization-baseline', previous.parent, collection/'sequence',
                     root=root, repeats=6, workloads=('codec',), cpu=2, taskset=str(taskset))
    validate_jobs(jobs, root=root, collection=collection, taskset=taskset,
                  contract=inventory['library_contract'])
    verified = verify_files(inventory['library_contract'], frozen_identities=inventory['frozen_identities'],
                            path_bindings=inventory['path_bindings'], injection=native_by_path[str(injection)])
    require(not os.path.lexists(collection), 'collection path created during preparation')
    result = {'schema_version': 1, 'kind': 'clock-file-snapshot', 'project_root': str(root),
              'collection_root': str(collection), 'candidate_head': candidate_head, 'environment': env,
              'native': native, 'python': py, 'loader': loader, 'preloads': preloads,
              'upstream': upstream, 'worktrees': worktrees, 'timezone': tz, 'utilities': utils,
              'tool_identities': tools, 'inventory': inventory, 'precollection_files': verified, 'jobs': jobs,
              'launch_ready': False, 'collection_started': False, 'gpu_identity_verified': False,
              'hardware_policy_verified': False, 'permissions_verified': False,
              'restoration_verified': False, 'availability_confirmed': False,
              'scope': 'File-only metadata snapshot and planned workload commands; no complete role entry, '
                       'GPU query, policy/permission/restoration check, runtime provenance or quiet-window claim.'}
    raw = json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode()+b'\n'
    require(len(raw) <= META_BYTES, 'preparation snapshot exceeds metadata bound')
    fd = os.open(evidence/'snapshot.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        require(stream.write(raw) == len(raw), 'short preparation snapshot write')
    return {'snapshot': str(evidence/'snapshot.json'), 'sha256': hashlib.sha256(raw).hexdigest(),
            'declared_file_count': verified['declared_file_count'], 'launch_ready': False,
            'collection_started': False}
