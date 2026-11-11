"""Read-only frozen-file checks and fixed job construction, not a launch command.

No child execution or GPU calls. Library resolution, worktree state, supervision,
telemetry and fresh user availability must still be verified before collection.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tarfile

from qualify_diagnostic_codec import EVIDENCE, NATIVE_SOURCES, ROOT
from readonly_trace_sequence import ORDER
from run_phase2_linux_comparison import validate_observer

ARTIFACT = Path('artifacts/canonical/sudoku-easy-qualified')
FIXTURE = Path('tests/fixtures/codec-bench-v1.json')
PROFILE_EVIDENCE = ROOT / 'docs/results/2026-11-10-phase2-cuda-remediation/evidence.json'
PAYLOAD_NAMES = {'codec.safetensors', 'config.json', 'generator.safetensors',
                 'provenance.json', 'tokenizer.json', 'training.safetensors'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_identity(path, expected=None, size=None):
    """Hash regular non-symlink leaves, detecting visible changes; parents are trusted."""
    path = Path(path).absolute()
    if expected is not None:
        require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected),
                'invalid expected file digest')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), 'identity input is not a regular file')
        require(size is None or before.st_size == size, 'file size mismatch: ' + str(path))
        digest, count = hashlib.sha256(), 0
        while chunk := os.read(fd, 1024 * 1024):
            count += len(chunk)
            require(count <= before.st_size, 'file grew during identity check')
            digest.update(chunk)
        after, current = os.fstat(fd), os.stat(path, follow_symlinks=False)
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        require(count == before.st_size and all(getattr(before, k) == getattr(s, k)
                                               for s in (after, current) for k in fields),
                'file changed during identity check')
        actual = digest.hexdigest()
        require(expected is None or actual == expected, 'file digest mismatch: ' + str(path))
        return dict(path=str(path), sha256=actual, bytes=count, device=before.st_dev,
                    inode=before.st_ino, expected_digest_verified=expected is not None)
    finally:
        os.close(fd)


def frozen_inputs():
    """Load only hash-bound v2 archive members; do not infer pins from current files."""
    evidence = json.loads(EVIDENCE.read_bytes())
    spec = evidence['archive']
    archive_path = ROOT / spec['path']
    file_identity(archive_path, spec['sha256'], spec['bytes'])
    members = {m['path']: m for m in spec['members']}
    require(len(members) == len(spec['members']), 'duplicate archive declarations')
    with tarfile.open(archive_path, 'r:xz') as archive:
        def member(name):
            matches = [m for m in archive.getmembers() if m.name == name]
            expected = members[name]
            require(len(matches) == 1 and matches[0].isfile()
                    and matches[0].size == expected['bytes'] <= 16 * 1024 ** 2,
                    'invalid frozen member')
            raw = archive.extractfile(matches[0]).read()
            require(len(raw) == expected['bytes'] and hashlib.sha256(raw).hexdigest() == expected['sha256'],
                    'frozen member digest mismatch')
            return json.loads(raw)
        reports = {key: member(f'phase2-linux-diagnostic-v2/{runtime}/codec-rust-1.json')
                   for key, runtime in (('A', 'previous'), ('C', 'candidate'))}
        environment = member('phase2-linux-diagnostic-v2/environment.json')
        payload = member('phase2-linux-diagnostic-v2-payload-preflight.json')
    files = [f for f in payload['files'] if Path(f['path']).parent == ARTIFACT]
    require({Path(f['path']).name for f in files} == PAYLOAD_NAMES and len(files) == len(PAYLOAD_NAMES),
            'incomplete frozen codec payload inventory')
    for report in reports.values():
        require(set(report['source_sha256']) == NATIVE_SOURCES, 'incomplete frozen source inventory')
    require(reports['A']['manifest_sha256'] == reports['C']['manifest_sha256']
            and reports['A']['workload_sha256'] == reports['C']['workload_sha256'], 'frozen inputs disagree')
    return dict(reports=reports, payload=files, archive_sha256=spec['sha256'],
                binaries={key: environment['binary_sha256'][runtime]['codec_bench']
                          for key, runtime in (('A', 'previous'), ('C', 'candidate'))})


def isolated_environment(home, inherited):
    """No inherited secrets or controls; reject contaminated launcher environments."""
    forbidden = [k for k, v in inherited.items() if v and (
        k.startswith(('LD_', 'CUPTI_', 'NVTX_', 'NCM_', 'CUBLAS_'))
        or (k.startswith('CUDA_') and k not in ('CUDA_PATH', 'CUDA_COMPUTE_CAP')))]
    require(not forbidden, 'inherited native-control variables must be cleared before preparation')
    home = Path(home).resolve(strict=True)
    require(home.is_dir(), 'HOME must be an existing directory')
    return {'PATH': os.defpath, 'HOME': str(home), 'LANG': 'C', 'LC_ALL': 'C',
            'PYTHONNOUSERSITE': '1'}


def prepare(root, output_dir, home, *, inherited=None, taskset=Path('/usr/bin/taskset'),
            reserved_parent=None):
    """Return jobs and identity evidence without writing outputs or starting children.

    Sanitized native environments differ from v2's inherited environment. All
    conditions share them; traced children differ only by CUDA_INJECTION64_PATH.
    The CUPTI sample/source pins are the retained original profiling identities.
    """
    base_env = isolated_environment(home, os.environ if inherited is None else inherited)
    root, output_dir = Path(root).resolve(strict=True), Path(output_dir).absolute()
    require(not os.path.lexists(output_dir), 'collection path already exists')
    if reserved_parent is None:
        require(output_dir.parent.is_dir(), 'collection parent must exist')
    else:
        # Coordinator will exclusively create this one future parent. Do not
        # create/remove a placeholder directory merely to construct job paths.
        reserved_parent = Path(reserved_parent).absolute()
        require(output_dir.parent == reserved_parent and not os.path.lexists(reserved_parent)
                and reserved_parent.parent.is_dir(), 'invalid reserved collection parent')
    validate_observer(2, 0)
    frozen = frozen_inputs()
    worktrees = {'A': root / '.reference/phase2-optimization-baseline', 'C': root}
    binaries = {'A': root / '.reference/phase2-optimization-baseline-bin/codec_bench',
                'C': root / 'target/release/examples/codec_bench'}
    identities = []
    for key in ('A', 'C'):
        identities.append(file_identity(binaries[key], frozen['binaries'][key]))
        require(os.access(binaries[key], os.X_OK), 'benchmark is not executable')
        for relative, expected in frozen['reports'][key]['source_sha256'].items():
            identities.append(file_identity(worktrees[key] / relative, expected))
        identities.append(file_identity(worktrees[key] / FIXTURE, frozen['reports'][key]['workload_sha256']))
    artifact = root / ARTIFACT
    identities.append(file_identity(artifact / 'manifest.json', frozen['reports']['A']['manifest_sha256']))
    for item in frozen['payload']:
        identities.append(file_identity(root / item['path'], item['sha256'], item['bytes']))
    profile = json.loads(PROFILE_EVIDENCE.read_bytes())['profiling']
    injection = root / '.reference/phase2-cupti/libcupti_trace_injection.so'
    identities.append(file_identity(injection, profile['original_injection_library_sha256']))
    for path, expected in profile['sample_sha256'].items():
        identities.append(file_identity(path, expected))
    taskset = Path(taskset).absolute()
    identities.append(file_identity(taskset))
    require(os.access(taskset, os.X_OK), 'taskset is not executable')
    jobs = []
    for ordinal, condition in enumerate(ORDER, 1):
        key = condition[0]
        report = output_dir / f'{ordinal:02d}-{condition}.report.json'
        env = dict(base_env)
        if condition.endswith('T'):
            env['CUDA_INJECTION64_PATH'] = str(injection)
        jobs.append(dict(condition=condition, cwd=str(worktrees[key]), env=env, report_path=str(report),
                         argv=[str(taskset), '-c', '2', str(binaries[key]), str(artifact),
                               str(worktrees[key] / FIXTURE), str(report)]))
    return {'schema_version': 1, 'diagnostic_only': True, 'launch_ready': False, 'jobs': jobs,
            'observer_env': base_env, 'workload_cpu': 2, 'observer_cpu': 0,
            'archive_sha256': frozen['archive_sha256'], 'file_identities': identities,
            'sources': {key: frozen['reports'][key]['source_sha256'] for key in ('A', 'C')},
            'declaration_identities': [file_identity(EVIDENCE), file_identity(PROFILE_EVIDENCE)],
            'library_resolution_verified': False, 'worktree_state_verified': False,
            'availability_confirmed': False, 'whole_run_supervision_verified': False,
            'scope': 'Read-only frozen byte checks and fixed argv/environment construction only. '
                     'No tensor validation, live executable provenance, library resolution, '
                     'observer pinning, telemetry, supervision or launch permission. No environment '
                     'inheritance beyond the explicit HOME; CUDA_PATH/COMPUTE_CAP are build hints '
                     'not propagated. Defaults and omitted runtime variables must be reviewed.'}
