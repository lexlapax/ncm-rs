"""Preparation comparison only; no workload, clock mutation or launch entry.

The caller supplies reviewed private snapshot/metadata digests. This code does
not authenticate the caller, policy, permissions, restoration, role ancestry or
quiet-window availability and cannot authorize collection.
"""
import hashlib
import json
import os
from pathlib import Path
import re

from clock_evidence import Evidence, META_BYTES
from compare_clock_gpu import bind_snapshot, compare as compare_gpu
from compare_clock_snapshots import WORKTREES, compare as compare_snapshots, validate as validate_snapshot
from finalize_clock_diagnostic import parsed
from prepare_clock_gpu import inspect as inspect_gpu
from prepare_clock_snapshot import prepare as prepare_snapshot
from readonly_trace_entry import fresh_interpreter
from readonly_trace_preflight import require
from readonly_trace_runtime import verify_worktrees
from verify_clock_files import canonical_spelling
from verify_clock_preparation_files import verify as verify_files

ROOT = Path(__file__).resolve().parents[2]


def read_private(path, expected_sha256):
    path = canonical_spelling(str(path))
    require(isinstance(expected_sha256, str) and re.fullmatch('[0-9a-f]{64}',expected_sha256),
            'explicit reviewed SHA-256 required')
    with Evidence(path.parent) as evidence:
        raw = evidence.raw(path.name)
        require(hashlib.sha256(raw).hexdigest() == expected_sha256, 'reviewed record digest differs')
        value = parsed(raw)
        evidence.verify_unchanged()
    return value


def assert_runner_image(snapshot):
    image = snapshot['utilities'][0]
    actual = os.stat('/proc/self/exe')
    require((actual.st_dev,actual.st_ino) == (image['device'],image['inode']), 'preflight Python image differs')


def compare_late_worktrees(reviewed, current):
    for key in WORKTREES:
        require(reviewed[key] == current[key], 'late worktree guard differs: '+key)
    for name in ('accepted','candidate'):
        for key in ('head','tracked_clean'):
            require(reviewed['states'][name][key] == current['states'][name][key],
                    'late worktree state differs: '+name+'.'+key)


def run(reviewed_snapshot_path, snapshot_sha256, reviewed_gpu_path, gpu_sha256, output_dir):
    """Repeat file and GPU metadata preparation and compare reviewed declarations.

    This can read GPU identity and supported-clock metadata; it never invokes lock,
    reset, Torch, telemetry, a benchmark or a collection runner. Late file rehash
    is after the GPU queries and repeated Git guard, but no atomic snapshot is claimed.
    """
    fresh_interpreter()
    reviewed = validate_snapshot(read_private(reviewed_snapshot_path,snapshot_sha256))
    reviewed_gpu = read_private(reviewed_gpu_path,gpu_sha256)
    require(Path(reviewed['project_root']) == ROOT, 'preflight must use its own reviewed checkout')
    assert_runner_image(reviewed)
    bind_snapshot(reviewed,reviewed_gpu)
    output = canonical_spelling(str(output_dir)); collection = canonical_spelling(reviewed['collection_root'])
    reviewed_dirs = [canonical_spelling(str(path)).parent for path in (reviewed_snapshot_path,reviewed_gpu_path)]
    require(output.parent.resolve(strict=True) == output.parent and not os.path.lexists(output)
            and not os.path.lexists(collection) and not output.is_relative_to(collection)
            and not collection.is_relative_to(output)
            and all(not output.is_relative_to(path) and not path.is_relative_to(output) for path in reviewed_dirs),
            'fresh disjoint preflight, reviewed and collection paths required')
    output.mkdir(mode=0o700)
    utilities = reviewed['utilities']; inventory = reviewed['inventory']
    taskset = inventory['library_contract']['taskset']['path']
    current_summary = prepare_snapshot(ROOT,collection,output/'files',candidate_head=reviewed['candidate_head'],
        home=reviewed['environment']['HOME'],python=utilities[0]['path'],nvidia_smi=utilities[1]['path'],
        git=utilities[2]['path'],taskset=taskset)
    current = read_private(current_summary['snapshot'],current_summary['sha256'])
    file_comparison = compare_snapshots(reviewed,current)
    expected_gpu = reviewed_gpu['gpu']
    gpu = inspect_gpu(ROOT,output/'gpu',nvidia_smi=utilities[1]['path'],
        environment=current['native']['observer_env'],expected_uuid=expected_gpu['uuid'],
        expected_pci=expected_gpu['pci_bus_id'],expected_name=expected_gpu['name'])
    gpu_raw = json.dumps(gpu,indent=2,sort_keys=True,allow_nan=False).encode()+b'\n'
    require(len(gpu_raw) <= META_BYTES, 'GPU metadata exceeds bound')
    gpu_fd = os.open(output/'gpu/result.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(gpu_fd,'wb') as stream:
        require(stream.write(gpu_raw)==len(gpu_raw), 'short GPU metadata write')
    current_gpu_sha256 = hashlib.sha256(gpu_raw).hexdigest()
    require(read_private(output/'gpu/result.json',current_gpu_sha256) == gpu,
            'current GPU metadata changed during publication')
    gpu_comparison = compare_gpu(reviewed_gpu,gpu,root=ROOT)
    gpu_binding = bind_snapshot(current,gpu)
    late = verify_worktrees(ROOT,reviewed['candidate_head'],output/'worktrees-late',git=Path(utilities[2]['path']))
    compare_late_worktrees(reviewed['worktrees'],late)
    injection_path = str(ROOT/'.reference/phase2-cupti/libcupti_trace_injection.so')
    injection = next(value for value in current['native']['file_identities'] if value['path'] == injection_path)
    files = verify_files(inventory['library_contract'],frozen_identities=inventory['frozen_identities'],
                         path_bindings=inventory['path_bindings'],injection=injection)
    require(files['supplied_files_rechecked'] is True and files['supplied_aliases_rechecked'] is True
            and files['launch_ready'] is False and not os.path.lexists(collection),
            'late file recheck incomplete or collection created')
    result = {'schema_version':1,'kind':'clock-readonly-preflight','reviewed_snapshot_sha256':snapshot_sha256,
        'reviewed_gpu_sha256':gpu_sha256,'current_snapshot_sha256':current_summary['sha256'],
        'current_gpu_sha256':current_gpu_sha256,
        'file_comparison':file_comparison,'gpu_comparison':gpu_comparison,'gpu_binding':gpu_binding,
        'late_files':files,'late_worktree_state_verified':True,'collection_started':False,
        'producer_authentication_verified':False,'permissions_verified':False,'hardware_policy_verified':False,
        'restoration_verified':False,'availability_confirmed':False,'launch_ready':False,
        'scope':'One caller-supplied reviewed pair versus newly generated file/GPU metadata, '
                'late Git guard and declared-file rehash. Not producer authentication, safe '
                'role ancestry, GPU-lock permission/enforcement, restoration or launch authorization.'}
    raw = json.dumps(result,indent=2,sort_keys=True,allow_nan=False).encode()+b'\n'
    require(len(raw) <= META_BYTES, 'preflight result exceeds metadata bound')
    fd=os.open(output/'preflight.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:
        require(stream.write(raw)==len(raw), 'short preflight output')
    return result
