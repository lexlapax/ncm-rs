"""Fail-closed comparison of two supplied file-only clock snapshots.

Pure comparison only. It neither creates a fresh snapshot nor authenticates its
producer and cannot authorize policy changes, launch, or acceptance.
"""
from copy import deepcopy
from pathlib import Path
import re

from prepare_clock_snapshot import validate_jobs
from readonly_trace_preflight import require
from verify_clock_files import declared_inventory

STABLE = ('project_root','collection_root','candidate_head','environment','native','python','preloads',
          'utilities','tool_identities','inventory','precollection_files','jobs')
LOADER = ('schema_version','diagnostic_only','standalone_loader_resolution_verified',
          'python_runtime_load_order_verified','runtime_loaded_libraries_verified',
          'historical_library_bytes_verified','launch_ready','loader_identity','targets','environment',
          'library_roots','listings','current_only_libraries','library_bytes')
UPSTREAM = ('schema_version','current_upstream_snapshot_verified','upstream_commit','tracked_clean',
            'untracked_count','tracked_file_count','tracked_bytes','replacement_objects_disabled',
            'raw_commit_sha1_verified','ignored_entry_count','ignored_entries_inspected',
            'git_identity','config_identity','index_identity','files','launch_ready',
            'python_import_closure_verified','historical_interpreter_library_bytes_verified','atomic_snapshot_verified')
WORKTREES = ('schema_version','worktree_state_verified','launch_ready','git_identity','production_base','manifest_base')
TIMEZONE = ('zone','localtime_path','link_target','file_identity')
FALSE_CLAIMS = ('launch_ready','collection_started','gpu_identity_verified','hardware_policy_verified',
                'permissions_verified','restoration_verified','availability_confirmed')


def validate(snapshot):
    require(isinstance(snapshot, dict) and type(snapshot.get('schema_version')) is int
            and snapshot['schema_version'] == 1 and snapshot.get('kind') == 'clock-file-snapshot'
            and all(snapshot.get(key) is False for key in FALSE_CLAIMS),
            'non-launch-ready file snapshot required')
    require(isinstance(snapshot.get('candidate_head'), str)
            and re.fullmatch('[0-9a-f]{40}', snapshot['candidate_head']), 'invalid candidate head')
    for key in STABLE: require(key in snapshot, 'missing stable snapshot field')
    inventory = snapshot['inventory']; native = snapshot['native']
    root = Path(snapshot['project_root']); collection = Path(snapshot['collection_root'])
    injection = str(root/'.reference/phase2-cupti/libcupti_trace_injection.so')
    candidates = [v for v in native['file_identities'] if v['path'] == injection]
    require(len(candidates) == 1, 'one declared injection identity required')
    pinned, aliases = declared_inventory(inventory['library_contract'],
        frozen_identities=inventory['frozen_identities'], path_bindings=inventory['path_bindings'], injection=candidates[0])
    preflight = snapshot['precollection_files']
    require(inventory['declaration_admission_passed'] is True and inventory['launch_ready'] is False
            and inventory['inventory_completeness_verified'] is False
            and preflight['supplied_files_rechecked'] is True and preflight['supplied_aliases_rechecked'] is True
            and preflight['launch_ready'] is False and preflight['inventory_completeness_verified'] is False
            and preflight['producer_authenticity_verified'] is False and preflight['atomic_snapshot_verified'] is False
            and type(preflight['declared_file_count']) is int and preflight['declared_file_count'] == len(pinned)
            and type(preflight['declared_file_bytes']) is int
            and preflight['declared_file_bytes'] == sum(v['bytes'] for v in pinned.values())
            and type(preflight['declared_alias_count']) is int and preflight['declared_alias_count'] == len(aliases),
            'snapshot declared-file recheck scope or accounting differs')
    validate_jobs(snapshot['jobs'],root=root,collection=collection,
                  taskset=inventory['library_contract']['taskset']['path'],contract=inventory['library_contract'])
    require(snapshot['worktrees']['worktree_state_verified'] is True
            and snapshot['worktrees']['states']['candidate']['head'] == snapshot['candidate_head']
            and all(snapshot['worktrees']['states'][key]['tracked_clean'] is True for key in ('accepted','candidate')),
            'snapshot worktree state differs')
    return snapshot


def compare(reviewed, current):
    """Compare stable evidence; ignore only bounded-probe receipts and observation times.

    Both arguments are supplied records. Full producer/caller authentication,
    exclusive reaping, GPU policy and readiness must be enforced by the entry.
    """
    reviewed, current = map(lambda x: validate(deepcopy(x)), (reviewed,current))
    for key in STABLE:
        require(reviewed[key] == current[key], 'prepared snapshot differs: '+key)
    for name, keys in (('loader',LOADER),('upstream',UPSTREAM),('worktrees',WORKTREES),('timezone',TIMEZONE)):
        for key in keys:
            require(reviewed[name][key] == current[name][key], 'prepared snapshot differs: '+name+'.'+key)
    for name in ('accepted','candidate'):
        for key in ('head','tracked_clean'):
            require(reviewed['worktrees']['states'][name][key] == current['worktrees']['states'][name][key],
                    'prepared snapshot differs: worktree '+name+'.'+key)
    return {'schema_version':1,'supplied_snapshot_fields_match':True,
            'fresh_producer_authenticated':False,'live_role_identity_verified':False,
            'hardware_policy_verified':False,'launch_ready':False,
            'scope':'Consistency of selected stable fields in two supplied snapshots only; '
                    'not authenticated fresh production, complete files, GPU permission, '
                    'safe ancestry, restoration or authorization to collect.'}
