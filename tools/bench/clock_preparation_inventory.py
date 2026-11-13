"""Combine prepared clock inputs without IO; not a complete launch preparation.

The caller supplies fresh producer records and explicit inventories for sources
not covered by those components. This module does not authenticate the records,
check current paths, prove inventory completeness, or authorize collection.
"""
from copy import deepcopy
from pathlib import Path
import re

from clock_library_contract import compile_contract, signature
from prepare_clock_python import UPSTREAM
from readonly_trace_preflight import require
from verify_clock_files import canonical_spelling, declared_inventory, frozen_signature

# These must be populated by the eventual preparation entry; explicit categories
# prevent an omitted group being silently mistaken for an empty verified inventory.
EXTRA_GROUPS = frozenset({'tools', 'utilities', 'timezone'})


def upstream_group(record, *, tree):
    """Validate producer scope and extract every pinned upstream source and Git input.

    Pure consistency check only: caller must authenticate the fresh producer and
    independently recheck current files. An asserted flag is not provenance.
    """
    record = deepcopy(record)
    tree = canonical_spelling(str(tree))
    require(isinstance(record, dict) and type(record.get('schema_version')) is int
            and record['schema_version'] == 1 and record.get('upstream_commit') == UPSTREAM
            and all(record.get(k) is True for k in ('current_upstream_snapshot_verified',
                    'tracked_clean', 'replacement_objects_disabled', 'raw_commit_sha1_verified'))
            and all(record.get(k) is False for k in ('launch_ready', 'python_import_closure_verified',
                    'historical_interpreter_library_bytes_verified', 'atomic_snapshot_verified'))
            and type(record.get('untracked_count')) is int and record['untracked_count'] == 0
            and type(record.get('ignored_entry_count')) is int and record['ignored_entry_count'] == 0,
            'unsupported upstream snapshot scope')
    rows = record.get('files')
    require(isinstance(rows, list) and 0 < len(rows) <= 512
            and type(record.get('tracked_file_count')) is int and record['tracked_file_count'] == len(rows)
            and type(record.get('tracked_bytes')) is int, 'invalid upstream source inventory')
    identities, names = [], set()
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get('relative_path'), str),
                'invalid upstream source entry')
        name = row['relative_path']
        path = Path(name)
        require(name and not path.is_absolute() and '..' not in path.parts and str(path) == name
                and name not in names and row.get('git_mode') in ('100644', '100755')
                and isinstance(row.get('git_blob_sha1'), str)
                and re.fullmatch('[0-9a-f]{40}', row['git_blob_sha1']), 'unsupported upstream source entry')
        value = row['identity']
        frozen_signature(value)
        require(value['path'] == str(tree / path) and value['bytes'] <= 8 * 1024**2,
                'upstream source path or size differs')
        names.add(name); identities.append(value)
    require(sum(value['bytes'] for value in identities) == record['tracked_bytes'] <= 64 * 1024**2,
            'upstream byte inventory differs')
    for key in ('git_identity', 'config_identity', 'index_identity'):
        signature(record[key])
        identities.append(record[key])
    require(record['config_identity']['path'] == str(tree / '.git/config')
            and record['index_identity']['path'] == str(tree / '.git/index'),
            'upstream Git state identity paths differ')
    return identities


def compose(native, python, loader, preloads, *, upstream_record, upstream_tree,
            injection, extra_groups, path_bindings):
    """Merge all identities exposed by the four preparers and bounded extras.

    The returned lists are suitable for early admission and eventual rehash, but
    their origins and coverage must be authenticated by a separate fresh entry.
    No filesystem access, probing, hashing, subprocess, or runtime claim occurs.
    """
    native, python, loader, preloads, injection, extra_groups, path_bindings = deepcopy(
        (native, python, loader, preloads, injection, extra_groups, path_bindings))
    require(type(native.get('schema_version')) is int and native['schema_version'] == 1
            and native.get('diagnostic_only') is True and native.get('launch_ready') is False
            and native.get('availability_confirmed') is False,
            'native preparation scope differs')
    contract = compile_contract(python, loader, preloads)
    require(isinstance(extra_groups, dict) and set(extra_groups) == EXTRA_GROUPS
            and all(isinstance(group, list) and group for group in extra_groups.values()),
            'all explicit extra inventory groups required')
    historical = python['historical']
    groups = {
        'native_files': native['file_identities'],
        'native_declarations': native['declaration_identities'],
        'python_sources': python['historical_source_identities'],
        'python_current': python['current_only_file_identities'],
        'historical_evidence': [historical['declaration'], historical['archive']],
        'loader': [loader['loader_identity'], *(v['image'] for v in loader['targets'].values())],
        'preload_sources': list(preloads['current_only_sources'].values()),
        'upstream': upstream_group(upstream_record, tree=upstream_tree),
        **extra_groups,
    }
    require(all(isinstance(values, list) and values for values in groups.values()),
            'each preparation group must be nonempty')
    merged = {}
    for values in groups.values():
        for value in values:
            identity = frozen_signature(value)
            path = value['path']
            require(path not in merged or frozen_signature(merged[path]) == identity,
                    'conflicting preparation identity')
            merged[path] = value
    # The source preparation identifies the injection explicitly. A separate
    # declared identity cannot replace or introduce a previously omitted file.
    require(injection['path'] in merged and signature(merged[injection['path']]) == signature(injection),
            'injection absent or different in preparation')
    native_paths = {v['path']: v for v in native['file_identities']}
    for role in ('previous', 'candidate', 'taskset'):
        value = contract['taskset'] if role == 'taskset' else contract['workloads'][role]['image']
        require(value['path'] in native_paths and signature(native_paths[value['path']]) == signature(value),
                'native and loader image identities differ')
    require(isinstance(path_bindings, list), 'explicit extra path bindings required')
    # Preserve interpreter chain spellings; the entry must separately verify each
    # link target and the final postflight resolves each spelling to the image.
    links = python['interpreter_links']
    require(isinstance(links, list), 'interpreter link declaration required')
    image = python['interpreter_image']['path']
    for link in links:
        require(isinstance(link, dict) and set(link) == {'path', 'target'}
                and isinstance(link['target'], str) and link['target']
                and '\0' not in link['target'], 'invalid interpreter link declaration')
        path_bindings.append({'path': link['path'], 'canonical_path': image})
    frozen = [merged[path] for path in sorted(merged)]
    pinned, aliases = declared_inventory(contract, frozen_identities=frozen,
                                         path_bindings=path_bindings, injection=injection)
    return {'schema_version': 1, 'library_contract': contract, 'frozen_identities': frozen,
            'path_bindings': path_bindings, 'declared_file_count': len(pinned),
            'declared_file_bytes': sum(value['bytes'] for value in pinned.values()),
            'declared_alias_count': len(aliases),
            'group_counts': {name: len(values) for name, values in sorted(groups.items())},
            'declaration_admission_passed': True, 'current_files_rechecked': False,
            'inventory_completeness_verified': False, 'launch_ready': False,
            'scope': 'Pure supplied-record composition and postflight-bound admission only; '
                     'not fresh preparation, complete inventory, symlink-chain proof, '
                     'producer authentication, runtime observation or launch permission.'}
