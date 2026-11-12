"""Final current-file checks for clock diagnostics, not a collection/finalizer entry.

Call verify_files only after supervision, telemetry, process and output/capture
validation. declared_inventory is pure early admission, not a file recheck. Neither
function authenticates its inventory or proves immutable mapped bytes. No subprocesses,
Torch imports or GPU calls.
"""
from copy import deepcopy
import os
from pathlib import Path
import re
import stat

from clock_library_contract import signature
from readonly_trace_preflight import file_identity, require
from verify_readonly_processes import key, mapping_key

FILE_COUNT_LIMIT = 256
FILE_BYTES_LIMIT = 8 * 1024**3
FILE_SIZE_LIMIT = 2 * 1024**3
PATH_COUNT_LIMIT = 512


def canonical_spelling(value):
    require(isinstance(value, str) and '\0' not in value and '\\' not in value,
            'invalid declared path')
    path = Path(value)
    require(path.is_absolute() and '..' not in path.parts and str(path) == value,
            'absolute normalized path required')
    return path


def declared_inventory(contract, *, frozen_identities, path_bindings, injection):
    """Admit supplied declarations without any file reads/stats/resolution/hashes.

    Reject a known-impossible inventory before collection, using the same bounds as
    final postflight. This does not establish completeness/current bytes or reserve
    room for additional mappings; those must also pass the final combined gate.
    Returns independent pinned-path and alias dictionaries, not readiness evidence.
    """
    contract, frozen_identities, path_bindings, injection = deepcopy(
        (contract, frozen_identities, path_bindings, injection))
    require(type(contract['schema_version']) is int and contract['schema_version'] == 1
            and contract['prepared_declaration_consistency_verified'] is True
            and contract['runtime_mapping_verified'] is False and contract['launch_ready'] is False,
            'prepared library contract required')
    require(isinstance(frozen_identities, list) and frozen_identities
            and isinstance(path_bindings, list), 'explicit frozen inventory/path bindings required')
    require(set(contract['workloads']) == set(contract['libraries']) == {'previous','python','candidate'}
            and isinstance(contract['known_libraries'], list) and contract['known_libraries'],
            'incomplete runtime/library contract')
    pinned = {}
    def add(value):
        current = signature(value)
        require(value['bytes'] <= FILE_SIZE_LIMIT, 'declared file exceeds per-file bound')
        canonical_spelling(value['path'])
        require(value['path'] not in pinned or signature(pinned[value['path']]) == current,
                'conflicting frozen identities')
        pinned[value['path']] = value
    for value in frozen_identities: add(value)
    known = {}
    for value in contract['known_libraries']:
        require(value['path'] not in known, 'duplicate known library path')
        add(value); known[value['path']] = value
    for required in contract['libraries'].values():
        require(isinstance(required, list) and required, 'mandatory libraries required')
        for value in required:
            require(value['path'] in known and signature(value) == signature(known[value['path']]),
                    'mandatory/known identities differ')
    for value in [injection, contract['taskset'], *(w['image'] for w in contract['workloads'].values())]: add(value)
    aliases = {}
    for row in [*contract['path_bindings'], *path_bindings,
                *({'path': w['command_path'], 'canonical_path': w['image']['path']}
                  for w in contract['workloads'].values())]:
        path, target = row['path'], row['canonical_path']
        canonical_spelling(path); canonical_spelling(target)
        require(target in pinned and (path not in aliases or aliases[path] == target),
                'unbound/conflicting path binding')
        aliases[path] = target
    require(len(aliases) <= PATH_COUNT_LIMIT, 'path binding count exceeds bound')
    require(len(pinned) <= FILE_COUNT_LIMIT, 'declared file inventory exceeds count bound')
    require(sum(value['bytes'] for value in pinned.values()) <= FILE_BYTES_LIMIT,
            'declared file inventory exceeds byte bound')
    return pinned, aliases


def verify_files(binding, contract, *, frozen_identities, path_bindings, injection, library_roots):
    """Recheck all supplied frozen files, known libraries/images and path aliases last.

    `frozen_identities` must come from the future full preparation, not from current
    observations. Additional mappings get post-only hashes under reviewed roots.
    Contract aliases and workload command spellings are always checked; callers
    supply additional interpreter/utility/payload link bindings separately.
    """
    binding, injection = deepcopy((binding, injection))
    require(type(binding['schema_version']) is int and binding['schema_version'] == 1
            and binding['process_receipt_consistency_verified'] is True
            and binding['frozen_files_rechecked'] is False
            and binding['mapped_file_bytes_verified'] is False
            and binding['launch_provenance_verified'] is False, 'process consistency binding required')
    pinned, aliases = declared_inventory(contract, frozen_identities=frozen_identities,
                                         path_bindings=path_bindings, injection=injection)
    observed = binding['observed_files']
    require(isinstance(observed, dict) and observed, 'observed file inventory required')
    names = sorted(pinned.keys() | observed.keys())
    require(len(names) <= FILE_COUNT_LIMIT, 'file inventory exceeds count bound')
    roots = tuple(canonical_spelling(str(p)) for p in library_roots)
    require(roots and len(set(roots)) == len(roots) and all(p != Path('/') and p.is_dir()
            and p.resolve(strict=True) == p for p in roots), 'reviewed canonical non-root library directories required')

    def check_paths():
        for name in names:
            require(str(canonical_spelling(name).resolve(strict=True)) == name,
                    'frozen or mapped path is no longer canonical')
        for path, target in aliases.items():
            require(str(Path(path).resolve(strict=True)) == target, 'path alias target changed')
        for root in roots:
            require(root.is_dir() and root.resolve(strict=True) == root, 'library root changed')

    check_paths()
    inventory, total = [], 0
    # Inventory every file, identity and size before hashing any file.
    for name in names:
        path = Path(name)
        expected, mapping = pinned.get(name), observed.get(name)
        if mapping is not None:
            require(mapping['path'] == name and mapping['kernel_map_link_verified'] is True
                    and key(mapping)[1] == mapping_key(mapping)[1], 'invalid mapping identity')
            require(name != injection['path'] and mapping_key(mapping) != key(injection),
                    'forbidden injection mapping')
            require(expected is not None or (any(path.is_relative_to(root) for root in roots)
                    and re.fullmatch(r'.+\.so(?:\.[0-9A-Za-z_.-]+)*', path.name)),
                    'unreviewed post-only library path')
        current = path.stat(follow_symlinks=False)
        require(stat.S_ISREG(current.st_mode) and 0 <= current.st_size <= FILE_SIZE_LIMIT,
                'non-regular or oversized current file')
        actual = current.st_dev, current.st_ino
        require(mapping is None or mapping_key(mapping) == actual, 'mapped pathname identity changed')
        require(expected is None or (key(expected) == actual and expected['bytes'] == current.st_size),
                'frozen pathname identity changed')
        total += current.st_size
        require(total <= FILE_BYTES_LIMIT, 'file byte inventory exceeds bound')
        inventory.append((path, current, expected, mapping))
    files = []
    for path, initial, expected, mapping in inventory:
        if mapping is not None:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            try:
                value = os.fstat(fd)
                require(stat.S_ISREG(value.st_mode) and (value.st_dev, value.st_ino) == mapping_key(mapping)
                        and value.st_size == initial.st_size and os.read(fd, 4) == b'\x7fELF',
                        'mapped path is not the expected ELF-magic regular file')
            finally:
                os.close(fd)
        identity = file_identity(path, expected['sha256'] if expected else None, initial.st_size)
        require(key(identity) == (initial.st_dev, initial.st_ino), 'pathname changed while hashing')
        files.append({'identity': identity, 'binding': 'prelaunch_and_post' if expected else 'post_only',
                      'observed_executable_mapping': mapping is not None,
                      'kernel_map_device': mapping['device'] if mapping else None,
                      'kernel_map_inode': mapping['inode'] if mapping else None})
    # Detect visible earlier-file/link changes during later-file hashing. This is
    # not an atomic filesystem snapshot or proof against adversarial replacement.
    check_paths()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    for path, initial, _, _ in inventory:
        current = path.stat(follow_symlinks=False)
        require(all(getattr(initial, f) == getattr(current, f) for f in fields),
                'file changed during final inventory check')
    return {'schema_version': 1, 'supplied_frozen_file_bytes_rechecked': True,
            'current_mapped_paths_verified': True, 'supplied_path_bindings_rechecked': True,
            'files': files, 'bytes_hashed': total,
            'path_bindings': [{'path': p, 'canonical_path': aliases[p]} for p in sorted(aliases)],
            'mapped_file_bytes_verified': False, 'launch_provenance_verified': False,
            'finalization_order_verified': False, 'whole_run_qualified': False,
            'scope': 'Current pathname identities/hashes and supplied aliases only. Additional libraries are '
                     'post-only. Not complete preparation, immutable mapped bytes, producer authentication, '
                     'atomic filesystem snapshot, restoration, telemetry, output quality or finalization ordering.'}
