"""Preparation-only standalone ELF listings, NOT Python's runtime load order.

No interpreter/package execution or GPU query. Torch's ctypes preloads can change
which SONAMEs win compared with standalone --list. Do not feed all these listings
straight into mandatory runtime observations or claim full import coverage.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import re

from prepare_clock_python import verify_snapshot
from readonly_trace_preflight import file_identity, require
from readonly_trace_runtime import clean_environment, metadata_probe, parse_loader_listing

TARGETS = {'previous', 'candidate', 'python', 'torch_extension', 'torch_global_deps', 'taskset'}
FILE_BYTES = 2 * 1024 ** 3
TOTAL_BYTES = 8 * 1024 ** 3
MAX_LIBRARIES = 128


def check_identity(value):
    require(isinstance(value, dict) and isinstance(value.get('path'), str)
            and Path(value['path']).is_absolute()
            and isinstance(value.get('sha256'), str) and re.fullmatch('[0-9a-f]{64}', value['sha256'])
            and all(type(value.get(k)) is int and value[k] >= 0 for k in ('bytes', 'device', 'inode'))
            and 0 < value['bytes'] <= FILE_BYTES, 'invalid bounded ELF identity')
    verify_snapshot(value)
    with Path(value['path']).open('rb') as source:
        require(source.read(4) == b'\x7fELF', 'ELF input required')


def inspect(targets, *, loader_identity, environment, library_roots, output_dir):
    """List six already-reviewed images under explicit metadata environment.

    targets maps each TARGETS key to {image: file_identity, required: SONAME list}.
    Requirements are caller declarations, not authenticated library provenance.
    All targets/loader are rechecked before the first utility and again at end.
    Only trusted installed inputs are allowed; this is not a sandbox for ELF files.
    """
    targets, loader_identity = deepcopy(targets), deepcopy(loader_identity)
    env = clean_environment({'observer_env': environment})
    require(isinstance(targets, dict) and set(targets) == TARGETS, 'incomplete clock loader target set')
    roots = [Path(p) for p in library_roots]
    require(0 < len(roots) <= 8 and len(set(roots)) == len(roots)
            and all(p.is_absolute() and p.is_dir() and p.resolve(strict=True) == p and p != Path('/')
                    for p in roots), 'explicit canonical library roots required')
    for value in targets.values():
        require(isinstance(value, dict) and set(value) == {'image', 'required'}, 'invalid target declaration')
        names = value['required']
        require(isinstance(names, list) and 0 < len(names) <= 32
                and all(isinstance(n, str) and len(n) <= 256 and re.fullmatch(r'lib[A-Za-z0-9_.+-]+', n) for n in names)
                and len(set(names)) == len(names) and 'libc.so.6' in names, 'invalid required SONAME declarations')
        check_identity(value['image'])
    check_identity(loader_identity)
    require(os.access(loader_identity['path'], os.X_OK), 'executable loader required')
    directory = Path(output_dir).absolute()
    require(directory.parent.resolve(strict=True) == directory.parent, 'canonical output parent required')
    directory.mkdir(mode=0o700)
    libraries, listings, receipts = {}, {}, {}
    total = 0
    for key in sorted(targets):
        target = targets[key]
        text, receipt = metadata_probe([loader_identity['path'], '--list', target['image']['path']],
                                       directory, env, directory, key)
        dependencies, virtual = parse_loader_listing(text)
        require(set(target['required']) <= dependencies.keys(), 'missing declared static dependency')
        resolved = {}
        for name, spelling in dependencies.items():
            canonical = Path(spelling).resolve(strict=True)
            require(any(canonical.is_relative_to(root) for root in roots), 'library outside reviewed roots')
            path = str(canonical)
            if path not in libraries:
                size = canonical.stat().st_size
                require(0 < size <= FILE_BYTES and len(libraries) < MAX_LIBRARIES
                        and total + size <= TOTAL_BYTES, 'library inventory exceeds preparation bounds')
                # Exact size recheck occurs before hashing, not after an unbounded read.
                identity = file_identity(canonical, size=size)
                with canonical.open('rb') as source:
                    require(source.read(4) == b'\x7fELF', 'non-ELF library listing')
                libraries[path] = identity
                total += size
            resolved[name] = {'listed_path': spelling, 'canonical_path': path}
        listings[key] = {'dependencies': resolved, 'virtual_dependencies': virtual}
        receipts[key] = receipt
    require(listings['previous']['dependencies'] == listings['candidate']['dependencies'],
            'previous/candidate standalone resolution differs')
    for listing in listings.values():
        for dependency in listing['dependencies'].values():
            require(str(Path(dependency['listed_path']).resolve(strict=True)) == dependency['canonical_path'],
                    'library link changed during preparation')
    for identity in [loader_identity, *(t['image'] for t in targets.values()), *libraries.values()]:
        verify_snapshot(identity)
    result = {'schema_version': 1, 'diagnostic_only': True,
              'standalone_loader_resolution_verified': True,
              'python_runtime_load_order_verified': False, 'runtime_loaded_libraries_verified': False,
              'historical_library_bytes_verified': False, 'launch_ready': False,
              'loader_identity': loader_identity, 'targets': targets, 'environment': env,
              'library_roots': [str(p) for p in roots], 'listings': listings,
              'current_only_libraries': libraries, 'library_bytes': total, 'receipts': receipts,
              'scope': 'Preparation-time standalone glibc --list and current file identities only. '
                       'Not Python import/ctypes preload order, lazy driver/JIT closure, historical '
                       'byte equivalence, mapped-byte authentication or permission to launch. '
                       'Do not promote all standalone dependencies into required runtime mappings.'}
    fd = os.open(directory / 'libraries.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write('\n')
    return result
