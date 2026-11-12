"""Pure compilation of prepared declarations, not authentication or fresh preflight.

No file reads/stats/resolution, hashing, imports of Torch or subprocesses. The full
entry must bind the producer records/tools and recheck files before collection;
clock postflight must independently validate all observed known candidates.
"""
from copy import deepcopy
from pathlib import Path
import re

from clock_process_observation import identity
from readonly_trace_preflight import require

CORE_TORCH = ('libtorch_python.so', 'libtorch_cpu.so', 'libtorch_cuda.so', 'libc10.so', 'libc10_cuda.so')
TARGETS = {'previous', 'candidate', 'python', 'taskset', 'torch_extension', 'torch_global_deps'}
NATIVE = {'libc.so.6', 'libcuda.so.1', 'libcurand.so.10', 'libcublas.so.13'}
FIELDS = ('path', 'device', 'inode', 'bytes', 'sha256')


def signature(value):
    identity(value)
    path = Path(value['path'])
    require('..' not in path.parts and str(path) == value['path'], 'canonical declared identity path required')
    require(value['bytes'] <= 2 * 1024**3, 'prepared identity exceeds per-file bound')
    return tuple(value[k] for k in FIELDS)


def compile_contract(python, loader, preloads):
    """Join current-only records and select required mappings without runtime guesses.

    Native: every standalone dependency. Python: interpreter dependencies, _C and
    five core Torch code libraries. All preload candidates and both Torch target
    images remain known even if optional; no absent candidate is made mandatory.
    """
    python, loader, preloads = deepcopy((python, loader, preloads))
    for record in (python, loader, preloads):
        require(type(record.get('schema_version')) is int and record['schema_version'] == 1
                and record.get('launch_ready') is False
                and record.get('historical_library_bytes_verified') is False,
                'unsupported preparation schema/scope')
    require(loader.get('standalone_loader_resolution_verified') is True
            and loader.get('python_runtime_load_order_verified') is False
            and loader.get('runtime_loaded_libraries_verified') is False
            and preloads.get('site_candidate_inventory_verified') is True
            and preloads.get('runtime_selection_verified') is False
            and preloads.get('complete_sys_path_verified') is False
            and preloads.get('dependency_closure_verified') is False,
            'preparation scope flags differ')
    require(all(python.get(flag) is False for flag in (
        'historical_interpreter_bytes_verified', 'runtime_version_probed', 'torch_imported',
        'startup_import_closure_verified', 'library_resolution_verified', 'current_upstream_worktree_verified')),
        'unsupported Python preparation claim')
    site = Path(python['site_packages'])
    require(site.is_absolute() and '..' not in site.parts and str(site) == str(python['site_packages'])
            and preloads['site_packages'] == str(site), 'Python/preload site mismatch')
    version = python['venv_version_metadata']
    require(isinstance(version, str) and re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', version)
            and python['historical']['python_version'] == 'Python ' + version,
            'Python version declarations differ')
    require(python['torch_source_build_tag'] == preloads['torch_source_build_tag']
            == python['historical']['report']['torch'], 'Torch build declarations differ')
    targets, listings = loader['targets'], loader['listings']
    require(set(targets) == set(listings) == TARGETS, 'incomplete loader targets/listings')
    for target in targets.values(): signature(target['image'])
    require(signature(targets['python']['image']) == signature(python['interpreter_image']),
            'Python interpreter identities differ')
    command = python['command_path']
    require(isinstance(command, str) and '\0' not in command and Path(command).is_absolute()
            and '..' not in Path(command).parts, 'explicit Python argv spelling required')
    require(Path(targets['torch_extension']['image']['path']).parent == site/'torch'
            and Path(targets['torch_extension']['image']['path']).name.startswith('_C.')
            and targets['torch_extension']['image']['path'].endswith('.so')
            and targets['torch_global_deps']['image']['path'] == str(site/'torch/lib/libtorch_global_deps.so'),
            'Torch target paths differ from prepared site')
    py_files = {}
    for value in python['current_only_file_identities']:
        signature(value)
        require(value['path'] not in py_files, 'duplicate Python file identity')
        py_files[value['path']] = value
    image = python['interpreter_image']
    require(image['path'] in py_files and signature(py_files[image['path']]) == signature(image),
            'Python image absent or different in its file inventory')
    sources = preloads['current_only_sources']
    require(set(sources) == {'__init__.py', '_utils_internal.py', 'version.py'}, 'incomplete preload sources')
    for name, value in sources.items():
        signature(value)
        require(value['path'] == str(site/'torch'/name), 'preload source path mismatch')
    require(signature(sources['version.py']) == signature(py_files[str(site/'torch/version.py')]),
            'Python/preload version-source identities differ')
    known, aliases = {}, {}
    def add(value):
        current = signature(value)
        require(value['path'] not in known or signature(known[value['path']]) == current,
                'conflicting prepared file identities')
        known[value['path']] = value
    def alias(path, canonical):
        require(isinstance(path, str) and '\0' not in path and Path(path).is_absolute()
                and (path not in aliases or aliases[path] == canonical), 'conflicting declared path bindings')
        aliases[path] = canonical
    static, candidates = loader['current_only_libraries'], preloads['current_only_files']
    for inventory in (static, candidates):
        require(isinstance(inventory, dict) and len(inventory) <= 128, 'oversized library inventory')
        for path, value in inventory.items():
            require(path == value['path'], 'inventory key/path mismatch')
            add(value)
    require(type(loader['library_bytes']) is int and type(preloads['file_bytes']) is int
            and loader['library_bytes'] == sum(v['bytes'] for v in static.values())
            and preloads['file_bytes'] == sum(v['bytes'] for v in candidates.values()), 'inventory byte accounting differs')
    referenced = set()
    for listing in listings.values():
        dependencies = listing['dependencies']
        require(isinstance(dependencies, dict) and 'libc.so.6' in dependencies, 'incomplete standalone dependencies')
        for dep in dependencies.values():
            require(dep['canonical_path'] in static, 'standalone dependency has no prepared identity')
            referenced.add(dep['canonical_path'])
            alias(dep['listed_path'], dep['canonical_path'])
    require(referenced == set(static), 'unbound standalone identities')
    entries = preloads['entries']
    require(isinstance(entries, list) and 0 < len(entries) <= 33, 'invalid preload entry inventory')
    referenced = set()
    for ordinal, entry in enumerate(entries):
        require(type(entry['ordinal']) is int and entry['ordinal'] == ordinal
                and entry['runtime_selected_path'] is None and isinstance(entry['matches'], list),
                'preload order or unsupported runtime selection')
        for match in entry['matches']:
            require(match['canonical_path'] in candidates
                    and Path(match['canonical_path']).is_relative_to(site)
                    and Path(match['path']).is_relative_to(site) and '..' not in Path(match['path']).parts,
                    'unbound or escaping preload match')
            referenced.add(match['canonical_path'])
            alias(match['path'], match['canonical_path'])
    require(referenced == set(candidates), 'unbound preload identities')
    require(listings['previous']['dependencies'] == listings['candidate']['dependencies']
            and NATIVE <= listings['previous']['dependencies'].keys(), 'native standalone contract differs')
    minor = '.'.join(version.split('.')[:2])
    require({'libc.so.6', f'libpython{minor}.so.1.0'} <= listings['python']['dependencies'].keys()
            and set(CORE_TORCH) <= listings['torch_extension']['dependencies'].keys(),
            'missing mandatory Python/core Torch declarations')
    for name in ('torch_extension', 'torch_global_deps'): add(targets[name]['image'])
    require(0 < len(known) <= 256 and sum(v['bytes'] for v in known.values()) <= 8 * 1024**3,
            'combined known-library inventory exceeds bound')
    required = {}
    for runtime in ('previous', 'python', 'candidate'):
        dependencies = listings[runtime]['dependencies'].values()
        paths = {dep['canonical_path'] for dep in dependencies}
        if runtime == 'python':
            paths |= {listings['torch_extension']['dependencies'][name]['canonical_path'] for name in CORE_TORCH}
            paths.add(targets['torch_extension']['image']['path'])
        required[runtime] = deepcopy([known[path] for path in sorted(paths)])
    workloads = {runtime: {'command_path': command if runtime == 'python' else targets[runtime]['image']['path'],
                           'image': targets[runtime]['image']} for runtime in required}
    return deepcopy({'schema_version': 1, 'workloads': workloads, 'taskset': targets['taskset']['image'],
                     'libraries': required, 'known_libraries': [known[p] for p in sorted(known)],
                     'path_bindings': [{'path': p, 'canonical_path': aliases[p]} for p in sorted(aliases)],
                     'prepared_declaration_consistency_verified': True, 'runtime_mapping_verified': False,
                     'launch_ready': False,
                     'scope': 'Pure consistency/selection over supplied preparation records, not fresh file '
                              'checks or producer authentication. Optional known candidates need not map. '
                              'Frozen role binding and independent clock postflight remain required.'})
