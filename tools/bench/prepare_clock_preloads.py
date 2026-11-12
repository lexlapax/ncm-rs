"""File-only Torch preload candidates in one reviewed site-packages directory.

Never imports Torch or chooses a runtime winner. Sorted candidate inventories are
not glob order, sys.path coverage, dlopen dependency closure or mandatory mappings.
Call only during preparation, before telemetry/workloads.
"""
import ast
import fnmatch
import os
from pathlib import Path
import re

from prepare_clock_python import contained, snapshot, verify_snapshot
from readonly_trace_preflight import file_identity, require

SOURCES = {'__init__.py', '_utils_internal.py', 'version.py'}
FILE_BYTES = 2 * 1024 ** 3
TOTAL_BYTES = 8 * 1024 ** 3
MAX_FILES = 128
MAX_DIRECTORY_ENTRIES = 4096


def literal(tree, name):
    values = []
    for node in tree.body:
        target = (node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1
                  else node.target if isinstance(node, ast.AnnAssign) else None)
        if isinstance(target, ast.Name) and target.id == name:
            values.append(ast.literal_eval(node.value))
    require(len(values) == 1, 'one literal declaration required: ' + name)
    return values[0]


def directory_inventory(paths):
    result, count = {}, 0
    for path in sorted(paths):
        require(path.resolve(strict=False) == path, 'symlinked preload search directory')
        if not path.exists():
            result[path] = None
            continue
        require(path.is_dir(), 'preload search path is not a directory')
        names = []
        with os.scandir(path) as entries:
            for entry in entries:
                count += 1
                require(count <= MAX_DIRECTORY_ENTRIES, 'preload directory inventory exceeds bound')
                names.append(entry.name)
        result[path] = sorted(names)
    return result


def inventory(site_packages, *, reviewed_sources):
    """Review hashes are explicit current-only declarations, not historical pins.

    Account for all matches in the three source-defined layouts under this site.
    Missing/ambiguous matches are retained, never silently installed or selected.
    Other sys.path roots and dependencies loaded by these objects are not covered.
    """
    site = Path(site_packages)
    require(site.is_absolute() and site.resolve(strict=True) == site and site.is_dir(),
            'canonical site-packages directory required')
    require(isinstance(reviewed_sources, dict) and set(reviewed_sources) == SOURCES
            and all(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) for v in reviewed_sources.values()),
            'three explicit reviewed source digests required')
    sources, trees = {}, {}
    for name, expected in sorted(reviewed_sources.items()):
        raw, identity = snapshot(contained(site, 'torch/' + name))
        require(identity['sha256'] == expected, 'reviewed Torch source changed')
        identity['expected_digest_verified'] = True
        sources[name], trees[name] = identity, ast.parse(raw)
    cuda = literal(trees['version.py'], 'cuda')
    build = literal(trees['version.py'], '__version__')
    require(isinstance(cuda, str) and re.fullmatch(r'[0-9]+\.[0-9]+', cuda)
            and isinstance(build, str) and 0 < len(build) <= 128, 'unsupported static Torch version')
    require(literal(trees['_utils_internal.py'], 'USE_GLOBAL_DEPS') is True
            and literal(trees['_utils_internal.py'], 'USE_RTLD_GLOBAL_WITH_LIBTORCH') is False,
            'Torch loader flags require separate review')
    functions = [n for n in trees['__init__.py'].body
                 if isinstance(n, ast.FunctionDef) and n.name == '_preload_cuda_deps']
    require(len(functions) == 1, 'one reviewed preload function required')
    pairs = literal(functions[0], 'cuda_libs')
    require(isinstance(pairs, list) and 0 < len(pairs) <= 32, 'unsupported preload declarations')
    # The reviewed implementation's final optional call is outside the for loop.
    calls = [n.value for n in functions[0].body if isinstance(n, ast.Expr)
             and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
             and n.value.func.id == '_preload_cuda_lib']
    require(len(calls) == 1 and len(calls[0].args) == 2 and len(calls[0].keywords) == 1
            and calls[0].keywords[0].arg == 'required'
            and ast.literal_eval(calls[0].keywords[0].value) is False,
            'optional preload call requires separate review')
    optional = tuple(ast.literal_eval(arg) for arg in calls[0].args)
    require(optional == ('nvtx', 'libnvToolsExt.so.*[0-9]'), 'unexpected optional preload')
    pairs = [*pairs, optional]
    require(all(isinstance(pair, tuple) and len(pair) == 2
                and isinstance(pair[0], str) and re.fullmatch('[a-z][a-z0-9_]*', pair[0])
                and isinstance(pair[1], str) and re.fullmatch(r'lib[A-Za-z0-9_-]+\.so\.\*\[0-9\]', pair[1])
                for pair in pairs) and len(set(pairs)) == len(pairs), 'unsupported preload paths/patterns')
    directories = {folder: [site/'nvidia'/folder/'lib', site/'nvidia'/('cu'+cuda.split('.')[0])/'lib',
                            site/folder/'lib'] for folder, _pattern in pairs}
    before = directory_inventory({p for paths in directories.values() for p in paths})
    files, entries, total = {}, [], 0
    for ordinal, (folder, pattern) in enumerate(pairs):
        matches = []
        for directory in directories[folder]:
            for name in before[directory] or []:
                if not fnmatch.fnmatchcase(name, pattern):
                    continue
                path = directory/name
                canonical = path.resolve(strict=True)
                require(canonical.is_relative_to(site), 'preload candidate escapes reviewed site')
                key = str(canonical)
                if key not in files:
                    size = canonical.stat().st_size
                    require(0 < size <= FILE_BYTES and len(files) < MAX_FILES and total + size <= TOTAL_BYTES,
                            'preload file inventory exceeds bound')
                    value = file_identity(canonical, size=size)
                    with canonical.open('rb') as source:
                        require(source.read(4) == b'\x7fELF', 'preload candidate is not ELF')
                    files[key] = value
                    total += size
                matches.append({'path': str(path), 'canonical_path': key})
        entries.append({'ordinal': ordinal, 'folder': folder, 'pattern': pattern,
                        'kind': 'optional_nvtx' if ordinal == len(pairs)-1 else 'cuda_list',
                        'matches': matches, 'runtime_selected_path': None})
    require(directory_inventory(set(before)) == before, 'preload directory inventory changed')
    for entry in entries:
        for match in entry['matches']:
            require(str(Path(match['path']).resolve(strict=True)) == match['canonical_path'],
                    'preload link changed during inventory')
    for value in [*sources.values(), *files.values()]:
        verify_snapshot(value)
    return {'schema_version': 1, 'site_packages': str(site), 'current_only_sources': sources,
            'torch_source_build_tag': build, 'cuda_source_version': cuda, 'entries': entries,
            'current_only_files': files, 'file_bytes': total,
            'directory_entry_count': sum(len(names or []) for names in before.values()),
            'site_candidate_inventory_verified': True, 'runtime_selection_verified': False,
            'complete_sys_path_verified': False, 'dependency_closure_verified': False,
            'historical_library_bytes_verified': False, 'launch_ready': False,
            'scope': 'File-only candidate inventory in one reviewed site-packages root. '
                     'Source list order retained; sorted matches are NOT runtime glob/selection order. '
                     'Missing matches and multiple matches do not select, install or authorize libraries. '
                     'No Torch import, candidate dlopen, GPU query, permission test or benchmark.'}
