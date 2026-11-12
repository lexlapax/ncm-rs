"""File-only Python preparation for the clock diagnostic; no package imports/probes.

Use before telemetry, never in the locked worker. This is an import-only component,
not a launch manifest or permission gate. Historical source/version evidence and
current-only interpreter/package/startup snapshots are intentionally separate.
"""
import ast
from email.parser import BytesParser
import hashlib
import os
from pathlib import Path
import re
import tarfile

from analyze_readonly_trace import read_bounded
from finalize_readonly_trace import parsed
from readonly_trace_preflight import file_identity, require
from readonly_trace_runtime import MANIFEST_HEAD, PRODUCTION_HEAD

EVIDENCE = 'docs/results/2026-11-11-phase2-optimization/linux-diagnostic-v2-evidence.json'
SOURCES = {'benchmark': 'codec_bench.py', 'inputs_helper': 'codec_trace.py', 'stages': 'codec_stages.py'}
FIELDS = ('runtime', 'torch', 'upstream_commit', 'reference_versions', 'source_sha256',
          'manifest_sha256', 'workload_sha256', 'precision', 'index_dtype', 'warmups', 'trials')
TEXT_BYTES = 1024 ** 2
ARCHIVE_BYTES = 16 * 1024 ** 2
ARCHIVE_EXPANDED_BYTES = 64 * 1024 ** 2
MAX_DISTRIBUTIONS = 128
UPSTREAM = 'c70274273f489abc73832b2b98584a8483b6b203'


def contained(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and relative.parts and not {'.', '..'} & set(relative.parts),
            'relative contained preparation path required')
    path = root / relative
    require(path.resolve(strict=True) == path, 'symlinked preparation input requires separate review')
    return path


def snapshot(path, limit=TEXT_BYTES):
    raw = read_bounded(path, limit)
    value = file_identity(path, size=len(raw))
    require(value['bytes'] == len(raw) and value['sha256'] == hashlib.sha256(raw).hexdigest(),
            'preparation input changed between parsing and snapshot')
    return raw, value


def verify_snapshot(value):
    path = Path(value['path'])
    require(path.resolve(strict=True) == path, 'preparation path became symlinked')
    current = file_identity(path, value['sha256'], value['bytes'])
    require(all(current[key] == value[key] for key in ('path', 'device', 'inode', 'bytes', 'sha256')),
            'preparation file identity changed')


def pinned_snapshot(path, expected):
    _raw, value = snapshot(path)
    require(value['sha256'] == expected, 'reviewed file digest mismatch')
    value['expected_digest_verified'] = True
    return value


def historical_reference(root):
    """Bind seven selected records to the published v2 archive, without raw Git/env output."""
    root = Path(root).resolve(strict=True)
    raw, declaration = snapshot(contained(root, EVIDENCE))
    evidence = parsed(raw)
    require(type(evidence['schema_version']) is int and evidence['schema_version'] == 1
            and evidence['measured_commit'] == MANIFEST_HEAD
            and evidence['production_candidate'] == PRODUCTION_HEAD, 'unexpected reference checkpoint')
    spec = evidence['archive']
    require(type(spec['bytes']) is int and 0 < spec['bytes'] <= ARCHIVE_BYTES, 'archive exceeds preparation bound')
    archive_path = contained(root, spec['path'])
    archive_id = file_identity(archive_path, spec['sha256'], spec['bytes'])
    pins = {item['path']: item for item in spec['members']}
    require(0 < len(pins) == len(spec['members']) <= 512
            and all(type(p['bytes']) is int and 0 <= p['bytes'] <= 16 * TEXT_BYTES for p in pins.values())
            and sum(p['bytes'] for p in pins.values()) <= ARCHIVE_EXPANDED_BYTES,
            'archive inventory exceeds preparation bounds')
    prefix = 'phase2-linux-diagnostic-v2/'
    selected = {prefix + 'environment.json'} | {prefix + f'python/codec-python-{i}.json' for i in range(1, 7)}
    records, seen = {}, set()
    with tarfile.open(archive_path, 'r|xz') as archive:
        for member in archive:
            require(member.name in pins and member.name not in seen and member.isfile()
                    and member.size == pins[member.name]['bytes'], 'unexpected archive member')
            seen.add(member.name)
            if member.name in selected:
                raw = archive.extractfile(member).read(member.size + 1)
                require(len(raw) == member.size and hashlib.sha256(raw).hexdigest() == pins[member.name]['sha256'],
                        'historical record digest mismatch')
                records[member.name] = parsed(raw)
    require(seen == set(pins) and set(records) == selected, 'incomplete historical reference inventory')
    reports = [records[prefix + f'python/codec-python-{i}.json'] for i in range(1, 7)]
    report = {key: reports[0][key] for key in FIELDS}
    require(all(all(item[key] == report[key] for key in FIELDS) for item in reports),
            'historical Python metadata differs between processes')
    require(report['runtime'] == 'pytorch-pinned-eager-snapshot-fix' and report['upstream_commit'] == UPSTREAM
            and set(report['source_sha256']) == set(SOURCES) and report['precision'] == 'fp32'
            and report['index_dtype'] == 'i64' and type(report['warmups']) is int and report['warmups'] == 5
            and type(report['trials']) is int and report['trials'] == 30, 'unexpected reference protocol')
    version = records[prefix + 'environment.json']['venv']['python']
    require(isinstance(version, str) and re.fullmatch(r'Python \d+\.\d+\.\d+', version),
            'unsupported historical interpreter version')
    verify_snapshot(archive_id)
    verify_snapshot(declaration)
    return {'report': report, 'python_version': version, 'declaration': declaration,
            'archive': archive_id, 'selected_members': [pins[name] for name in sorted(selected)]}


def interpreter_chain(command):
    links, cursor = [], command
    while cursor.is_symlink():
        require(len(links) < 8, 'interpreter link chain exceeds bound')
        target = cursor.readlink()
        links.append({'path': str(cursor), 'target': str(target)})
        cursor = target if target.is_absolute() else cursor.parent / target
    return links, cursor.resolve(strict=True)


def prepare(root, *, reviewed_startup):
    """Snapshot current metadata, requiring explicit reviewed startup-file hashes.

    reviewed_startup maps site-packages-relative files to current reviewed digests;
    all top-level .pth files must be included. Helpers named by hooks must also be
    reviewed, but import closure cannot be inferred here. No hook is executed.
    """
    root = Path(root).resolve(strict=True)
    historical = historical_reference(root)
    report = historical['report']
    sources = [pinned_snapshot(contained(root, 'tools/reference/' + name), report['source_sha256'][key])
               for key, name in SOURCES.items()]
    require(isinstance(reviewed_startup, dict) and 0 < len(reviewed_startup) <= 32
            and all(isinstance(k, str) and isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v)
                    for k, v in reviewed_startup.items()), 'explicit reviewed startup hashes required')
    venv = contained(root, '.reference/venv')
    raw, config_id = snapshot(contained(venv, 'pyvenv.cfg'))
    config = {}
    for line in raw.decode().splitlines():
        key, separator, value = line.partition('=')
        key, value = key.strip(), value.strip()
        require(separator and key not in config, 'unsupported or duplicate venv config field')
        config[key] = value
    require(set(config) == {'home', 'include-system-site-packages', 'version', 'executable', 'command'}
            and config['include-system-site-packages'] == 'false'
            and 'Python ' + config['version'] == historical['python_version'], 'venv configuration mismatch')
    command = venv / 'bin/python'
    image_path = command.resolve(strict=True)
    require(Path(config['executable']).is_absolute() and Path(config['executable']).resolve(strict=True) == image_path
            and Path(config['home']).is_absolute() and Path(config['home']).resolve(strict=True) == image_path.parent,
            'venv executable/home declaration mismatch')
    image_raw, image = snapshot(image_path, 64 * TEXT_BYTES)
    require(image_raw.startswith(b'\x7fELF') and os.access(image_path, os.X_OK), 'executable Linux ELF interpreter required')
    # Preserve each link spelling; this is not a frozen runtime version probe.
    links, target = interpreter_chain(command)
    require(target == image_path, 'interpreter link chain changed')
    minor = '.'.join(config['version'].split('.')[:2])
    site = contained(venv, f'lib/python{minor}/site-packages')
    require(not any((site / name).exists() or (site / name).is_symlink()
                    for name in ('sitecustomize', 'sitecustomize.py', 'usercustomize', 'usercustomize.py')),
            'custom startup modules require separate review')
    require({p.name for p in site.glob('*.pth')} == {name for name in reviewed_startup if '/' not in name and name.endswith('.pth')},
            'startup hook inventory differs from review')
    startup = [pinned_snapshot(contained(site, name), expected) for name, expected in sorted(reviewed_startup.items())]
    directories = sorted(site.glob('*.dist-info'))
    require(0 < len(directories) <= MAX_DISTRIBUTIONS and not list(site.glob('*.egg-info')),
            'unsupported distribution inventory')
    normalize = lambda name: re.sub(r'[-_.]+', '-', name.lower())
    distributions, metadata_ids = {}, []
    for directory in directories:
        path = contained(site, str(directory.relative_to(site)) + '/METADATA')
        raw, value = snapshot(path)
        message = BytesParser().parsebytes(raw, headersonly=True)
        names, versions = message.get_all('Name', []), message.get_all('Version', [])
        require(len(names) == len(versions) == 1 and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', names[0])
                and len(versions[0]) <= 128 and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.+!_-]*', versions[0]),
                'unsupported distribution metadata')
        name = normalize(names[0])
        require(name not in distributions, 'duplicate normalized distribution name')
        distributions[name] = versions[0]
        metadata_ids.append(value)
    expected = report['reference_versions']
    require(isinstance(expected, dict) and expected and len({normalize(n) for n in expected}) == len(expected),
            'invalid historical distribution versions')
    require(all(distributions.get(normalize(n)) == v for n, v in expected.items()), 'reference distribution version mismatch')
    lock_raw, lock_id = snapshot(contained(root, 'tools/reference/requirements-linux.lock'))
    lock = {}
    for line in lock_raw.decode().splitlines():
        name, separator, version = line.partition('==')
        require(separator and name not in lock, 'unsupported requirements lock')
        lock[name] = version
    require(set(lock) == set(expected) and all(expected[n].split('+')[0] == v for n, v in lock.items()),
            'reference lock differs from historical versions')
    raw, version_id = snapshot(contained(site, 'torch/version.py'))
    literals = {}
    for node in ast.parse(raw).body:
        name = node.targets[0].id if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) else None
        if name == '__version__':
            require(name not in literals, 'duplicate Torch version declaration')
            literals[name] = ast.literal_eval(node.value)
    require(literals.get('__version__') == report['torch'], 'Torch source build tag differs from historical runtime')
    snapshots = [config_id, image, lock_id, version_id, *metadata_ids, *startup]
    for value in [*sources, *snapshots]:
        verify_snapshot(value)
    require(interpreter_chain(command) == (links, image_path), 'interpreter link chain changed during preparation')
    require(sorted(site.glob('*.dist-info')) == directories and not list(site.glob('*.egg-info'))
            and {p.name for p in site.glob('*.pth')} == {name for name in reviewed_startup if '/' not in name and name.endswith('.pth')}
            and not any((site / name).exists() or (site / name).is_symlink()
                        for name in ('sitecustomize', 'sitecustomize.py', 'usercustomize', 'usercustomize.py')),
            'startup/distribution inventory changed during preparation')
    return {'schema_version': 1, 'historical': historical, 'historical_source_identities': sources,
            'command_path': str(command), 'interpreter_image': image, 'interpreter_links': links,
            'venv_version_metadata': config['version'], 'site_packages': str(site),
            'distribution_versions': distributions,
            'extra_distributions': {n: v for n, v in distributions.items() if n not in {normalize(k) for k in expected}},
            'torch_source_build_tag': literals['__version__'], 'current_only_file_identities': snapshots,
            'historical_interpreter_bytes_verified': False, 'historical_library_bytes_verified': False,
            'runtime_version_probed': False, 'torch_imported': False, 'startup_import_closure_verified': False,
            'library_resolution_verified': False, 'current_upstream_worktree_verified': False, 'launch_ready': False,
            'scope': 'Preparation-time file checks only. Current-only snapshots are not historical byte equivalence. '
                     'Explicit startup review is not complete import/stdlib provenance. No runtime import, utility '
                     'probe, GPU command, clock authorization, collection or numerical qualification.'}
