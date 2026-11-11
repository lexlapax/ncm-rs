"""Post-collection process/receipt consistency and current mapped-path file checks.

No collection or GPU commands. Run only after supervisor and telemetry completion.
Current pathname hashes cannot prove immutable mapped bytes or whole-lifetime
coverage; a file first seen dynamically has no invented historical/prelaunch pin.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from observe_readonly_process import MAX_SAMPLES, SAMPLE_INTERVAL_NS, MAP_INVENTORY_BYTES
from readonly_trace_preflight import file_identity, require
from readonly_trace_sequence import ORDER
from readonly_trace_runtime import clean_environment

FILE_COUNT_LIMIT = 256
FILE_BYTES_LIMIT = 8 * 1024 ** 3
DEFAULT_LIBRARY_ROOTS = (Path('/usr/lib'), Path('/opt/cuda/targets/x86_64-linux/lib'))


def number(value, minimum=0):
    require(type(value) is int and value >= minimum, 'invalid integer in process evidence')
    return value


def key(value):
    return number(value['device']), number(value['inode'], 1)


def mapping_key(value):
    return number(value['path_device']), number(value['path_inode'], 1)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_processes(jobs, observations, prepared, libraries):
    """Bind all 12 supplied records to predeclared identities; no file reads here.

    Full report/capture/trace qualification remains the existing analyzer's job.
    Expected prelaunch structures must be separately bound to reviewed source and
    loader probes, not inferred from the measured process files.
    """
    require(len(jobs) == len(observations) == len(prepared['jobs']) == len(ORDER),
            'expected twelve process observations and jobs')
    require(libraries['static_loader_resolution_verified'] is True, 'missing loader preflight')
    require(type(prepared['workload_cpu']) is int and prepared['workload_cpu'] == 2
            and type(prepared['observer_cpu']) is int and prepared['observer_cpu'] == 0,
            'unexpected observer/workload CPUs')
    base_env = clean_environment(prepared)
    targets = libraries['targets']
    identities = {value['path']: value for value in prepared['file_identities']}
    require(len(identities) == len(prepared['file_identities']), 'duplicate prepared file paths')
    for target in targets.values():
        expected = identities[target['path']]
        require(key(expected) == key(target) and expected['sha256'] == target['sha256']
                and expected['bytes'] == target['bytes'], 'loader/prepared target identities differ')
    files, pids, bindings = {}, set(), []
    for ordinal, (condition, job, observed, expected) in enumerate(zip(ORDER, jobs, observations, prepared['jobs']), 1):
        require(number(job['ordinal'], 1) == ordinal and job['condition'] == expected['condition'] == condition,
                'process job order mismatch')
        env = dict(base_env)
        if condition.endswith('T'):
            env['CUDA_INJECTION64_PATH'] = targets['injection']['path']
        require(expected['env'] == env and len(expected['argv']) == 7
                and expected['argv'][1:3] == ['-c', '2']
                and expected['argv'][-1] == expected['report_path'], 'unexpected declared command/environment')
        receipt = job['receipt']
        require(receipt['status'] == 'complete' and receipt['issues'] == [] and receipt['returncode'] == 0
                and type(receipt['returncode']) is int and receipt['direct_child_reaped'] is True
                and receipt['capture_eof'] is True and receipt['interrupt_signal'] is None
                and receipt['process_observer_enabled'] is True, 'incomplete/unobserved process receipt')
        pid = number(receipt['pid'], 1)
        require(pid not in pids and observed['pid'] == pid and type(observed['pid']) is int,
                'duplicate/mismatched process PID')
        pids.add(pid)
        require(receipt['argv'] == expected['argv'] and receipt['cwd'] == expected['cwd'], 'process command binding mismatch')
        require(type(observed['schema_version']) is int and observed['schema_version'] == 1
                and observed['status'] == 'complete' and observed['issues'] == []
                and observed['observation_failures'] == [] and number(observed['process_start_ticks'], 1),
                'incomplete process observations')
        for flag in ('launch_provenance_verified', 'whole_lifetime_mapping_coverage',
                     'mapped_file_bytes_verified', 'environment_values_serialized'):
            require(observed[flag] is False, 'unsupported stronger observation claim')
        transient = number(observed['transient_samples_discarded'])
        unavailable = sum(number(observed[name]) for name in (
            'startup_incomplete_samples', 'proc_unavailable_samples', 'startup_permission_denied_samples'))
        require(unavailable <= transient, 'transient counters inconsistent')
        samples, maps = observed['samples'], observed['mapping_sets']
        require(isinstance(samples, list) and samples and len(samples) + transient <= MAX_SAMPLES
                and isinstance(maps, dict) and maps, 'missing/oversized observation inventory')
        encoded_bytes, used, work_keys, prior_start, prior_end = 0, set(), set(), None, None
        injected = False
        for map_hash, mapping in maps.items():
            require(map_hash == digest(mapping), 'mapping digest mismatch')
            encoded_bytes += len(json.dumps(mapping, sort_keys=True).encode())
            require(set(mapping) == {'files', 'non_file_executable_mappings'} and mapping['files'], 'invalid mapping fields')
            special = mapping['non_file_executable_mappings']
            require(isinstance(special, list) and all(isinstance(v, str) for v in special)
                    and special == sorted(set(special)), 'invalid non-file mapping labels')
            paths = []
            for value in mapping['files']:
                require(set(value) == {'path', 'device', 'inode', 'path_device', 'path_inode', 'kernel_map_link_verified'},
                        'unsupported executable mapping fields')
                path = value['path']
                require(isinstance(path, str) and Path(path).is_absolute() and '..' not in Path(path).parts
                        and not path.endswith(' (deleted)') and '\\' not in path,
                        'unsupported executable mapping path')
                key(value)  # retain kernel device separately; do not conflate it with pathname device
                mapping_key(value)
                require(value['inode'] == value['path_inode'] and value['kernel_map_link_verified'] is True,
                        'mapping/path inode or symlink check missing')
                require(path not in files or files[path] == value, 'mapped identity changed across processes')
                files[path] = value
                paths.append(path)
            require(paths == sorted(set(paths)), 'duplicate/unordered mapped files')
        require(encoded_bytes <= MAP_INVENTORY_BYTES, 'mapping inventory bound changed')
        runtime = 'accepted' if condition[0] == 'A' else 'candidate'
        image, injection = targets[runtime], targets['injection']
        require(expected['argv'][0] == targets['taskset']['path'] and expected['argv'][3] == image['path'],
                'declared executable differs from loader preflight')
        for sample in samples:
            start, end = number(sample['monotonic_start_ns']), number(sample['monotonic_end_ns'])
            require(start <= end and (prior_start is None or start - prior_start >= SAMPLE_INTERVAL_NS)
                    and (prior_end is None or prior_end <= start), 'unordered/overlapping process samples')
            prior_start, prior_end = start, end
            require(sample['phase'] in ('launcher', 'workload'), 'unknown process phase')
            affinity = sample['affinity']
            require(isinstance(affinity, list) and affinity
                    and all(type(cpu) is int and cpu >= 0 for cpu in affinity)
                    and affinity == sorted(set(affinity)), 'invalid sample affinity')
            mapping = maps[sample['mapping_sha256']]
            used.add(sample['mapping_sha256'])
            keys = {mapping_key(value) for value in mapping['files']}
            wanted = image if sample['phase'] == 'workload' else targets['taskset']
            require(key(wanted) in keys, 'declared executable not mapped')
            seen = key(injection) in keys
            require(type(sample['injection_observed']) is bool and sample['injection_observed'] == seen,
                    'injection observation does not match maps')
            require(condition.endswith('T') or not seen, 'injection in plain observations')
            if sample['phase'] == 'workload':
                require(sample['affinity'] == [prepared['workload_cpu']], 'workload observation affinity mismatch')
                work_keys.update(keys)
                injected |= seen
        require(used == maps.keys() and work_keys, 'unused mapping set or missing workload observations')
        require(not condition.endswith('T') or injected, 'traced injection not observed during workload')
        required = {key(libraries['libraries'][v['canonical_path']])
                    for v in libraries['listings'][runtime]['dependencies'].values()}
        require(required <= work_keys, 'required native libraries not observed')
        bindings.append({'ordinal': ordinal, 'condition': condition, 'pid': pid,
                         'process_start_ticks': observed['process_start_ticks'], 'samples': len(samples),
                         'transient_samples_discarded': transient, 'observation_canonical_sha256': digest(observed),
                         'receipt_canonical_sha256': digest(receipt)})
    return {'schema_version': 1, 'process_receipt_consistency_verified': True, 'jobs': bindings,
            'observed_files': files, 'launch_provenance_verified': False, 'mapped_file_bytes_verified': False,
            'scope': 'Supplied process/receipt/prelaunch metadata consistency only, not independent '
                     'producer proof, receipt time bracketing, mapped bytes or whole-lifetime observation.'}


def verify_files(binding, prepared, libraries, *, library_roots=DEFAULT_LIBRARY_ROOTS):
    """Recheck all predeclared files and hash mapped paths after the full collection.

    Caller must first verify surviving-supervisor/observer completion. Known files
    must retain prelaunch bytes and inode. Additional mapped libraries must be
    ELF-magic regular files under explicitly reviewed canonical library roots. They are labelled
    post-only, not silently promoted to frozen prelaunch libraries.
    """
    require(binding['process_receipt_consistency_verified'] is True, 'process binding required')
    pinned = {}
    for value in [*prepared['file_identities'], *prepared['declaration_identities'],
                  *libraries['libraries'].values(), libraries['loader_identity']]:
        path = value['path']
        require(isinstance(value['sha256'], str) and re.fullmatch(r'[0-9a-f]{64}', value['sha256'])
                and type(value['bytes']) is int and value['bytes'] >= 0, 'invalid prelaunch hash/size')
        key(value)
        require(path not in pinned or all(pinned[path][k] == value[k] for k in ('sha256', 'bytes', 'device', 'inode')),
                'conflicting prelaunch file declarations')
        pinned[path] = value
    observed = binding['observed_files']
    names = sorted(pinned.keys() | observed.keys())
    require(observed and len(names) <= FILE_COUNT_LIMIT, 'postflight file count exceeds bound')
    roots = tuple(Path(path).resolve(strict=True) for path in library_roots)
    require(roots and all(path.is_dir() and path != Path('/') for path in roots), 'explicit non-root library directories required')
    inventory = []
    total = 0
    # Validate the entire path/type/identity/budget inventory before hashing any file.
    for name in names:
        mapping, expected = observed.get(name), pinned.get(name)
        path = Path(name)
        require(path.is_absolute() and '..' not in path.parts, 'invalid predeclared path')
        if mapping is not None:
            require(mapping['path'] == name, 'mapped path key mismatch')
            require(str(path.resolve(strict=True)) == name,
                    'mapped path is not canonical or changed through symlinks')
            require(expected is not None or (any(path.is_relative_to(root) for root in roots)
                                            and re.fullmatch(r'.+\.so(?:\.[0-9A-Za-z_.-]+)*', path.name)),
                    'unreviewed post-only mapped library path')
        value = path.stat(follow_symlinks=False)
        require(stat.S_ISREG(value.st_mode), 'postflight input is not a regular file')
        actual_key = value.st_dev, value.st_ino
        require(mapping is None or actual_key == mapping_key(mapping), 'mapped pathname file changed after observation')
        total += value.st_size
        require(total <= FILE_BYTES_LIMIT, 'postflight hash byte budget exceeded')
        if expected is not None:
            require(key(expected) == actual_key and expected['bytes'] == value.st_size,
                    'file differs from prelaunch identity')
        inventory.append((path, mapping, value.st_size, expected))
    results = []
    for path, mapping, size, expected in inventory:
        wanted_key = mapping_key(mapping) if mapping is not None else key(expected)
        if mapping is not None:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            try:
                current = os.fstat(fd)
                require(stat.S_ISREG(current.st_mode) and (current.st_dev, current.st_ino) == wanted_key
                        and current.st_size == size and os.read(fd, 4) == b'\x7fELF',
                        'mapped file is not the expected ELF-magic regular file')
            finally:
                os.close(fd)
        identity = file_identity(path, expected['sha256'] if expected else None, size)
        require(key(identity) == wanted_key, 'pathname changed while hashing')
        results.append({'identity': identity, 'binding': 'prelaunch_and_post' if expected else 'post_only',
                        'observed_executable_mapping': mapping is not None,
                        'kernel_map_device': mapping['device'] if mapping else None,
                        'kernel_map_inode': mapping['inode'] if mapping else None})
    return {'schema_version': 1, 'current_mapped_paths_verified': True, 'prelaunch_file_bytes_rechecked': True,
            'files': results,
            'bytes_hashed': total, 'mapped_file_bytes_verified': False, 'launch_provenance_verified': False,
            'scope': 'Current mapped pathname ELF-magic regular file identities after collection and '
                     'all predeclared file hashes unchanged, additional libraries explicitly post-only. Not immutable mapped-byte '
                     'proof, library equivalence over unsampled periods or historical system-library parity.'}
