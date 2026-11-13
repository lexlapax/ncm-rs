"""Read /proc for an owned waitable child and, on mismatch, the collector itself.

Samples cannot prove whole-lifetime absence of injection or capture every mapping.
Prelaunch hashes and post-collection file checks must be bound separately. Never
serialize environment values or unexpected command lines, including on failure.
No ptrace, signals or GPU calls.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

class ObservationMismatch(ValueError):
    """Only fixed local guard messages; never unexpected environment/argv contents."""


def require(condition, message):
    if not condition:
        raise ObservationMismatch(message)


PROC_LIMIT = 4 * 1024 ** 2
ENV_LIMIT = 64 * 1024
MAX_SAMPLES = 3000
SAMPLE_INTERVAL_NS = 50_000_000
MAP_INVENTORY_BYTES = 8 * 1024 ** 2


def read_proc(path, limit):
    # /proc leaf sizes are generally zero; bound actual reads rather than st_size.
    with Path(path).open('rb') as file:
        raw = file.read(limit + 1)
    require(len(raw) <= limit, 'process observation exceeded read limit')
    return raw


def process_stat(raw):
    text = raw.decode('utf-8')
    pid, rest = text.split(' (', 1)
    _, fields = rest.rsplit(') ', 1)
    fields = fields.split()
    require(len(fields) >= 20 and pid.isdecimal() and fields[1].isdecimal()
            and fields[19].isdecimal(), 'unsupported process stat')
    return int(pid), int(fields[1]), int(fields[19])


def environment(raw):
    require(not raw or raw.endswith(b'\0'), 'truncated child environment')
    result = {}
    for entry in raw.split(b'\0')[:-1]:
        key, value = entry.split(b'=', 1)
        require(key not in result, 'duplicate environment key')
        result[key] = value
    return result


def executable_maps(raw):
    """Record file-backed executable mappings without ASLR addresses or file bytes."""
    files = {}
    special = set()
    for line in raw.decode('utf-8').splitlines():
        parts = line.split(maxsplit=5)
        require(len(parts) >= 5 and re.fullmatch(r'[0-9a-f]+-[0-9a-f]+', parts[0])
                and re.fullmatch(r'[r-][w-][x-][ps]', parts[1])
                and re.fullmatch(r'[0-9a-f]+', parts[2])
                and re.fullmatch(r'[0-9a-f]+:[0-9a-f]+', parts[3])
                and parts[4].isdecimal(), 'unsupported process maps row')
        if parts[1][2] != 'x':
            continue
        path = parts[5] if len(parts) == 6 else ''
        if not path.startswith('/'):
            require(int(parts[4]) == 0, 'unnamed file-backed executable mapping')
            special.add(path or '[anonymous]')
            continue
        require(not path.endswith(' (deleted)') and '\\' not in path,
                'deleted/escaped executable mapping path unsupported')
        major, minor = (int(value, 16) for value in parts[3].split(':'))
        value = dict(path=path, device=os.makedev(major, minor), inode=int(parts[4]))
        require(value['inode'] > 0 and (path not in files or files[path] == value),
                'ambiguous executable mapping identity')
        files[path] = value
    return [files[path] for path in sorted(files)], sorted(special)


def identity_key(value):
    return value.get('path_device', value['device']), value.get('path_inode', value['inode'])


def mapped_paths(pid, raw):
    """Bind kernel map links to current path metadata without opening map_files.

    Some filesystems expose a different device in maps than stat(path). Preserve
    both instead of treating them as the same identity. This does NOT prove mapped
    bytes equal current path bytes; direct map_files access may need extra privilege.
    """
    files, special = executable_maps(raw)
    by_path = {value['path']: value for value in files}
    for line in raw.decode('utf-8').splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) != 6 or parts[1][2] != 'x' or not parts[5].startswith('/'):
            continue
        path = parts[5]
        require(os.readlink(f'/proc/{pid}/map_files/{parts[0]}') == path, 'kernel mapping link changed')
        current = os.stat(path)
        require(stat.S_ISREG(current.st_mode) and current.st_ino == by_path[path]['inode'],
                'mapped pathname inode changed or unsupported filesystem inode translation')
        value = by_path[path]
        require('path_device' not in value or (value['path_device'], value['path_inode']) == (current.st_dev, current.st_ino),
                'mapped path identity changed while sampling')
        value.update(path_device=current.st_dev, path_inode=current.st_ino, kernel_map_link_verified=True)
    return files, special


def command_mismatch_category(raw_argv, alternate_argv, collector_argv):
    """Classify a mismatch without recording any command bytes or accepting it."""
    if alternate_argv is not None and raw_argv == alternate_argv:
        return 'other_declared_phase'
    if collector_argv and raw_argv == collector_argv:
        return 'collector_command'
    return 'unclassified'


class ProcessObservation:
    """Trusted synchronous callback for run_bounded's unreaped direct child.

    Commands/env and inode identities come from reviewed prelaunch evidence. A
    launcher (e.g. taskset) may exec the workload; no other image is supported.
    No large file hashing is done while the workload or full telemetry runs.
    """
    def __init__(self, *, workload, workload_argv, expected_env, cpu, injection,
                 traced, required_libraries=(), launcher=None, launcher_argv=None, known_libraries=()):
        require(type(cpu) is int and cpu >= 0 and type(traced) is bool, 'invalid observation condition')
        required_libraries, known_libraries = tuple(required_libraries), tuple(known_libraries)
        require(len(known_libraries) <= 256, 'known library inventory exceeds bound')
        for item in [workload, injection, *required_libraries, *known_libraries, *([launcher] if launcher else [])]:
            require(isinstance(item, dict) and Path(item['path']).is_absolute()
                    and type(item['device']) is int and type(item['inode']) is int
                    and item['inode'] > 0, 'expected prelaunch file identity')
        require(bool(launcher) == bool(launcher_argv), 'launcher command/identity mismatch')
        require(isinstance(expected_env, dict) and all(isinstance(k, str) and isinstance(v, str)
                                                      for k, v in expected_env.items()), 'explicit environment required')
        for argv in [workload_argv] + ([launcher_argv] if launcher else []):
            require(isinstance(argv, (list, tuple)) and argv
                    and all(isinstance(a, str) and a and '\0' not in a for a in argv), 'invalid expected argv')
        self.workload, self.launcher, self.injection = dict(workload), dict(launcher) if launcher else None, dict(injection)
        self.argv = {'workload': b'\0'.join(a.encode() for a in workload_argv) + b'\0'}
        if launcher:
            self.argv['launcher'] = b'\0'.join(a.encode() for a in launcher_argv) + b'\0'
        self.env = {key.encode(): value.encode() for key, value in expected_env.items()}
        self.cpu, self.traced = cpu, traced
        self.required = {identity_key(value) for value in required_libraries}
        # Known candidates need not load. If a known path appears, even its first
        # complete sample must match preparation; it cannot become "post-only".
        self.known = {}
        for value in known_libraries:
            expected = (value['device'], value['inode'])
            require(value['path'] not in self.known or self.known[value['path']] == expected,
                    'conflicting known library identities')
            self.known[value['path']] = expected
        self.pid = self.start_ticks = None
        self.samples, self.mapping_sets, self.files = [], {}, {}
        self.transient_count = 0
        self.last_attempt_ns = None
        self.mapping_bytes = 0
        self.failure_reasons = []
        self.startup_incomplete_count = 0
        self.proc_unavailable_count = 0
        self.startup_permission_count = 0

    def observe(self, pid):
        try:
            self._observe(pid)
        except Exception as error:
            self.failure_reasons.append(str(error) if type(error) is ObservationMismatch else type(error).__name__)
            raise

    def _observe(self, pid):
        require(type(pid) is int and pid > 0 and (self.pid is None or self.pid == pid),
                'observer PID changed')
        # Verify waitable ownership without reaping. Never inspect unrelated PIDs.
        exited = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        if exited is not None:
            return
        self.pid = pid
        started = time.monotonic_ns()
        if self.last_attempt_ns is not None and started - self.last_attempt_ns < SAMPLE_INTERVAL_NS:
            return
        self.last_attempt_ns = started
        require(len(self.samples) + self.transient_count < MAX_SAMPLES, 'process observation sample limit')
        base = Path('/proc') / str(pid)
        raw_maps = None
        try:
            before = process_stat(read_proc(base / 'stat', ENV_LIMIT))
            image = os.stat(base / 'exe')
            require(stat.S_ISREG(image.st_mode), 'child executable is not regular')
            image_key = image.st_dev, image.st_ino
            raw_env = read_proc(base / 'environ', ENV_LIMIT)
            raw_argv = read_proc(base / 'cmdline', ENV_LIMIT)
            raw_maps = read_proc(base / 'maps', PROC_LIMIT)
            files, special = mapped_paths(pid, raw_maps)
            late_env = read_proc(base / 'environ', ENV_LIMIT)
            late_argv = read_proc(base / 'cmdline', ENV_LIMIT)
            affinity = sorted(os.sched_getaffinity(pid))
            image_after = os.stat(base / 'exe')
            after = process_stat(read_proc(base / 'stat', ENV_LIMIT))
        except PermissionError:
            # exec temporarily changes dumpability. This is not permission proof:
            # a later complete workload sample is mandatory, otherwise finish fails.
            if not any(row['phase'] == 'workload' for row in self.samples):
                self.transient_count += 1
                self.startup_permission_count += 1
                return
            raise
        except (FileNotFoundError, ProcessLookupError):
            # Linux may remove proc entries/VMAs before the child becomes waitable
            # on exit. Ownership was verified above; never promote this sample.
            self.transient_count += 1
            self.proc_unavailable_count += 1
            return
        require(before == after and before[:2] == (pid, os.getpid()), 'process identity/parent changed')
        if image_key != (image_after.st_dev, image_after.st_ino):
            self.transient_count += 1  # exec raced this sample; do not infer mixed observations
            return
        require(self.start_ticks is None or self.start_ticks == before[2], 'process start identity changed')
        self.start_ticks = before[2]
        phase = 'workload' if image_key == identity_key(self.workload) else 'launcher'
        require(phase == 'workload' or (self.launcher is not None and image_key == identity_key(self.launcher)),
                'unexpected live executable identity')
        if raw_env != late_env or raw_argv != late_argv:
            self.transient_count += 1
            return  # observed mixed exec/argument data, not a verified sample
        # Linux can expose the new exe before exec has installed argv/env/maps.
        # This is not readiness: require a later complete workload observation.
        if not files or not raw_argv or (self.env and not raw_env):
            self.transient_count += 1
            if not any(row['phase'] == 'workload' for row in self.samples):
                self.startup_incomplete_count += 1
            else:
                self.proc_unavailable_count += 1
            return
        require(environment(raw_env) == self.env, 'live child environment mismatch')
        if raw_argv != self.argv[phase]:
            # Classify but never promote or erase a mismatch. During exec Linux
            # can expose mixed fields; a sampled equality alone does not prove it.
            alternate = self.argv.get('launcher' if phase == 'workload' else 'workload')
            try:
                collector = read_proc('/proc/self/cmdline', ENV_LIMIT)
            except (OSError, ValueError):
                collector = None
            category = command_mismatch_category(raw_argv, alternate, collector)
            raise ObservationMismatch('live child command mismatch: ' + category)
        # During exec /proc can expose new cmdline/maps before exe changes. Only
        # stable supported snapshots are accepted; mismatches fail, never guessed.
        require(image_key in {identity_key(value) for value in files}, 'executable not mapped in sample')
        if phase == 'workload':
            require(affinity == [self.cpu], 'workload affinity mismatch')
        require(all(value['path'] != self.injection['path'] or identity_key(value) == identity_key(self.injection)
                    for value in files), 'mapped injection inode differs from prelaunch identity')
        injection_seen = any(identity_key(value) == identity_key(self.injection) for value in files)
        require(self.traced or not injection_seen, 'injection observed in plain process')
        for value in files:
            require(value['path'] not in self.known or identity_key(value) == self.known[value['path']],
                    'known library identity differs from preparation')
            previous = self.files.get(value['path'])
            require(previous is None or previous == value, 'mapped file identity changed between samples')
            self.files[value['path']] = value
        mapping = {'files': files, 'non_file_executable_mappings': special}
        encoded = json.dumps(mapping, sort_keys=True).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        if digest not in self.mapping_sets:
            require(self.mapping_bytes + len(encoded) <= MAP_INVENTORY_BYTES, 'mapping inventory exceeds bound')
            self.mapping_bytes += len(encoded)
            self.mapping_sets[digest] = mapping
        self.samples.append({'monotonic_start_ns': started, 'monotonic_end_ns': time.monotonic_ns(),
                             'phase': phase, 'affinity': affinity, 'mapping_sha256': digest,
                             'injection_observed': injection_seen})

    def finish(self, receipt, output):
        """Write bounded observations after capture; no post-collection hashes yet."""
        work = [row for row in self.samples if row['phase'] == 'workload']
        mapped = {identity_key(value) for row in work
                  for value in self.mapping_sets[row['mapping_sha256']]['files']}
        issues = []
        if self.failure_reasons:
            issues.append('process_observer_failed')
        if receipt['status'] != 'complete' or receipt['pid'] != self.pid or not receipt['direct_child_reaped']:
            issues.append('child_capture_or_identity_incomplete')
        if not work:
            issues.append('workload_not_observed')
        if self.traced and not any(row['injection_observed'] for row in work):
            issues.append('injection_not_observed_in_traced_workload')
        if not self.required <= mapped:
            issues.append('required_libraries_not_observed')
        result = {'schema_version': 1, 'status': 'complete' if not issues else 'failed', 'issues': issues,
                  'pid': self.pid, 'process_start_ticks': self.start_ticks, 'samples': self.samples,
                  'mapping_sets': self.mapping_sets, 'transient_samples_discarded': self.transient_count,
                  'observation_failures': self.failure_reasons,
                  'startup_incomplete_samples': self.startup_incomplete_count,
                  'proc_unavailable_samples': self.proc_unavailable_count,
                  'startup_permission_denied_samples': self.startup_permission_count,
                  'launch_provenance_verified': False, 'whole_lifetime_mapping_coverage': False,
                  'mapped_file_bytes_verified': False, 'environment_values_serialized': False,
                  'scope': 'Owned-child sampled executable inode, command/environment equality, affinity and '
                           'file-backed executable maps only. No whole-lifetime absence proof, immutable '
                           'mapped-byte snapshot, transient/JIT coverage or post-collection library hashes. '
                           'Kernel map devices and current pathname devices are preserved separately; '
                           'map_files symlinks are read, underlying map_files objects are not opened.'}
        raw = json.dumps(result, indent=2).encode() + b'\n'
        require(len(raw) <= 16 * 1024 ** 2, 'process observation result exceeds bound')
        with Path(output).open('xb') as file:
            file.write(raw)
        return result
