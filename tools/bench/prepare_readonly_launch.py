"""Prepare the read-only diagnostic's frozen metadata contract; NEVER collect.

Run the CLI with python -S. Only bounded loader/Git/nvidia-smi identity probes are
permitted here. Output remains launch_ready=false until the final launch/lifecycle
bindings and readiness review exist. A fresh quiet window is a separate requirement.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

from readonly_trace_preflight import file_identity, prepare, require
from readonly_trace_runtime import clean_environment, inspect_libraries, metadata_probe, verify_worktrees
from readonly_trace_sequence import (ORDER, SEQUENCE_SECONDS, CHILD_SECONDS, CAPTURE_BYTES,
                                     TOTAL_BYTES, MIN_FREE_BYTES)
from readonly_trace_telemetry import commands as telemetry_commands, SECONDS
from verify_readonly_processes import DEFAULT_LIBRARY_ROOTS, FILE_BYTES_LIMIT, FILE_COUNT_LIMIT

TOOL_DIRS = ('tools/bench', 'tools/reference')


def parse_device(text, *, expected_pci, expected_name):
    """Exactly one visible NVIDIA device: native frozen benchmarks select device 0."""
    require(re.fullmatch(r'[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]', expected_pci),
            'expected full PCI bus identity')
    rows = list(csv.reader(io.StringIO(text), skipinitialspace=True, strict=True))
    require(len(rows) == 1 and len(rows[0]) == 3, 'exactly one visible NVIDIA device required')
    uuid, pci, name = [value.strip() for value in rows[0]]
    require(re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', uuid), 'invalid GPU UUID')
    require(pci.upper() == expected_pci.upper() and name == expected_name, 'GPU PCI/name differs from declaration')
    return {'uuid': uuid, 'pci_bus_id': pci.upper(), 'name': name,
            'visible_device_count': 1, 'historical_uuid_equality_verified': False}


def timezone_identity(localtime=Path('/etc/localtime'), zone_root=Path('/usr/share/zoneinfo')):
    """Freeze a system IANA TZif symlink; never infer timezone from GPU CSV times.

    Copied non-symlink /etc/localtime is deliberately unsupported by this entry;
    an explicit reviewed mapping would be needed instead of guessing a zone.
    """
    localtime, zone_root = Path(localtime).absolute(), Path(zone_root).resolve(strict=True)
    require(localtime.is_symlink(), 'system timezone must have an explicit IANA symlink')
    link = os.readlink(localtime)
    canonical = localtime.resolve(strict=True)
    require(canonical.is_relative_to(zone_root), 'timezone outside reviewed system zoneinfo root')
    name = canonical.relative_to(zone_root).as_posix()
    identity = file_identity(canonical)
    with canonical.open('rb') as file:
        zone = ZoneInfo.from_file(file, key=name)
    now = datetime.now(timezone.utc)
    require(os.readlink(localtime) == link and localtime.resolve(strict=True) == canonical,
            'system timezone changed during preparation')
    return {'zone': name, 'localtime_path': str(localtime), 'link_target': link, 'file_identity': identity,
            'observed_utc': now.isoformat(), 'observed_local_with_offset': now.astimezone(zone).isoformat()}


def tool_inventory(root):
    """Freeze both local Python tool directories, including validation dependencies.

    These tracked source files may be more than the dynamic import closure; avoid
    pretending a hand-written short list proves all local validator dependencies.
    Global Python stdlib/native-library provenance remains a separate limitation.
    """
    root = Path(root)
    paths = sorted(path for relative in TOOL_DIRS for path in (root / relative).glob('*.py'))
    require(paths and all(path.is_file() and not path.is_symlink() for path in paths), 'invalid tool inventory')
    return [file_identity(path) for path in paths]


def merge_identities(*inventories):
    result = {}
    for inventory in inventories:
        for item in inventory:
            old = result.get(item['path'])
            require(old is None or all(old[key] == item[key] for key in ('sha256', 'bytes', 'device', 'inode')),
                    'conflicting prepared file identity')
            result[item['path']] = item
    return [result[name] for name in sorted(result)]


def validate_contract(contract):
    """Structural consistency of a prepared, NOT launch-ready contract."""
    require(type(contract['schema_version']) is int and contract['schema_version'] == 1
            and contract['kind'] == 'readonly_trace_preparation'
            and contract['launch_ready'] is False and contract['availability_confirmed'] is False
            and contract['collection_started'] is False,
            'preparation cannot assert readiness or availability')
    root, collection = Path(contract['project_root']), Path(contract['collection_root'])
    require(root.is_absolute() and collection.is_absolute(), 'absolute contract paths required')
    prepared = contract['prepared']
    env = clean_environment(prepared)
    require(type(prepared['workload_cpu']) is int and prepared['workload_cpu'] == 2
            and type(prepared['observer_cpu']) is int and prepared['observer_cpu'] == 0
            and len(prepared['jobs']) == len(ORDER), 'unexpected prepared sequence')
    require(contract['worktrees']['worktree_state_verified'] is True
            and contract['libraries']['static_loader_resolution_verified'] is True,
            'metadata guard did not complete')
    targets = contract['libraries']['targets']
    for ordinal, (condition, job) in enumerate(zip(ORDER, prepared['jobs']), 1):
        expected_env = dict(env)
        if condition.endswith('T'):
            expected_env['CUDA_INJECTION64_PATH'] = targets['injection']['path']
        runtime = 'accepted' if condition[0] == 'A' else 'candidate'
        report = collection / 'sequence' / f'{ordinal:02d}-{condition}.report.json'
        require(job['condition'] == condition and job['env'] == expected_env
                and len(job['argv']) == 7 and job['argv'][:4] == [targets['taskset']['path'], '-c', '2', targets[runtime]['path']]
                and job['report_path'] == str(report) and job['argv'][-1] == str(report), 'prepared job contract mismatch')
    commands = contract['telemetry_commands']
    cpu, gpu = telemetry_commands(root, collection / 'telemetry', contract['utilities']['python']['path'],
                                  targets['taskset']['path'], contract['utilities']['nvidia_smi']['path'], contract['gpu']['uuid'])
    # Standard-library-only observer: do not execute global site .pth code.
    cpu.insert(cpu.index(contract['utilities']['python']['path']) + 1, '-S')
    require(commands == {'cpu': cpu, 'gpu': gpu}, 'telemetry command contract mismatch')
    require(contract['limits'] == {'sequence_seconds': SEQUENCE_SECONDS, 'child_seconds': CHILD_SECONDS,
            'capture_bytes': CAPTURE_BYTES, 'total_capture_bytes': TOTAL_BYTES, 'initial_free_bytes': MIN_FREE_BYTES,
            'telemetry_seconds': SECONDS, 'postflight_files': FILE_COUNT_LIMIT, 'postflight_bytes': FILE_BYTES_LIMIT},
            'preparation limits changed')
    require(contract['worktrees']['states']['candidate']['head'] == contract['expected_candidate_head'],
            'candidate worktree/contract mismatch')
    ids = merge_identities(prepared['file_identities'])
    require(len(ids) == len(prepared['file_identities']), 'duplicate contract identities')
    by_path = {value['path']: value for value in ids}
    for value in [*contract['utilities'].values(), *contract['tool_identities'], contract['timezone']['file_identity']]:
        require(by_path[value['path']] == value, 'utility/tool/timezone not bound in prepared identities')
    require(contract['timezone']['zone'] and contract['library_roots'], 'timezone and library roots required')
    return contract


def prepare_contract(project_root, collection_root, evidence_dir, *, expected_candidate_head,
                     expected_gpu_pci, expected_gpu_name, home, inherited=None,
                     python=None, nvidia_smi=Path('/usr/bin/nvidia-smi'), taskset=Path('/usr/bin/taskset')):
    """Collect metadata only, then exclusively serialize a reviewable frozen contract.

    Does not reserve/create collection_root, run a benchmark, start telemetry, claim
    quietness or invoke postflight/acceptance. Partial metadata evidence is retained.
    """
    root = Path(project_root).resolve(strict=True)
    collection, evidence = Path(collection_root).absolute(), Path(evidence_dir).absolute()
    require(not os.path.lexists(collection) and not os.path.lexists(evidence), 'fresh output paths required')
    collection, evidence = collection.resolve(), evidence.resolve()
    require(re.fullmatch(r'[0-9a-f]{40}', expected_candidate_head), 'expected committed candidate HEAD required')
    require(collection != evidence and not evidence.is_relative_to(collection) and not collection.is_relative_to(evidence),
            'preparation and collection directories must be separate')
    require(not os.path.lexists(collection) and collection.parent.is_dir(), 'fresh collection path required')
    require(not os.path.lexists(evidence) and evidence.parent.is_dir(), 'fresh private preparation path required')
    prepared = prepare(root, collection / 'sequence', home, inherited=inherited, taskset=taskset,
                       reserved_parent=collection)
    env = clean_environment(prepared)
    utility_ids = {'python': file_identity(Path(python or sys.executable).resolve(strict=True)),
                   'nvidia_smi': file_identity(Path(nvidia_smi).resolve(strict=True))}
    require(all(os.access(item['path'], os.X_OK) for item in utility_ids.values()), 'utilities must be executable')
    tools = tool_inventory(root)
    tz = timezone_identity()
    roots = [str(path.resolve(strict=True)) for path in DEFAULT_LIBRARY_ROOTS]
    require(all(Path(path).is_dir() for path in roots), 'reviewed CUDA/native library roots missing')
    # Expand recheck inventory to include observer/runtime/validator sources and utilities.
    prepared['file_identities'] = merge_identities(prepared['file_identities'], list(utility_ids.values()), tools, [tz['file_identity']])
    evidence.mkdir(mode=0o700)
    worktrees = verify_worktrees(root, expected_candidate_head, evidence / 'worktrees')
    utility_ids['git'] = worktrees['git_identity']
    prepared['file_identities'] = merge_identities(prepared['file_identities'], [utility_ids['git']])
    libraries = inspect_libraries(prepared, evidence / 'libraries')
    gpu_dir = evidence / 'gpu'
    gpu_dir.mkdir(mode=0o700)
    text, gpu_receipt = metadata_probe([utility_ids['nvidia_smi']['path'],
        '--query-gpu=uuid,pci.bus_id,name', '--format=csv,noheader'], root, env, gpu_dir, 'identity')
    gpu = parse_device(text, expected_pci=expected_gpu_pci, expected_name=expected_gpu_name)
    cpu_command, gpu_command = telemetry_commands(root, collection / 'telemetry', utility_ids['python']['path'],
                          str(Path(taskset).absolute()), utility_ids['nvidia_smi']['path'], gpu['uuid'])
    cpu_command.insert(cpu_command.index(utility_ids['python']['path']) + 1, '-S')
    # Metadata probes may take time. Recheck all files, including separate evidence
    # declarations and loader libraries. No workload or telemetry is active here.
    frozen = merge_identities(prepared['file_identities'], prepared['declaration_identities'],
                              list(libraries['targets'].values()), [libraries['loader_identity']],
                              list(libraries['libraries'].values()))
    require(len(frozen) <= FILE_COUNT_LIMIT and sum(item['bytes'] for item in frozen) <= FILE_BYTES_LIMIT,
            'prepared files already exceed postflight budget')
    for item in frozen:
        current = file_identity(item['path'], item['sha256'], item['bytes'])
        require((current['device'], current['inode']) == (item['device'], item['inode']), 'prepared file replaced during probes')
    for listing in libraries['listings'].values():
        for dependency in listing['dependencies'].values():
            require(str(Path(dependency['listed_path']).resolve(strict=True)) == dependency['canonical_path'],
                    'library symlink changed during probes')
    require(os.readlink(tz['localtime_path']) == tz['link_target']
            and str(Path(tz['localtime_path']).resolve(strict=True)) == tz['file_identity']['path'],
            'timezone link changed during probes')
    contract = dict(schema_version=1, kind='readonly_trace_preparation', launch_ready=False, availability_confirmed=False,
                    collection_started=False,
                    prepared_at_utc=datetime.now(timezone.utc).isoformat(), project_root=str(root),
                    collection_root=str(collection), expected_candidate_head=expected_candidate_head,
                    prepared=prepared, worktrees=worktrees, libraries=libraries, gpu=gpu,
                    gpu_identity_receipt=gpu_receipt, timezone=tz, utilities=utility_ids, tool_identities=tools,
                    library_roots=roots, telemetry_commands={'cpu': cpu_command, 'gpu': gpu_command},
                    limits={'sequence_seconds': SEQUENCE_SECONDS, 'child_seconds': CHILD_SECONDS,
                            'capture_bytes': CAPTURE_BYTES, 'total_capture_bytes': TOTAL_BYTES,
                            'initial_free_bytes': MIN_FREE_BYTES, 'telemetry_seconds': SECONDS,
                            'postflight_files': FILE_COUNT_LIMIT, 'postflight_bytes': FILE_BYTES_LIMIT},
                    scope='Metadata preparation only. Native environments are sanitized as declared; Python observer '
                          'uses -S to omit site initialization. No quiet-window proof, runtime launch, output quality, '
                          'global Python stdlib/site provenance, historical GPU UUID equality or acceptance.')
    validate_contract(contract)
    raw = json.dumps(contract, indent=2, allow_nan=False).encode() + b'\n'
    with (evidence / 'manifest.json').open('xb') as output:
        output.write(raw)
    return {'manifest': str(evidence / 'manifest.json'), 'sha256': hashlib.sha256(raw).hexdigest(),
            'launch_ready': False, 'collection_started': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project_root', type=Path)
    parser.add_argument('collection_root', type=Path)
    parser.add_argument('evidence_dir', type=Path)
    parser.add_argument('--candidate-head', required=True)
    parser.add_argument('--gpu-pci', required=True)
    parser.add_argument('--gpu-name', required=True)
    args = parser.parse_args()
    require(sys.flags.no_site, 'invoke preparation with python -S')
    require(args.project_root.resolve(strict=True) == Path(__file__).resolve().parents[2],
            'CLI must prepare its own reviewed repository checkout')
    result = prepare_contract(args.project_root, args.collection_root, args.evidence_dir,
                              expected_candidate_head=args.candidate_head, expected_gpu_pci=args.gpu_pci,
                              expected_gpu_name=args.gpu_name, home=os.environ['HOME'])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
