"""Compose read-only diagnostic validators after an owned supervisor runner returns.

Import-only integration, NOT a launch CLI. The caller still owes reviewed frozen
supervisor/worker commands, a dedicated exclusive-reaper runner, fresh availability
and its actual surviving supervisor. The callback is trusted code, not an untrusted
success flag or a sandbox. This module orders validation and frozen-file rehashing;
receipt consistency cannot independently prove that the callback really reaped.
"""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat

from analyze_readonly_trace import analyze, read_bounded, REPORT_LIMIT
from prepare_readonly_launch import validate_contract
from readonly_trace_preflight import require
from readonly_trace_sequence import ORDER
from validate_readonly_lifecycle import validate as validate_lifecycle, receipt_complete
from validate_readonly_telemetry import (validate as validate_telemetry, unique_object, reject_constant,
                                        FILE_LIMIT)
from verify_readonly_processes import validate_processes, verify_files


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def parsed(raw):
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)


class Evidence:
    """Read only caller-owned fixed paths; hash the bytes actually supplied to validators."""
    def __init__(self):
        self.inputs = {}

    def raw(self, root, name, limit=REPORT_LIMIT):
        path = root / name
        raw = read_bounded(path, limit)
        require(str(path) not in self.inputs, 'duplicate evidence read')
        self.inputs[str(path)] = {'sha256': sha(raw), 'bytes': len(raw)}
        return raw

    def record(self, root, name):
        return parsed(self.raw(root, name))


def equal(first, second, reason):
    # No bool/int or nonfinite substitutions hidden by Python dictionary equality.
    require(json.dumps(first, sort_keys=True, allow_nan=False)
            == json.dumps(second, sort_keys=True, allow_nan=False), reason)


def publish(directory, name, value):
    raw = json.dumps(value, indent=2, allow_nan=False).encode() + b'\n'
    with (directory / name).open('xb') as file:
        file.write(raw)
    return {'sha256': sha(raw), 'bytes': len(raw)}


def run_then_finalize(contract_raw, expected_sha256, launch_dir, run_supervisor):
    """Invoke the trusted owned-child runner ONCE, then validate before frozen rehash.

    Runner must return its outer run_bounded receipt after exclusive reaping, and
    retain that same receipt as launch_dir/supervisor.receipt.json. It owns creation
    of collection_root and launch_dir/supervision. No callback retry or alternative
    qualifier exists. An exception/incomplete run preserves partial evidence and
    never reaches verify_files. The final launch entry is still required to supply
    the frozen runner; calling this helper is not authorization to collect.
    """
    require(isinstance(contract_raw, bytes) and 0 < len(contract_raw) <= REPORT_LIMIT
            and sha(contract_raw) == expected_sha256, 'preparation manifest digest/size mismatch')
    contract = validate_contract(parsed(contract_raw))
    require(callable(run_supervisor), 'explicit owned-supervisor runner required')
    collection = Path(contract['collection_root'])
    launch = Path(launch_dir).absolute()
    require(launch.is_dir() and not launch.is_symlink()
            and stat.S_IMODE(launch.stat().st_mode) == 0o700, 'private non-symlink launch directory required')
    launch = launch.resolve(strict=True)
    require(launch.is_dir() and not collection.exists() and not collection.is_symlink()
            and collection != launch and not launch.is_relative_to(collection)
            and not collection.is_relative_to(launch), 'fresh collection and separate launch directory required')
    output = launch / 'completion'
    require(not os.path.lexists(output) and not os.path.lexists(launch / 'supervisor.receipt.json')
            and not os.path.lexists(launch / 'supervision')
            and not os.path.lexists(launch / 'preparation.json'), 'launch evidence already exists')
    with (launch / 'preparation.json').open('xb') as file:
        file.write(contract_raw)
    # Sole submission point. No validation callback or frozen-file hashing can be
    # invoked while the runner is active. Its implementation remains the entry's duty.
    returned = run_supervisor()
    receipt_complete(returned)
    require(returned['lifetime']['pid'] == os.getpid(), 'outer receipt is not this collector')
    for directory in (collection, collection / 'sequence', collection / 'telemetry', launch / 'supervision'):
        require(directory.is_dir() and not directory.is_symlink(), 'missing or symlinked evidence directory')
    evidence = Evidence()
    require(evidence.raw(launch, 'preparation.json') == contract_raw, 'preparation copy changed during runner')
    outer = evidence.record(launch, 'supervisor.receipt.json')
    equal(outer, returned, 'returned/on-disk outer receipt mismatch')
    supervisor = evidence.record(launch, 'supervision/supervisor.json')
    equal(supervisor['worker'], evidence.record(launch, 'supervision/worker.receipt.json'),
          'worker receipt copy mismatch')
    coordinator = evidence.record(collection, 'collection.json')
    sequence = evidence.record(collection, 'sequence/sequence-result.json')
    plan = evidence.record(collection, 'sequence/sequence-plan.json')
    telemetry = evidence.record(collection, 'telemetry/telemetry.json')
    ready = evidence.record(collection, 'telemetry/ready.json')
    equal(telemetry['cpu_receipt'], evidence.record(collection, 'telemetry/cpu-command.receipt.json'),
          'CPU receipt copy mismatch')
    prepared, libraries = contract['prepared'], contract['libraries']
    require(plan['process_observation_enabled'] is True and len(plan['jobs']) == len(ORDER)
            and len(sequence['jobs']) == len(ORDER), 'complete observed plan required')
    observations = []
    for ordinal, (condition, declaration, planned, recorded) in enumerate(zip(
            ORDER, prepared['jobs'], plan['jobs'], sequence['jobs']), 1):
        equal(planned, dict(ordinal=ordinal, condition=condition, argv=declaration['argv'],
                            cwd=declaration['cwd'], report_path=declaration['report_path']),
              'sequence plan differs from frozen preparation')
        name = f'sequence/{ordinal:02d}-{condition}'
        equal(recorded['receipt'], evidence.record(collection, name + '.receipt.json'),
              'job receipt copy mismatch')
        observations.append(evidence.record(collection, name + '.process.json'))
    cpu_raw = evidence.raw(collection, 'telemetry/cpu.jsonl', FILE_LIMIT)
    gpu_raw = evidence.raw(collection, 'telemetry/gpu.csv', FILE_LIMIT)
    cpu_receipt = telemetry['cpu_receipt']
    require(cpu_receipt['argv'] == contract['telemetry_commands']['cpu'], 'CPU command not launch-bound')
    cpu_binding = cpu_receipt['report_capture']
    require(cpu_binding['post_exit_hash_verified'] is True and cpu_binding['sha256'] == sha(cpu_raw)
            and cpu_binding['size_bytes'] == len(cpu_raw)
            and cpu_binding['path'] == str(collection / 'telemetry/cpu.jsonl'), 'CPU raw receipt binding mismatch')
    require(telemetry['gpu_sha256'] == sha(gpu_raw), 'GPU raw receipt binding mismatch')
    source_path = str(Path(contract['project_root']) / 'tools/bench/sample_linux_cpu.py')
    source = {item['path']: item['sha256'] for item in contract['tool_identities']}[source_path]
    contents = validate_telemetry(cpu_raw, gpu_raw, sequence['jobs'], expected_cpu_source=source,
                                 gpu_uuid=contract['gpu']['uuid'], timezone_name=contract['timezone']['zone'])
    require(contents['telemetry_contents_validated'] is True and contents['receipt_windows_bracketed'] is True,
            'full telemetry validation did not complete')
    equal(ready['cpu_metadata'], contents['cpu_metadata'], 'readiness CPU metadata differs from full telemetry')
    rows = list(csv.reader(io.StringIO(gpu_raw.decode()), skipinitialspace=True, strict=True))
    equal(ready['first_gpu_row'], rows[1], 'readiness GPU row differs from full telemetry')
    cpu_rows = [parsed(line) for line in cpu_raw.splitlines()]
    lifecycle = validate_lifecycle(outer, supervisor, coordinator, sequence, telemetry, ready, observations, cpu_rows[1:-1])
    processes = validate_processes(sequence['jobs'], observations, prepared, libraries)
    require(lifecycle['lifecycle_consistency_verified'] is True
            and processes['process_receipt_consistency_verified'] is True, 'lifecycle/process validation incomplete')
    analysis = analyze(collection / 'sequence', prepared['sources'])
    require(analysis['receipt_consistency_verified'] is True and analysis['reported_codec_outputs_qualified'] is True
            and analysis['performance_acceptance_evaluated'] is False, 'trace/output diagnostic validation incomplete')
    require(analysis['plan_sha256'] == evidence.inputs[str(collection / 'sequence/sequence-plan.json')]['sha256']
            and analysis['sequence_result_sha256'] == evidence.inputs[str(collection / 'sequence/sequence-result.json')]['sha256'],
            'analyzer consumed changed sequence metadata')
    require(len(analysis['jobs']) == len(ORDER), 'analyzer job inventory mismatch')
    for ordinal, (condition, bound, job) in enumerate(zip(ORDER, analysis['jobs'], sequence['jobs']), 1):
        prefix = collection / 'sequence' / f'{ordinal:02d}-{condition}'
        receipt = job['receipt']
        equal({key: bound[key] for key in ('ordinal', 'condition', 'pid', 'receipt_sha256', 'capture_sha256', 'report_sha256')},
              dict(ordinal=ordinal, condition=condition, pid=receipt['pid'],
                   receipt_sha256=evidence.inputs[str(prefix) + '.receipt.json']['sha256'],
                   capture_sha256=receipt['capture_sha256'], report_sha256=receipt['report_capture']['sha256']),
              'analyzer receipt/raw bindings disagree')
        # Analyzer consumed these bounded bytes itself; do not read large traces twice.
        evidence.inputs[str(prefix) + '.capture'] = dict(sha256=bound['capture_sha256'], bytes=receipt['captured_bytes'])
        evidence.inputs[str(prefix) + '.report.json'] = dict(sha256=bound['report_sha256'], bytes=receipt['report_capture']['size_bytes'])
    for root, name, receipt in ((launch, 'supervisor.txt', outer),
                                 (launch, 'supervision/worker.txt', supervisor['worker']),
                                 (collection, 'telemetry/cpu-command.txt', cpu_receipt)):
        raw = evidence.raw(root, name, 1024**2)
        require(sha(raw) == receipt['capture_sha256'] and len(raw) == receipt['captured_bytes'] == receipt['observed_bytes'],
                'collector capture receipt mismatch')
    evidence.raw(collection, 'observer-worker.txt', 1024**2)
    # This is intentionally LAST: no frozen source/payload/library rehash before
    # runner return AND full telemetry, lifecycle, maps, traces and outputs pass.
    postflight = verify_files(processes, prepared, libraries, library_roots=tuple(map(Path, contract['library_roots'])))
    require(postflight['prelaunch_file_bytes_rechecked'] is True and postflight['current_mapped_paths_verified'] is True
            and postflight['mapped_file_bytes_verified'] is False and postflight['launch_provenance_verified'] is False,
            'postflight verification incomplete or overclaimed')
    output.mkdir(mode=0o700)
    results = {name: publish(output, name, value) for name, value in (
        ('telemetry-validation.json', contents), ('lifecycle-validation.json', lifecycle),
        ('process-validation.json', processes), ('trace-analysis.json', analysis), ('postflight.json', postflight))}
    result = {'schema_version': 1, 'status': 'complete', 'diagnostic_only': True,
              'preparation_sha256': expected_sha256, 'consumed_raw_files': evidence.inputs, 'results': results,
              'validation_composition_complete': True, 'frozen_rehash_after_runner_and_validation': True,
              'launch_provenance_verified': False, 'performance_acceptance_evaluated': False,
              'scope': 'Single trusted runner return followed by mandatory validators and frozen rehash. '
                       'The final entry still owes reviewed argv, actual exclusive reaping and fresh availability. '
                       'Raw/canonical consistency is not independent production or immutable mapped-byte proof.'}
    publish(output, 'completion.json', result)
    return result
