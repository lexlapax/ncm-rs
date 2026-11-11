"""Fresh-interpreter composition tests; archived reports and synthetic trace/process data."""
import copy
import csv
from datetime import datetime, timedelta
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import finalize_readonly_trace as finalizer
from readonly_trace_preflight import file_identity
from readonly_trace_sequence import ORDER
from readonly_trace_telemetry import commands
import test_verify_readonly_processes as process_fixtures
import test_validate_readonly_lifecycle as life_fixtures
from test_validate_readonly_telemetry import cpu_fixture, gpu_fixture, UUID
from test_analyze_readonly_trace import make_collection

HERE = Path(__file__).resolve().parent


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def build_fixture():
    fixture = process_fixtures.ProcessPostflightTests()
    fixture.setUp()
    root = fixture.root
    launch, collection = root / 'launch', root / 'collection'
    launch.mkdir(mode=0o700)
    sampler = root / 'tools/bench/sample_linux_cpu.py'
    sampler.parent.mkdir(parents=True)
    sampler.write_bytes((HERE / 'sample_linux_cpu.py').read_bytes())
    utilities = {}
    for name in ('python', 'nvidia_smi', 'git'):
        path = root / name
        path.write_bytes(b'never executed fixture utility')
        utilities[name] = file_identity(path)
    timezone_path = root / 'timezone-fixture'
    timezone_path.write_bytes(b'not used as a real timezone file')
    zone_id, tool_id = file_identity(timezone_path), file_identity(sampler)
    fixture.prepared['file_identities'] += [zone_id, tool_id, *utilities.values()]
    for ordinal, job in enumerate(fixture.prepared['jobs'], 1):
        job['report_path'] = str(collection / 'sequence' / f'{ordinal:02d}-{job["condition"]}.report.json')
        job['argv'][-1] = job['report_path']
    cpu_command, gpu_command = commands(root, collection / 'telemetry', utilities['python']['path'],
                                       fixture.ids['taskset']['path'], utilities['nvidia_smi']['path'], UUID)
    cpu_command.insert(cpu_command.index(utilities['python']['path']) + 1, '-S')
    contract = dict(schema_version=1, kind='readonly_trace_preparation', launch_ready=False,
                    availability_confirmed=False, collection_started=False, project_root=str(root),
                    collection_root=str(collection), expected_candidate_head='a'*40, prepared=fixture.prepared,
                    libraries=fixture.libraries, worktrees=dict(worktree_state_verified=True,
                    states={'candidate': {'head': 'a'*40}}), utilities=utilities, tool_identities=[tool_id],
                    timezone={'zone': 'UTC', 'file_identity': zone_id}, gpu={'uuid': UUID},
                    telemetry_commands={'cpu': cpu_command, 'gpu': gpu_command}, library_roots=[str(fixture.libroot)],
                    limits=dict(sequence_seconds=600, child_seconds=120, capture_bytes=1024**3,
                                total_capture_bytes=6*1024**3, initial_free_bytes=12*1024**3,
                                telemetry_seconds=900, postflight_files=256, postflight_bytes=8*1024**3))
    # Save source declarations before submission; the real prepare helper obtains
    # these from pinned evidence. No codec, native loader, Git or GPU command runs.
    # Archived make_collection provides the full A/C inventories in the runner.
    # Read them independently here from the same published v2 archive fixture.
    import tarfile
    from qualify_diagnostic_codec import EVIDENCE, ROOT
    declaration = json.loads(EVIDENCE.read_bytes())
    with tarfile.open(ROOT / declaration['archive']['path']) as archive:
        contract['prepared']['sources'] = {key: json.loads(archive.extractfile(
            f'phase2-linux-diagnostic-v2/{runtime}/codec-rust-1.json').read())['source_sha256']
            for key, runtime in (('A', 'previous'), ('C', 'candidate'))}
    raw = json.dumps(contract).encode()
    return fixture, contract, raw, launch, collection


def populate(fixture, contract, launch, collection, case):
    collection.mkdir(mode=0o700)
    sequence_root, telemetry_root, sup_root = collection / 'sequence', collection / 'telemetry', launch / 'supervision'
    for directory in (sequence_root, telemetry_root, sup_root):
        directory.mkdir()
    sources = make_collection(sequence_root, archived=True)
    assert sources == contract['prepared']['sources']
    plan = json.loads((sequence_root / 'sequence-plan.json').read_bytes())
    sequence = json.loads((sequence_root / 'sequence-result.json').read_bytes())
    outer, sup, coord, clock_seq, tel, ready, time_observations, _samples = life_fixtures.fixture()
    # Translate the clock fixture to the real serializer's virtual clock domain:
    # monotonic 1000 s corresponds to 12:00 UTC; no wall-clock sleeping/sampling.
    def translate(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.startswith('monotonic_') and key.endswith('_ns'):
                    value[key] += 994 * 10**9
                elif key in ('utc', 'started_utc', 'finished_utc'):
                    value[key] = (datetime.fromisoformat(item) + timedelta(hours=12, seconds=-6)).isoformat()
                else:
                    translate(item)
        elif isinstance(value, list):
            for item in value:
                translate(item)
    for value in (outer, sup, coord, clock_seq, tel, ready, time_observations):
        translate(value)
    outer['lifetime']['pid'] = os.getpid()
    sup['lifetime']['parent_pid'] = sup['worker']['lifetime']['parent_pid'] = os.getpid()
    sequence['lifetime'] = clock_seq['lifetime']
    plan['process_observation_enabled'] = True
    for i, (condition, declaration, clock_job) in enumerate(zip(ORDER, contract['prepared']['jobs'], clock_seq['jobs'])):
        receipt = sequence['jobs'][i]['receipt']
        pid = receipt['pid']
        receipt.update(clock_job['receipt'])
        receipt.update(pid=pid, argv=declaration['argv'], cwd=declaration['cwd'])
        plan['jobs'][i] = dict(ordinal=i+1, condition=condition, argv=declaration['argv'],
                              cwd=declaration['cwd'], report_path=declaration['report_path'])
        observation = copy.deepcopy(fixture.observations[i])
        observation['pid'] = pid
        observation['samples'][0].update(time_observations[i]['samples'][0])
        if case == 'process' and i == 2:
            observation['samples'][0]['injection_observed'] = False
        if case == 'lifecycle' and i == 0:
            observation['samples'][0]['monotonic_start_ns'] = 0
        prefix = sequence_root / f'{i+1:02d}-{condition}'
        if case == 'quality' and i == 0:
            report_path = Path(str(prefix) + '.report.json')
            report = json.loads(report_path.read_bytes())
            report['measurements'][1]['first_logits'][0] = 99.
            write(report_path, report)
            data = report_path.read_bytes()
            receipt['report_capture'].update(sha256=finalizer.sha(data), size_bytes=len(data))
        write(Path(str(prefix) + '.receipt.json'), receipt)
        write(Path(str(prefix) + '.process.json'), observation)
    write(sequence_root / 'sequence-plan.json', plan)
    write(sequence_root / 'sequence-result.json', sequence)
    cpu_raw, gpu_raw = cpu_fixture(), gpu_fixture()
    cpu_rows = [json.loads(line) for line in cpu_raw.splitlines()]
    ready['cpu_metadata'] = cpu_rows[0]
    ready['first_gpu_row'] = list(csv.reader(io.StringIO(gpu_raw.decode()), skipinitialspace=True))[1]
    if case == 'readiness':
        ready['cpu_metadata'] = {**ready['cpu_metadata'], 'source_sha256': 'f'*64}
    if case == 'telemetry':
        cpu_raw = b'\n'.join(cpu_raw.splitlines()[:-1]) + b'\n'
    coord['ready_canonical_sha256'] = finalizer.sha(json.dumps(ready, sort_keys=True).encode())
    (telemetry_root / 'cpu.jsonl').write_bytes(cpu_raw)
    (telemetry_root / 'gpu.csv').write_bytes(gpu_raw)
    tel['cpu_receipt'].update(argv=contract['telemetry_commands']['cpu'],
        report_capture=dict(path=str(telemetry_root / 'cpu.jsonl'), post_exit_hash_verified=True,
                            sha256=finalizer.sha(cpu_raw), size_bytes=len(cpu_raw)))
    tel['gpu_sha256'] = finalizer.sha(gpu_raw)
    for path, receipt in ((launch / 'supervisor.txt', outer), (sup_root / 'worker.txt', sup['worker']),
                          (telemetry_root / 'cpu-command.txt', tel['cpu_receipt'])):
        path.write_bytes(b'')
        receipt.update(capture_sha256=finalizer.sha(b''), captured_bytes=0, observed_bytes=0)
    (collection / 'observer-worker.txt').write_bytes(b'fake observer\n')
    for path, record in ((launch / 'supervisor.receipt.json', outer), (sup_root / 'supervisor.json', sup),
                         (sup_root / 'worker.receipt.json', sup['worker']), (collection / 'collection.json', coord),
                         (telemetry_root / 'telemetry.json', tel), (telemetry_root / 'ready.json', ready),
                         (telemetry_root / 'cpu-command.receipt.json', tel['cpu_receipt'])):
        write(path, record)
    if case == 'source':
        path = Path(fixture.ids['source']['path'])
        path.write_bytes(b'X' * path.stat().st_size)
    if case == 'capture':
        (launch / 'supervisor.txt').write_bytes(b'changed capture')
    if case == 'receipt_copy':
        value = copy.deepcopy(outer)
        value['pid'] += 1
        write(launch / 'supervisor.receipt.json', value)
    return outer


def exercise(case):
    fixture, contract, raw, launch, collection = build_fixture()
    state = {'running': False, 'calls': 0, 'hashes': 0}
    actual_verify, actual_analyze = finalizer.verify_files, finalizer.analyze
    def analyze(*args, **kwargs):
        if case == 'analyzer_metadata':
            path = collection / 'sequence/sequence-plan.json'
            path.write_bytes(path.read_bytes() + b' ')
        return actual_analyze(*args, **kwargs)
    def verify(*args, **kwargs):
        assert not state['running'] and state['calls'] == 1, 'rehash while runner active'
        state['hashes'] += 1
        return actual_verify(*args, **kwargs)
    def runner():
        state['calls'] += 1
        state['running'] = True
        if case == 'runner_error':
            raise RuntimeError('synthetic supervised runner failure')
        result = populate(fixture, contract, launch, collection, case)
        state['running'] = False
        return result
    try:
        with patch.object(finalizer, 'verify_files', side_effect=verify), \
                patch.object(finalizer, 'analyze', side_effect=analyze):
            if case == 'complete':
                result = finalizer.run_then_finalize(raw, finalizer.sha(raw), launch, runner)
                assert result['validation_composition_complete'] and state['hashes'] == 1
                assert not result['launch_provenance_verified'] and not result['performance_acceptance_evaluated']
                assert len(result['consumed_raw_files']) == 64, len(result['consumed_raw_files'])
                for name, identity in result['results'].items():
                    data = (launch / 'completion' / name).read_bytes()
                    assert finalizer.sha(data) == identity['sha256'] and len(data) == identity['bytes']
                try:
                    finalizer.run_then_finalize(raw, finalizer.sha(raw), launch, runner)
                except ValueError:
                    pass
                else:
                    raise AssertionError('allowed collection rerun')
            else:
                try:
                    finalizer.run_then_finalize(raw, '0'*64 if case == 'manifest' else finalizer.sha(raw), launch, runner)
                except (ValueError, RuntimeError) as error:
                    expected = {'manifest': 'preparation manifest digest/size mismatch',
                                'runner_error': 'synthetic supervised runner failure',
                                'telemetry': 'expected metadata, 901 samples and completion',
                                'readiness': 'readiness CPU metadata differs from full telemetry',
                                'receipt_copy': 'returned/on-disk outer receipt mismatch',
                                'process': 'injection observation does not match maps',
                                'quality': 'logit tolerance', 'source': 'file digest mismatch',
                                'lifecycle': 'process sample outside receipt/order',
                                'capture': 'collector capture receipt mismatch',
                                'analyzer_metadata': 'analyzer consumed changed sequence metadata'}[case]
                    assert expected in str(error), (case, str(error))
                    print(case, type(error).__name__, str(error))
                else:
                    raise AssertionError('accepted invalid case: ' + case)
                assert not (launch / 'completion').exists()
                assert state['hashes'] == (1 if case == 'source' else 0), state
            if case == 'runner_error':
                try:
                    finalizer.run_then_finalize(raw, finalizer.sha(raw), launch, runner)
                except ValueError as error:
                    assert 'launch evidence already exists' in str(error)
                else:
                    raise AssertionError('retried failed runner')
            assert state['calls'] == (0 if case == 'manifest' else 1), state
    finally:
        fixture.doCleanups()


class FinalizerTests(unittest.TestCase):
    def run_case(self, case):
        # Archive parsing and full numerical qualification stay in a fresh process
        # so earlier HWM cannot invalidate the separate resource regression test.
        run = subprocess.run([sys.executable, '-S', '-c',
            f'from test_finalize_readonly_trace import exercise; exercise({case!r})'], cwd=HERE,
            env={'PATH': os.defpath, 'PYTHONNOUSERSITE': '1', 'LANG': 'C', 'LC_ALL': 'C'},
            capture_output=True, text=True, timeout=60)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_all_real_validators_compose_over_synthetic_archived_evidence(self):
        self.run_case('complete')

    def test_no_runner_or_retry_on_invalid_manifest(self):
        self.run_case('manifest')

    def test_runner_exception_never_reaches_rehash(self):
        self.run_case('runner_error')

    def test_incomplete_telemetry_and_readiness_mismatch_block_rehash(self):
        for case in ('telemetry', 'readiness'):
            with self.subTest(case=case):
                self.run_case(case)

    def test_receipt_and_process_mismatch_block_rehash(self):
        for case in ('receipt_copy', 'process'):
            with self.subTest(case=case):
                self.run_case(case)

    def test_lifecycle_failure_blocks_rehash(self):
        self.run_case('lifecycle')

    def test_collector_capture_mismatch_blocks_rehash(self):
        self.run_case('capture')

    def test_metadata_changed_between_validators_blocks_rehash(self):
        self.run_case('analyzer_metadata')

    def test_updated_hashes_do_not_hide_corrupt_codec_logits(self):
        self.run_case('quality')

    def test_changed_frozen_source_fails_postflight_without_completion(self):
        self.run_case('source')


if __name__ == '__main__':
    unittest.main()
