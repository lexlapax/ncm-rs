"""Offline validation using the real serializer over a virtual clock/fake procfs."""
import copy
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import sample_linux_cpu as sampler
from readonly_trace_sequence import ORDER
from readonly_trace_telemetry import GPU_HEADER
from validate_readonly_telemetry import cpu_records, gpu_records, validate

BASE = datetime(2026, 11, 11, 12, tzinfo=timezone.utc)
UUID = 'GPU-12345678-1234-1234-1234-123456789abc'
SOURCE = hashlib.sha256(Path(sampler.__file__).read_bytes()).hexdigest()


def cpu_fixture():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        proc, sys = root / 'proc', root / 'sys'
        proc.mkdir()
        (proc / 'loadavg').write_text('0.10 0.20 0.30 1/10 123\n')
        for cpu in (0, 2, 3):
            directory = sys / f'cpu{cpu}/cpufreq'
            directory.mkdir(parents=True)
            (directory / 'scaling_cur_freq').write_text('1000000\n')
        topology = sys / 'cpu2/topology'
        topology.mkdir()
        (topology / 'thread_siblings_list').write_text('2,3\n')
        (sys / 'cpu2/cpufreq/scaling_governor').write_text('powersave\n')
        (sys / 'cpu2/cpufreq/scaling_driver').write_text('fake_driver\n')

        class Clock:
            seconds = 0

            @classmethod
            def monotonic(cls):
                return cls.seconds + 1000

            @classmethod
            def monotonic_ns(cls):
                return int(cls.monotonic() * 10**9)

            @classmethod
            def sleep(cls, seconds):
                cls.seconds += seconds
                tick = int(cls.seconds) + 1
                (proc / 'stat').write_text(''.join(f'{cpu} {tick} 0 0 {tick} 0 0 0 0 0 0\n'
                                                  for cpu in ('cpu', 'cpu0', 'cpu2', 'cpu3'))
                                          + f'ctxt {tick}\nprocs_running 1\nprocs_blocked 0\n')

        class Date:
            @staticmethod
            def now(zone):
                return (BASE + timedelta(seconds=Clock.seconds)).astimezone(zone)

        Clock.sleep(0)
        with patch.object(sampler, 'time', Clock), patch.object(sampler, 'datetime', Date), \
                patch.object(sampler.platform, 'system', return_value='Linux'), \
                patch.object(sampler.os, 'sched_getaffinity', side_effect=[{0, 2, 3}, {0}]), \
                patch.object(sampler.os, 'sched_setaffinity'):
            sampler.run(root / 'cpu.jsonl', 2, 900, 1, 0, proc=proc, sys_cpu=sys)
        return (root / 'cpu.jsonl').read_bytes()


def cpu_bytes(rows):
    return ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows).encode()


def gpu_fixture():
    rows = [GPU_HEADER]
    for index in range(901):
        rows.append([(BASE + timedelta(seconds=index)).strftime('%Y/%m/%d %H:%M:%S.%f'), UUID,
                     '10 %', '100 MiB', '40', '25.50 W', '1710 MHz', '7000 MHz'])
    return gpu_bytes(rows)


def gpu_bytes(rows):
    output = io.StringIO()
    csv.writer(output).writerows(rows)
    return output.getvalue().encode()


def job_fixture():
    return [dict(ordinal=i+1, condition=condition,
                 receipt=dict(pid=i+100, status='complete', issues=[], returncode=0,
                              direct_child_reaped=True, capture_eof=True, interrupt_signal=None,
                              started_utc=(BASE + timedelta(seconds=10+i*20.0+.25)).isoformat(),
                              finished_utc=(BASE + timedelta(seconds=11+i*20.0+.75)).isoformat()))
            for i, condition in enumerate(ORDER)]


class TelemetryValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cpu = cpu_fixture()
        cls.gpu = gpu_fixture()

    def validate(self, cpu=None, gpu=None, jobs=None):
        return validate(self.cpu if cpu is None else cpu, self.gpu if gpu is None else gpu,
                        job_fixture() if jobs is None else jobs, expected_cpu_source=SOURCE,
                        gpu_uuid=UUID, timezone_name='UTC')

    def test_real_sampler_serialization_with_virtual_clock_and_twelve_windows(self):
        result = self.validate()
        self.assertTrue(result['telemetry_contents_validated'])
        self.assertTrue(result['receipt_windows_bracketed'])
        self.assertEqual(result['cpu_completion']['samples'], 901)
        self.assertEqual(result['cpu_sha256'], hashlib.sha256(self.cpu).hexdigest())
        self.assertEqual(len(result['missing_cpu_optional']['pressure']), 2703)
        self.assertEqual(result['unavailable_cpu_activity'], [])
        self.assertEqual(result['jobs'][0]['cpu_interval_end_indices'], [11, 12])
        self.assertEqual(result['jobs'][0]['gpu_sample_indices_inside'], [11])
        self.assertEqual(len(result['gpu_samples']), 901)
        self.assertFalse(result['launch_provenance_verified'])
        self.assertFalse(result['continuous_gpu_observation_verified'])
        self.assertFalse(result['performance_acceptance_evaluated'])

    def test_offline_cli_binds_inputs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'cpu.jsonl').write_bytes(self.cpu)
            (root / 'gpu.csv').write_bytes(self.gpu)
            raw = json.dumps(dict(schema_version=1, status='complete', stop_reason=None, jobs=job_fixture())).encode()
            (root / 'sequence.json').write_bytes(raw)
            command = [sys.executable, str(Path(__file__).with_name('validate_readonly_telemetry.py').resolve()),
                       str(root / 'cpu.jsonl'), str(root / 'gpu.csv'), str(root / 'sequence.json'), str(root / 'result.json'),
                       '--cpu-source-sha256', SOURCE, '--gpu-uuid', UUID, '--timezone', 'UTC']
            env = {'PATH': os.defpath, 'PYTHONNOUSERSITE': '1'}
            run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stderr)
            original = (root / 'result.json').read_bytes()
            result = json.loads(original)
            self.assertEqual(result['sequence_result_sha256'], hashlib.sha256(raw).hexdigest())
            self.assertIn('validate_readonly_telemetry.py', result['tool_source_sha256'])
            run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(run.returncode, 0)
            self.assertEqual((root / 'result.json').read_bytes(), original)

    def test_archived_v2_cpu_in_fresh_interpreter(self):
        # Keep large fixture parsing outside the test runner's inherited RSS.
        archive = Path(__file__).resolve().parents[2] / 'docs/results/2026-11-11-phase2-optimization/raw/linux-diagnostic-v2.tar.xz'
        script = '''
import hashlib, pathlib, sys, tarfile
from validate_readonly_telemetry import cpu_records
archive = pathlib.Path(sys.argv[1])
assert hashlib.sha256(archive.read_bytes()).hexdigest() == 'feb5ec2138e070811614af86f05777d0e0c09616ca2134e1730a2533e95a368c'
with tarfile.open(archive) as bundle:
    members = [m for m in bundle.getmembers() if m.name.split('/')[-1] == 'phase2-linux-diagnostic-v2-cpu.jsonl']
    assert len(members) == 1 and members[0].isfile() and members[0].size < 16*1024**2
    raw = bundle.extractfile(members[0]).read()
result = cpu_records(raw, sys.argv[2])
assert len(result['samples']) == 901
assert not any(result['missing_optional'].values())
assert not result['unavailable_activity']
'''
        result = subprocess.run([sys.executable, '-c', script, str(archive), SOURCE],
                                cwd=Path(__file__).parent, env={'PATH': os.defpath, 'PYTHONNOUSERSITE': '1'},
                                text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_late_samples_and_unavailable_counter_deltas_remain_explicit(self):
        rows = [json.loads(line) for line in self.cpu.splitlines()]
        rows[2]['monotonic_start_ns'] += 300000000
        rows[2]['monotonic_end_ns'] += 300000000
        rows[2]['utc'] = (BASE + timedelta(seconds=1.3)).isoformat()
        rows[2]['lateness_seconds'] = .3
        rows[2]['cpu_ticks']['cpu2']['user'] = 0
        for index in (2, 3):
            rows[index]['activity_since_previous'] = sampler.activity(rows[index-1]['cpu_ticks'], rows[index]['cpu_ticks'])
        result = self.validate(cpu=cpu_bytes(rows))
        self.assertEqual(result['cpu_lateness_seconds'][1], .3)
        self.assertEqual(result['cpu_sample_gaps_seconds'][:2], [1.3, .7])
        self.assertEqual(result['unavailable_cpu_activity'], [{'index': 1, 'cpu': 'cpu2',
                         'details': {'available': False, 'reason': 'counter decreased or CPU changed'}}])

    def test_cpu_truncation_duplicate_keys_source_and_metadata(self):
        rows = [json.loads(line) for line in self.cpu.splitlines()]
        bad = [self.cpu[:-1], cpu_bytes(rows[:-1]), cpu_bytes(rows[:100] + rows[101:]),
               self.cpu.replace(b'"kind": "sample"', b'"kind":"sample", "kind":"sample"', 1)]
        for key, value in (('source_sha256', 'b'*64), ('smt_siblings', [2, 12]),
                           ('observer_affinity', [2]), ('tick_rate_hz', True), ('seconds', 899),
                           ('platform', float('nan')), ('limitations', [float('inf')])):
            changed = copy.deepcopy(rows)
            changed[0][key] = value
            bad.append(cpu_bytes(changed))
        for raw in bad:
            with self.subTest(size=len(raw)), self.assertRaises(ValueError):
                cpu_records(raw, SOURCE)

    def test_cpu_counters_arithmetic_optional_fields_and_nonfinite_values(self):
        mutations = [lambda r: r[2]['cpu_ticks']['cpu2'].update(user=-1),
                     lambda r: r[2]['cpu_ticks']['cpu2'].update(user=True),
                     lambda r: r[2]['activity_since_previous']['cpu2'].update(total_ticks=99),
                     lambda r: r[2]['system_counters'].update(ctxt=0),
                     lambda r: r[2]['frequency_khz']['0'].update(value='nan'),
                     lambda r: r[2]['loadavg'].update(value='0 0 0 11/10 123'),
                     lambda r: r[2]['pressure'].update(cpu={'available': False}),
                     lambda r: r[2]['pressure'].update(cpu={'available': True, 'value':'some avg10=101 avg60=0 avg300=0 total=1'}),
                     lambda r: r[2].update(lateness_seconds=float('nan'))]
        for mutate in mutations:
            rows = [json.loads(line) for line in self.cpu.splitlines()]
            mutate(rows)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                cpu_records(cpu_bytes(rows), SOURCE)

    def test_cpu_temporal_schedule_utc_steps_and_completion_inconsistency(self):
        mutations = [lambda r: r[2].update(index=0),
                     lambda r: r[2].update(scheduled_elapsed_seconds=1.1),
                     lambda r: r[2].update(monotonic_end_ns=r[2]['monotonic_start_ns']-1),
                     lambda r: r[2].update(utc='2026-11-11T12:00:01'),
                     lambda r: r[2].update(utc=(BASE+timedelta(seconds=1.1)).isoformat()),
                     lambda r: r[2].update(lateness_seconds=5),
                     lambda r: r[-1].update(actual_elapsed_seconds=899),
                     lambda r: r[-1].update(samples=True)]
        for mutate in mutations:
            rows = [json.loads(line) for line in self.cpu.splitlines()]
            mutate(rows)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                cpu_records(cpu_bytes(rows), SOURCE)

    def test_gpu_uuid_units_truncation_nonfinite_duplicates_and_dst(self):
        original = list(csv.reader(io.StringIO(self.gpu.decode())))
        changes = [(1, 'GPU-wrong'), (2, '101 %'), (3, '1 GB'), (4, 'NaN'), (5, 'inf W'), (6, '-1 MHz')]
        for column, value in changes:
            rows = copy.deepcopy(original)
            rows[2][column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                gpu_records(gpu_bytes(rows), UUID, 'UTC')
        for raw in (self.gpu[:-1], gpu_bytes([original[0], original[1], original[1]])):
            with self.assertRaises(ValueError):
                gpu_records(raw, UUID, 'UTC')
        for stamp in ('2026/11/01 01:30:00.000', '2026/03/08 02:30:00.000'):
            rows = copy.deepcopy(original[:3])
            rows[1][0] = stamp
            with self.assertRaisesRegex(ValueError, 'ambiguous/nonexistent'):
                gpu_records(gpu_bytes(rows), UUID, 'America/Los_Angeles')
        shifted = gpu_records(self.gpu, UUID, 'America/Los_Angeles')
        self.assertEqual(shifted[0]['utc'], '2026-11-11T20:00:00+00:00')

    def test_missing_gpu_values_and_gaps_are_retained_not_continuous_coverage(self):
        rows = list(csv.reader(io.StringIO(self.gpu.decode())))
        rows[1][5] = '[Not Supported]'
        del rows[11:15]
        result = self.validate(gpu=gpu_bytes(rows))
        self.assertEqual(result['gpu_samples'][0]['power_w'], {'available': False, 'reason': '[Not Supported]'})
        self.assertIn(5.0, result['gpu_sample_gaps_seconds'])
        self.assertFalse(result['continuous_gpu_observation_verified'])
        self.assertTrue(result['receipt_windows_bracketed'])
        self.assertEqual(result['jobs'][0]['gpu_sample_indices_inside'], [])

    def test_incomplete_outside_duplicate_and_overlapping_receipt_windows(self):
        mutations = [lambda jobs: jobs.pop(), lambda jobs: jobs.reverse(),
                     lambda jobs: jobs[1]['receipt'].update(pid=100),
                     lambda jobs: jobs[0]['receipt'].update(returncode=True),
                     lambda jobs: jobs[0]['receipt'].update(status='failed'),
                     lambda jobs: jobs[1]['receipt'].update(started_utc=jobs[0]['receipt']['started_utc']),
                     lambda jobs: jobs[-1]['receipt'].update(finished_utc=(BASE+timedelta(seconds=901)).isoformat())]
        for mutate in mutations:
            jobs = job_fixture()
            mutate(jobs)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.validate(jobs=jobs)
        rows = list(csv.reader(io.StringIO(self.gpu.decode())))
        with self.assertRaisesRegex(ValueError, 'does not bracket'):
            self.validate(gpu=gpu_bytes(rows[:200]))


if __name__ == '__main__':
    unittest.main()
