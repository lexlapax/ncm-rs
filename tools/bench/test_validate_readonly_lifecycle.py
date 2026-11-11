"""Clock tampering and CPU-only writer/reader integration; no GPU/model runs."""
import copy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import diagnostic_clock
from bounded_child import run_bounded
from readonly_trace_sequence import ORDER
from validate_readonly_lifecycle import validate, digest

BASE = datetime(2026, 11, 11, tzinfo=timezone.utc)
HERE = Path(__file__).resolve().parent


def stamp(second):
    ns = round(second * 1e9)
    return {'monotonic_before_ns': ns, 'monotonic_after_ns': ns + 10,
            'utc': (BASE + timedelta(seconds=second)).isoformat()}


def life(pid, parent, start, end):
    return {'pid': pid, 'parent_pid': parent, 'start': stamp(start), 'end': stamp(end)}


def receipt(pid, collector, parent, start, end):
    return {'status': 'complete', 'issues': [], 'returncode': 0, 'pid': pid,
            'direct_child_reaped': True, 'capture_eof': True, 'group_kill': 'sent', 'interrupt_signal': None,
            'started_utc': stamp(start + .01)['utc'], 'finished_utc': stamp(end - .01)['utc'],
            'elapsed_seconds_including_cleanup': end - start - .02, 'process_observer_enabled': True,
            'lifetime': life(collector, parent, start, end)}


def fixture():
    outer = receipt(2, 1, 99, 0, 1000)
    sup = dict(lifetime=life(2, 1, 1, 990), status='complete', worker=receipt(3, 2, 1, 2, 980),
               errors=[], parent_lost=False, orphan_observed=False, subreaper_restored=True,
               interrupt_signal=None, cleanup=dict(complete=True, errors=[], remaining=[], signalled=[], reaped=[]))
    coord = dict(lifetime=life(3, 2, 3, 970), status='complete', errors=[], interrupt_signal=None,
                 sequence_status='complete', telemetry_status='complete', observer_pid=4,
                 observer_shutdown=dict(verified=True, returncode=0, kill_escalation=False),
                 ready_observed_clock=stamp(9))
    telemetry = dict(lifetime=life(4, 3, 4, 960), status='complete', ready=True, errors=[], interrupt_signal=None,
                     cpu_receipt=receipt(5, 4, 3, 5, 950), gpu_pid=6,
                     gpu_start_clock=stamp(4.5), gpu_stop_clock=stamp(951),
                     gpu_shutdown=dict(verified=True, returncode=-15, kill_escalation=False))
    ready = dict(schema_version=1, kind='telemetry_ready', worker_pid=4, gpu_pid=6, record_clock=stamp(8))
    coord['ready_canonical_sha256'] = digest(ready)
    seq = dict(lifetime=life(3, 2, 10, 600), status='complete', stop_reason=None, jobs=[])
    observations = []
    for ordinal, condition in enumerate(ORDER, 1):
        start = 20 + ordinal * 30
        pid = ordinal + 100
        seq['jobs'].append(dict(ordinal=ordinal, condition=condition, receipt=receipt(pid, 3, 2, start, start + 2)))
        observations.append(dict(schema_version=1, pid=pid, status='complete', issues=[], samples=[
            dict(monotonic_start_ns=round((start + .5) * 1e9), monotonic_end_ns=round((start + .6) * 1e9))]))
    samples = [dict(monotonic_start_ns=stamp(6 + i)['monotonic_before_ns'],
                    monotonic_end_ns=stamp(6 + i)['monotonic_after_ns'], utc=stamp(6 + i)['utc']) for i in range(901)]
    for record in (outer, sup, sup['worker'], coord, seq, telemetry, telemetry['cpu_receipt'],
                   *(job['receipt'] for job in seq['jobs'])):
        record['schema_version'] = 1
    return [outer, sup, coord, seq, telemetry, ready, observations, samples]


class LifecycleTests(unittest.TestCase):
    def test_complete_clock_binding_without_stronger_claims(self):
        result = validate(*fixture())
        self.assertTrue(result['lifecycle_consistency_verified'])
        self.assertEqual(len(result['jobs']), 12)
        for name in ('launch_provenance_verified', 'telemetry_contents_qualified',
                     'benchmark_outputs_qualified', 'postflight_call_order_enforced'):
            self.assertFalse(result[name])
        self.assertEqual(result['utc_minus_monotonic_ns']['quantization_ns'], 1000)

    def test_missing_clock_never_inferred_from_elapsed_or_utc(self):
        data = fixture()
        del data[3]['jobs'][0]['receipt']['lifetime']
        with self.assertRaises(KeyError):
            validate(*data)

    def test_hierarchy_readiness_and_cleanup_tampering(self):
        mutations = (
            lambda d: d[0].update(direct_child_reaped=False),
            lambda d: d[1].update(schema_version=True),
            lambda d: d[5].update(schema_version=2),
            lambda d: d[6][0].update(schema_version=2),
            lambda d: d[1]['lifetime'].update(parent_pid=500),
            lambda d: d[1].update(subreaper_restored=False),
            lambda d: d[1]['cleanup']['reaped'].append({'pid': 500}),
            lambda d: d[1]['worker'].update(pid=500),
            lambda d: d[2]['observer_shutdown'].update(kill_escalation=True),
            lambda d: d[2].update(ready_canonical_sha256='0'*64),
            lambda d: d[2].update(ready_observed_clock=stamp(50)),
            lambda d: d[3]['lifetime'].update(start=stamp(1)),
            lambda d: d[3]['jobs'][0]['receipt']['lifetime'].update(pid=500),
            lambda d: d[4]['gpu_shutdown'].update(returncode=1),
            lambda d: d[4]['cpu_receipt'].update(pid=6),
            lambda d: d[4].update(status='failed'),
            lambda d: d[4].update(gpu_pid=500),
            lambda d: d[4].update(gpu_start_clock=stamp(7)),
            lambda d: d[4].update(gpu_stop_clock=stamp(949)),
            lambda d: d[5].update(worker_pid=500),
            lambda d: d[3]['jobs'][0].update(condition='CP'),
        )
        for mutate in mutations:
            data = fixture()
            mutate(data)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                validate(*data)

    def test_outside_sample_utc_step_and_unbracketed_jobs_fail(self):
        mutations = (
            lambda d: d[6][0]['samples'][0].update(monotonic_start_ns=0),
            lambda d: d[6][0]['samples'][0].update(monotonic_end_ns=10**15),
            lambda d: d[6][0]['samples'][0].update(monotonic_start_ns=True),
            lambda d: d[6][0].update(pid=999),
            lambda d: d[3]['jobs'][0]['receipt'].update(elapsed_seconds_including_cleanup=10**6),
            lambda d: d[3]['jobs'][0]['receipt']['lifetime']['start'].update(utc=stamp(51)['utc']),
            lambda d: d[7].__setitem__(-1, dict(monotonic_start_ns=60*10**9, monotonic_end_ns=60*10**9+10,
                                               utc=stamp(60)['utc'])),
            lambda d: d[7][1].update(utc=stamp(8)['utc']),
        )
        for mutate in mutations:
            data = fixture()
            mutate(data)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                validate(*data)
        data = fixture()
        data[7] = data[7][:100]
        with self.assertRaisesRegex(ValueError, 'not bracketed'):
            validate(*data)

    def test_clock_writer_copies_start_and_brackets_utc(self):
        start = diagnostic_clock.begin()
        original = copy.deepcopy(start)
        end = diagnostic_clock.finish(start)
        self.assertEqual(start, original)
        self.assertNotIn('end', start)
        self.assertEqual(end['pid'], os.getpid())
        self.assertLessEqual(end['start']['monotonic_after_ns'], end['end']['monotonic_before_ns'])

    @unittest.skipUnless(sys.platform == 'linux', 'Linux independently supervised CPU children')
    def test_real_writers_and_owned_process_samples_compose(self):
        # Five seconds of fake observers, all plain Python workloads. This is NOT
        # CUDA/injection/CPU2/full-900s telemetry qualification or launch readiness.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = {'PATH': os.defpath, 'PYTHONNOUSERSITE': '1', 'LANG': 'C', 'LC_ALL': 'C'}
            argv = [sys.executable, '-c', 'from test_readonly_trace_coordinator import supervised_driver; '
                    f'supervised_driver({str(root)!r},"lifecycle")']
            outer = run_bounded(argv, cwd=HERE, env=env, capture_path=root / 'outer.txt',
                                receipt_path=root / 'outer.receipt.json', timeout_seconds=20,
                                output_limit_bytes=1024**2)
            def read(name):
                return json.loads((root / name).read_text())
            sup = read('supervision/supervisor.json')
            coord = read('collection/collection.json')
            seq = read('collection/sequence/sequence-result.json')
            tel = read('collection/telemetry/telemetry.json')
            ready = read('collection/telemetry/ready.json')
            self.assertEqual(outer['status'], 'complete', (root / 'outer.txt').read_text())
            self.assertEqual(sup['status'], 'complete', {'supervisor': sup, 'collection': coord, 'telemetry': tel})
            self.assertEqual(coord['status'], 'complete', coord)
            observations = [read(f'collection/sequence/{i:02d}-{c}.process.json') for i, c in enumerate(ORDER, 1)]
            samples = [json.loads(line) for line in (root / 'collection/telemetry/cpu.jsonl').read_text().splitlines()]
            result = validate(outer, sup, coord, seq, tel, ready, observations, samples[1:-1])
            self.assertTrue(result['lifecycle_consistency_verified'])
            self.assertEqual(len(result['jobs']), 12)
            changed = copy.deepcopy(observations)
            changed[0]['samples'][0]['monotonic_start_ns'] = 0
            with self.assertRaises(ValueError):
                validate(outer, sup, coord, seq, tel, ready, changed, samples[1:-1])


if __name__ == '__main__':
    unittest.main()
