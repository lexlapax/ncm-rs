import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_phase4_wmt14_mlx_bench as bench
from run_phase4_wmt14_mlx_bench import CELLS, checked_output, checked_rss, ordered_runs, summarize


class Wmt14MlxBenchTests(unittest.TestCase):
    def test_fresh_process_order(self):
        for cell in CELLS:
            runs = list(ordered_runs(cell))
            self.assertEqual(len(set(runs)), 6)
            orders = [[runtime for runtime, _ in runs[i:i + 2]] for i in range(0, len(runs), 2)]
            self.assertEqual(orders[0], orders[2])
            self.assertEqual(orders[0], orders[1][::-1])

    def test_checkpoint_screen_rotates_three_fresh_processes(self):
        variants = {'rust', 'rust_control', 'python'}
        for cell in CELLS:
            runs = list(ordered_runs(cell, with_control=True))
            self.assertEqual(len({stem for _, stem in runs}), 9)
            for position in range(3):
                self.assertEqual({runtime for runtime, _ in runs[position::3]}, variants)
            for start in range(0, 9, 3):
                self.assertEqual({runtime for runtime, _ in runs[start:start + 3]}, variants)

        orders = set()
        for cell in ((0, 16), (0, 32)):
            runs = list(ordered_runs(cell, with_control=True))
            orders.update(tuple(runtime for runtime, _ in runs[i:i + 3]) for i in range(0, 9, 3))
        self.assertEqual(len(orders), 6)

    def test_screen_live_and_replay_validate_both_rust_settings(self):
        cases = (('checkpoint', 'materialize_after', 'checkpoint_screen_level', 5, None, ['--with-control']),
                 ('decoder', 'decoder_rows', 'decoder_screen_rows', 2, None, ['--control', 'decoder']),
                 ('decoder', 'decoder_rows', 'decoder_screen_rows', 8, 2, ['--control', 'decoder']))
        for case in cases:
            with self.subTest(factor=case[0], control=case[4]):
                self.check_screen(*case)

    def check_screen(self, factor, field, manifest_key, value, control_value, flags):
        rows = [{'codes': [0] * 313, 'raw_ids': [0] * 64}]
        decoder_rows = value if factor == 'decoder' else 2
        allocator_trace = control_value is not None
        settings = {'backend': 'mlx', 'reusable_loader': True, 'deferred_head': True,
                    'drop_logits': True, 'release_outputs': False, 'cache_limit_bytes': None,
                    'decoder_rows': decoder_rows, 'materialize_after': 5}
        if allocator_trace:
            flags = [*flags, '--control-decoder-rows', str(control_value), '--allocator-trace']

        def fake_run(command, stem, directory):
            variant = stem.split('-')[2]
            candidate = variant == 'rust'
            self.assertEqual('allocator-trace' in command, allocator_trace and variant != 'python')
            self.assertEqual('materialize-after=5' in command,
                             candidate or (variant == 'rust_control' and factor == 'decoder'))
            wanted_decoder_rows = decoder_rows if candidate or factor == 'checkpoint' else control_value
            expected_decoder_args = (
                [] if variant == 'python' or wanted_decoder_rows is None else [f'decoder-rows={wanted_decoder_rows}'])
            self.assertEqual([arg for arg in command if arg.startswith('decoder-rows=')], expected_decoder_args)
            row = rows[0]
            report = {'cfg': 0, 'batch': 1, 'warmups': 5, 'load_seconds': 1,
                      'trials_seconds': [.03 if candidate else .02] * 30,
                      'mlx_memory': {'active': 2, 'peak': 8, 'cache': 4}}
            if variant == 'python':
                report.update({k: [v] for k, v in row.items()}, case_start=0)
                Path(command[4]).write_text(json.dumps(report))
            else:
                report.update(row, **settings, start=0)
                if variant == 'rust_control':
                    report[field] = control_value
                if allocator_trace:
                    report.update(allocator_boundaries=[report['mlx_memory']] * 31,
                                  trial_start_unix_seconds=list(range(1, 31)))
                (directory / f'{stem}.stdout').write_text(json.dumps(report))
            (directory / f'{stem}.meta.json').write_text(json.dumps(
                {'returncode': 0, 'child_peak_rss_platform_units': 128}))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / 'reports'
            reports.mkdir()
            reference = reports / 'base.json'
            reference.write_text(json.dumps({'outputs': rows}))
            reference_sha = hashlib.sha256(reference.read_bytes()).hexdigest()
            live = reports / 'live'
            replay = reports / 'replay'
            base_args = ['runner', '--cell', '0:1', '--base-reference', str(reference)]
            with (patch.object(bench, 'ROOT', root), patch.object(bench, 'REFERENCE_SHA256', {0: reference_sha}),
                  patch.object(bench, 'digest') as digest, patch.object(bench, 'subprocess') as subprocess,
                  patch.object(bench, 'platform') as platform, patch.dict(bench.os.environ),
                  patch.object(bench, 'run', side_effect=fake_run) as run,
                  patch('builtins.print')):
                digest.side_effect = lambda path: {
                    bench.PREFIX: '22c76d597ae51dffadf61661d6b7c4d4290fd14b8809bfa6f1959b5c668df93f',
                    bench.ARTIFACT / 'manifest.json':
                        '01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81',
                }.get(path, 'digest')
                platform.platform.return_value = 'macOS'
                platform.system.return_value = 'Darwin'
                platform.machine.return_value = 'arm64'
                subprocess.check_output.side_effect = lambda command, **_: {
                    ('git', 'status'): '',
                    ('git', 'rev-parse'): 'head',
                }.get(tuple(command[:2]), json.dumps({'core': str(root / 'core')}))
                with patch.object(sys, 'argv', [*base_args, '--output', str(live),
                                                *flags, '--materialize-after', '5',
                                                '--decoder-rows', str(decoder_rows)]):
                    bench.main()
                self.assertEqual(run.call_count, 9)
                manifest = json.loads((live / 'manifest.json').read_bytes())
                self.assertEqual(manifest['status'], 'complete')
                self.assertFalse(manifest['full_timing_matrix'])
                self.assertEqual(manifest[manifest_key], value)
                self.assertEqual(manifest['rust_settings'][field], value)
                self.assertEqual(manifest['rust_control_settings'][field], control_value)
                self.assertEqual(manifest.get('decoder_control_rows'), control_value)
                self.assertEqual(manifest.get('allocator_trace', False), allocator_trace)
                other = 'decoder_rows' if factor == 'checkpoint' else 'materialize_after'
                self.assertEqual(manifest['rust_settings'][other], manifest['rust_control_settings'][other])
                summary = json.loads((live / 'summary.json').read_bytes())['0:1']
                self.assertTrue(summary[f'{factor}_vs_control']['median_not_faster'])
                self.assertEqual(len(summary['python']['process_measurements']), 3)
                self.assertEqual('allocator_boundaries' in summary['rust']['process_measurements'][0],
                                 allocator_trace)
                self.assertNotIn('allocator_boundaries', summary['python']['process_measurements'][0])
                with patch.object(sys, 'argv', [*base_args, '--output', str(replay), '--replay', str(live)]):
                    bench.main()
                self.assertEqual(run.call_count, 9)
                self.assertEqual((replay / 'summary.json').read_bytes(), (live / 'summary.json').read_bytes())

                if control_value is not None:
                    manifest_path = live / 'manifest.json'
                    manifest_path.write_text(json.dumps(dict(manifest, decoder_control_rows=value)))
                    wrong_width = reports / 'wrong-width'
                    with (patch.object(sys, 'argv', [*base_args, '--output', str(wrong_width),
                                                     '--replay', str(live)]),
                          self.assertRaisesRegex(ValueError, 'invalid decoder control tile size')):
                        bench.main()
                    self.assertFalse(wrong_width.exists())
                    manifest_path.write_text(json.dumps(manifest))

                for stem in ('0-1-rust_control-1', '0-1-rust_control-2', '0-1-rust_control-3'):
                    report_path = live / f'{stem}.json'
                    report = json.loads(report_path.read_bytes())
                    report['drop_logits'] = False
                    report_path.write_text(json.dumps(report))
                wrong = reports / 'wrong'
                with (patch.object(sys, 'argv', [*base_args, '--output', str(wrong), '--replay', str(live)]),
                      self.assertRaisesRegex(ValueError, f'differ beyond the requested {factor}')):
                    bench.main()
                self.assertEqual(json.loads((wrong / 'manifest.json').read_bytes())['status'], 'incomplete')
                self.assertFalse((wrong / 'summary.json').exists())

    def test_allocator_trace_requires_complete_consistent_boundaries(self):
        counters = {'active': 2, 'peak': 8, 'cache': 4}
        boundaries = [dict(counters) for _ in range(31)]
        starts = list(range(1, 31))
        receipt = {'allocator_boundaries': boundaries, 'trial_start_unix_seconds': starts,
                   'mlx_memory': counters}
        self.assertEqual(bench.checked_allocator_trace(receipt)['allocator_boundaries'], boundaries)
        for update in (
            {'allocator_boundaries': boundaries[:-1]},
            {'trial_start_unix_seconds': starts[::-1]},
            {'allocator_boundaries': [dict(counters, cache=True), *boundaries[1:]]},
            {'allocator_boundaries': [dict(counters, peak=9), *boundaries[1:]]},
            {'mlx_memory': dict(counters, cache=0)},
        ):
            with self.subTest(fields=list(update)), self.assertRaisesRegex(ValueError, 'allocator trace'):
                bench.checked_allocator_trace({**receipt, **update})

    def test_rejects_mismatch_or_incomplete_trials(self):
        rows = [{'codes': [0] * 313, 'raw_ids': [0] * 64}]
        record = {'cfg': 0, 'batch': 1, 'warmups': 5, 'start': 0,
                  'trials_seconds': [.02] * 30, 'load_seconds': 1,
                  'mlx_memory': {'active': 3, 'cache': 4, 'peak': 5}, **rows[0]}
        self.assertEqual(checked_output(record, 'rust', 0, 1, rows)['load_seconds'], 1)
        for update in ({'start': 4}, {'codes': [False] * 313}, {'raw_ids': [1] * 64},
                       {'trials_seconds': [float('nan')] * 30}, {'trials_seconds': [.02] * 29}):
            with self.subTest(fields=list(update)), self.assertRaises(ValueError):
                checked_output(dict(record, **update), 'rust', 0, 1, rows)

    def test_process_rss_requires_macos_units(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = root / 'test.meta.json'
            meta.write_text(json.dumps({'returncode': 0, 'child_peak_rss_platform_units': 128}))
            self.assertEqual(checked_rss(root, 'test', True, 'Darwin'), 128)
            with self.assertRaisesRegex(ValueError, 'macOS process RSS'):
                checked_rss(root, 'test', True, 'Linux')
            meta.write_text(json.dumps({'returncode': 1, 'child_peak_rss_platform_units': 128}))
            with self.assertRaisesRegex(ValueError, 'invalid process status'):
                checked_rss(root, 'test', True, 'Darwin')

    def test_pooled_tail_can_exceed_each_process_median(self):
        records = [{'trials_seconds': [.01] * 25 + [.12] * 5, 'load_seconds': 1,
                    'peak_rss_bytes': 64, 'mlx_memory': {'active': 10, 'cache': 30, 'peak': 20}}] * 3
        summary = summarize(records)
        self.assertEqual(summary['median_ms'], 10)
        self.assertEqual(summary['p95_ms'], 120)
        self.assertEqual(summary['median_peak_rss_bytes'], 64)
        self.assertEqual(summary['median_mlx_memory']['cache'], 30)


if __name__ == '__main__':
    unittest.main()
