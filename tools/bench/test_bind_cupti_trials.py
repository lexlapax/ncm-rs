import copy
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from bind_cupti_trials import LAYOUT_SOURCES, ROOT, SCOPES, bind_trials, digest, main
from test_cupti_kernel_windows import encode, kernel, launch, records
from test_cupti_trace_boundaries import fixture


def report_fixture(layout='historical-block4'):
    return {
        'schema_version': 1, 'runtime': 'rust-candle', 'device': 'cuda', 'precision': 'fp32',
        'index_dtype': 'u32', 'warmups': 5, 'trials': 30,
        'diagnostic_profiling': layout == 'historical-block4',
        'source_sha256': dict(LAYOUT_SOURCES),
        'workload_sha256': digest((ROOT / 'tests/fixtures/codec-bench-v1.json').read_bytes()),
        'manifest_sha256': digest((ROOT / 'docs/results/2026-11-10-sudoku-artifacts/raw/artifact-manifest.json').read_bytes()),
        'measurements': [dict(batch=b, scope=s, first_ms=41., first_cuda_stream_span_ms=40.,
                              trials_ms=list(range(1, 31)), cuda_stream_spans_ms=list(range(31, 61)))
                         for b in (1, 2, 4) for s in SCOPES],
    }


def declaration(trace, report, layout='historical-block4', process=1):
    return {'schema_version': 1, 'layout': layout, 'trace_sha256': digest(trace),
            'report_sha256': digest(report), 'source_sha256': json.loads(report)['source_sha256'],
            'process_id': process}


def archived_join():
    root = ROOT / 'docs/results/2026-11-10-phase2-cuda-remediation'
    archive_path = root / 'raw/cupti-profiles.tar.gz'
    evidence = json.loads((root / 'evidence.json').read_text())
    if digest(archive_path.read_bytes()) != evidence['files']['raw/cupti-profiles.tar.gz']['sha256']:
        raise ValueError('historical archive hash mismatch')
    with tarfile.open(archive_path) as archive:
        trace = archive.extractfile('codec-diagnostic-block4-retained-cupti.txt').read()
        raw = archive.extractfile('codec-diagnostic-block4-retained.json').read()
    bound = bind_trials(trace, raw, declaration(trace, raw, process=2465343))
    report = json.loads(raw)
    row = next(r for r in report['measurements'] if (r['batch'], r['scope']) == (4, 'resident_decoder_block'))
    windows = bound['windows']
    return {'count': len(windows), 'first_matches': windows[0]['recorded_timing']['wall_ms'] == row['first_ms'],
            'warmups_unavailable': all(w['recorded_timing'] is None for w in windows[1:6]),
            'trials_match': [w['recorded_timing']['wall_ms'] for w in windows[6:]] == row['trials_ms'],
            'events_match': [w['recorded_timing']['cuda_stream_span_ms'] for w in windows[6:]] == row['cuda_stream_spans_ms'],
            'qualified': bound['benchmark_outputs_qualified'], 'provenance': bound['launch_provenance_verified']}


class TrialBindingTests(unittest.TestCase):
    def setUp(self):
        self.trace = encode(records())
        self.report = report_fixture()

    def bind(self, report=None, decl=None):
        raw = json.dumps(self.report if report is None else report).encode()
        return bind_trials(self.trace, raw, declaration(self.trace, raw) if decl is None else decl)

    def test_ordinal_join_preserves_first_and_marks_missing_warmups(self):
        result = self.bind()
        windows = result['windows']
        self.assertEqual(windows[0]['recorded_timing'], {'wall_ms': 41., 'cuda_stream_span_ms': 40.})
        self.assertEqual([w['recorded_timing'] for w in windows[1:6]], [None] * 5)
        self.assertEqual([w['recorded_timing']['wall_ms'] for w in windows[6:]], list(range(1, 31)))
        self.assertEqual([w['recorded_timing']['cuda_stream_span_ms'] for w in windows[6:]], list(range(31, 61)))
        self.assertTrue(result['diagnostic_only'])
        self.assertTrue(result['declared_hash_binding_verified'])
        for field in ('launch_provenance_verified', 'benchmark_outputs_qualified', 'activity_attribution_complete'):
            self.assertFalse(result[field])

    def test_synthetic_b1_binding_does_not_infer_quality_from_native_profile_flag(self):
        rows = fixture('codec-bench-b1')
        for ordinal in range(110, 146):
            start, corr = ((ordinal - 1) * 3 + 1) * 10, 10000 + ordinal
            rows.extend([launch(start + 3, start + 4, corr), kernel(start + 5, start + 8, corr)])
        trace = encode(rows)
        raw = json.dumps(report_fixture('codec-bench-b1')).encode()
        result = bind_trials(trace, raw, declaration(trace, raw, 'codec-bench-b1'))
        self.assertTrue(result['diagnostic_only'])
        self.assertFalse(result['benchmark_outputs_qualified'])
        self.assertEqual([w['ordinal'] for w in result['windows']], list(range(110, 146)))

    def test_hash_mismatches_and_process_identity(self):
        raw = json.dumps(self.report).encode()
        for key, value in (('trace_sha256', '0'*64), ('report_sha256', '0'*64), ('process_id', 2),
                           ('process_id', True), ('schema_version', True), ('layout', 'unknown')):
            decl = declaration(self.trace, raw); decl[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                bind_trials(self.trace, raw, decl)
        decl = declaration(self.trace, raw); decl['extra'] = True
        with self.assertRaises(ValueError):
            bind_trials(self.trace, raw, decl)

    def test_wrong_sources_even_when_report_hash_is_updated(self):
        for path in LAYOUT_SOURCES:
            changed = copy.deepcopy(self.report); changed['source_sha256'][path] = '0'*64
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'unreviewed'):
                self.bind(changed)
        raw = json.dumps(self.report).encode()
        decl = declaration(self.trace, raw); decl['source_sha256']['extra.rs'] = 'f'*64
        with self.assertRaisesRegex(ValueError, 'inventory mismatch'):
            bind_trials(self.trace, raw, decl)
        decl['source_sha256']['extra.rs'] = 'not-a-hash'
        with self.assertRaisesRegex(ValueError, 'digests'):
            bind_trials(self.trace, raw, decl)

    def test_protocol_artifact_and_profiling_mismatch(self):
        for key, value in (('runtime', 'python'), ('device', 'metal'), ('precision', 'fp16'),
                           ('index_dtype', 'i64'), ('warmups', 4), ('trials', 29),
                           ('schema_version', 2), ('diagnostic_profiling', False),
                           ('workload_sha256', '0'*64), ('manifest_sha256', '0'*64)):
            changed = copy.deepcopy(self.report); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.bind(changed)

    def test_missing_duplicate_reordered_scopes_and_invalid_timings(self):
        for change in (lambda r: r['measurements'].pop(),
                       lambda r: r['measurements'].reverse(),
                       lambda r: r['measurements'].append(r['measurements'][0]),
                       lambda r: r['measurements'][0]['trials_ms'].pop()):
            changed = copy.deepcopy(self.report); change(changed)
            with self.assertRaises(ValueError):
                self.bind(changed)
        for field in ('first_ms', 'first_cuda_stream_span_ms', 'trials_ms', 'cuda_stream_spans_ms'):
            for value in (0., -1., float('nan'), float('inf'), True):
                changed = copy.deepcopy(self.report)
                changed['measurements'][0][field] = [value]*30 if field.endswith('s_ms') else value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.bind(changed)

    def test_reviewed_layout_hashes_match_checkout(self):
        for path, expected in LAYOUT_SOURCES.items():
            self.assertEqual(digest((ROOT / path).read_bytes()), expected)

    def test_archived_report_pair_in_child(self):
        child = subprocess.run([sys.executable, '-c', 'import json; from test_bind_cupti_trials import '
                                'archived_join; print(json.dumps(archived_join()))'],
                               cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=30)
        self.assertEqual(child.returncode, 0, child.stderr)
        result = json.loads(child.stdout)
        self.assertEqual(result['count'], 36)
        for key in ('first_matches', 'warmups_unavailable', 'trials_match', 'events_match'):
            self.assertTrue(result[key], key)
        self.assertFalse(result['qualified'])
        self.assertFalse(result['provenance'])

    def test_cli_exclusive_output_and_binding_source_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace, report, decl, output = [Path(tmp) / n for n in ('trace', 'report', 'declaration', 'output')]
            trace.write_bytes(self.trace); report.write_text(json.dumps(self.report))
            decl.write_text(json.dumps(declaration(self.trace, report.read_bytes())))
            with patch('sys.argv', ['bind', str(trace), str(report), str(decl), str(output)]):
                main()
                result = json.loads(output.read_text())
                self.assertEqual(result['declaration_sha256'], digest(decl.read_bytes()))
                self.assertEqual(len(result['source_sha256']), 3)
                with self.assertRaises(FileExistsError):
                    main()
                output.unlink(); report.write_text('{}')
                with self.assertRaises(ValueError):
                    main()
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
