"""Synthetic collection integration; archived reports never masquerade as new runs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import analyze_readonly_trace as analyzer
from bind_cupti_trials import LAYOUT_SOURCES, digest
from qualify_diagnostic_codec import EVIDENCE, NATIVE_SOURCES, ROOT
from readonly_trace_sequence import ORDER
from test_bind_cupti_trials import report_fixture
from test_cupti_kernel_windows import encode, kernel, launch
from test_cupti_trace_boundaries import fixture


def make_collection(root, archived=False):
    sources = {impl: {**{p: impl.lower() * 64 for p in NATIVE_SOURCES}, **LAYOUT_SOURCES}
               for impl in ('A', 'C')}
    archived_reports = {}
    if archived:
        evidence = json.loads(EVIDENCE.read_bytes())
        archive_path = ROOT / evidence['archive']['path']
        assert digest(archive_path.read_bytes()) == evidence['archive']['sha256']
        with tarfile.open(archive_path) as archive:
            for impl, runtime in (('A', 'previous'), ('C', 'candidate')):
                archived_reports[impl] = [json.loads(archive.extractfile(
                    f'phase2-linux-diagnostic-v2/{runtime}/codec-rust-{i}.json').read())
                    for i in range(1, 7)]
                sources[impl] = archived_reports[impl][0]['source_sha256']
    plan = dict(schema_version=1, sequence_seconds=600, child_seconds=120,
                capture_limit_bytes=analyzer.CAPTURE_BYTES, total_limit_bytes=analyzer.TOTAL_BYTES,
                minimum_initial_free_bytes=analyzer.MIN_FREE_BYTES, jobs=[])
    result = dict(schema_version=1, status='complete', stop_reason=None, captured_bytes=0, jobs=[])
    occurrences = {'A': 0, 'C': 0}
    for ordinal, condition in enumerate(ORDER, 1):
        prefix = f'{ordinal:02d}-{condition}'
        impl = condition[0]
        if archived:
            report = archived_reports[impl][occurrences[impl]]
        else:
            report = report_fixture('codec-bench-b1')
            report['source_sha256'] = sources[impl]
            report.update(load_trials_ms=list(range(1, 31)), first_load_ms=1., first_execution_ms=2.)
            report['measurements'][0]['first_ms'] += ordinal
        occurrences[impl] += 1
        raw = json.dumps(report).encode()
        report_path = str(root / (prefix + '.report.json'))
        (root / (prefix + '.report.json')).write_bytes(raw)
        pid = 1000 + ordinal
        capture = b''
        if condition.endswith('T'):
            rows = fixture('codec-bench-b1')
            for window in range(110, 146):
                start, corr = ((window - 1) * 3 + 1) * 10, 10000 + window
                rows.extend([launch(start + 3, start + 4, corr), kernel(start + 5, start + 8, corr)])
            capture = encode(rows).replace(b'processId 1,', f'processId {pid},'.encode())
        (root / (prefix + '.capture')).write_bytes(capture)
        job = dict(ordinal=ordinal, condition=condition, argv=['/synthetic/codec', report_path],
                   cwd='/synthetic/worktree', report_path=report_path)
        receipt = dict(schema_version=1, status='complete', issues=[], returncode=0,
                       direct_child_reaped=True, capture_eof=True, interrupt_signal=None,
                       group_kill='sent', pid=pid, argv=job['argv'], cwd=job['cwd'],
                       timeout_seconds=120, output_limit_bytes=analyzer.CAPTURE_BYTES,
                       report_limit_bytes=analyzer.REPORT_LIMIT,
                       capture_sha256=digest(capture), captured_bytes=len(capture), observed_bytes=len(capture),
                       report_capture=dict(path=report_path, post_exit_hash_verified=True,
                                           sha256=digest(raw), size_bytes=len(raw)))
        (root / (prefix + '.receipt.json')).write_text(json.dumps(receipt))
        plan['jobs'].append(job)
        result['jobs'].append(dict(ordinal=ordinal, condition=condition, receipt=receipt))
        result['captured_bytes'] += len(capture)
    (root / 'sequence-plan.json').write_text(json.dumps(plan))
    (root / 'sequence-result.json').write_text(json.dumps(result))
    return sources


def synthetic_cases():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sources = make_collection(root)
        with patch.object(analyzer, 'qualify', return_value={'synthetic_qualifier_stub': True}) as qualifier:
            result = analyzer.analyze(root, sources)
            assert qualifier.call_count == 4
            assert len(result['jobs']) == 12 and len(result['traced_trials']) == 6
            assert len(result['instrumentation_comparisons']) == 78
            assert sum(r['scope'] == 'checked_load' for r in result['instrumentation_comparisons']) == 6
            assert all(len(call.args[0]) == 3 for call in qualifier.call_args_list)
            assert all(len(t['binding']['windows']) == 36 for t in result['traced_trials'])
            assert all(t['binding']['windows'][1]['recorded_timing'] is None for t in result['traced_trials'])
            assert not result['launch_provenance_verified'] and not result['performance_acceptance_evaluated']
        pristine = {p.name: p.read_bytes() for p in root.iterdir()}
        cases = ['order', 'limits', 'incomplete', 'receipt_copy', 'failed_child', 'pid', 'reused_pid',
                 'report_hash', 'capture_hash', 'aggregate', 'missing', 'report_path', 'duplicate_report']
        for case in cases:
            for name, raw in pristine.items():
                (root / name).write_bytes(raw)
            plan = json.loads(pristine['sequence-plan.json'])
            result = json.loads(pristine['sequence-result.json'])
            if case == 'order':
                plan['jobs'][0]['condition'] = 'CP'
            elif case == 'limits':
                plan['child_seconds'] = 121
            elif case == 'incomplete':
                result['status'] = 'failed'
            elif case == 'receipt_copy':
                result['jobs'][0]['receipt']['pid'] += 1
            elif case in ('failed_child', 'pid', 'reused_pid', 'report_path'):
                index = 2 if case == 'pid' else 1
                receipt = result['jobs'][index]['receipt']
                if case == 'failed_child':
                    receipt['status'] = 'failed'
                elif case == 'report_path':
                    receipt['report_capture']['path'] = '/elsewhere/report.json'
                else:
                    receipt['pid'] = 1001 if case == 'reused_pid' else 9999
                (root / f'{index+1:02d}-{ORDER[index]}.receipt.json').write_text(json.dumps(receipt))
            elif case in ('report_hash', 'capture_hash'):
                suffix = 'report.json' if case == 'report_hash' else 'capture'
                (root / ('01-AP.' + suffix)).write_bytes(b'tampered')
            elif case == 'aggregate':
                result['captured_bytes'] += 1
            elif case == 'missing':
                (root / '12-CP.report.json').unlink()
            elif case == 'duplicate_report':
                raw = pristine['01-AP.report.json']
                (root / '02-CP.report.json').write_bytes(raw)
                receipt = result['jobs'][1]['receipt']
                receipt['report_capture'].update(sha256=digest(raw), size_bytes=len(raw))
                (root / '02-CP.receipt.json').write_text(json.dumps(receipt))
            (root / 'sequence-plan.json').write_text(json.dumps(plan))
            (root / 'sequence-result.json').write_text(json.dumps(result))
            with patch.object(analyzer, 'qualify'):
                try:
                    analyzer.analyze(root, sources)
                except (ValueError, FileNotFoundError):
                    pass
                else:
                    raise AssertionError('accepted invalid synthetic case: ' + case)
        return len(cases)


def archived_cases():
    # Real archived outputs, but synthetic traces/receipts: never actual collection provenance.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sources = make_collection(root, archived=True)
        result = analyzer.analyze(root, sources)
        assert all(len(q['quality']) == 12 for q in result['qualification'].values())
        assert not result['launch_provenance_verified']
        source_path, output = root / 'sources.json', root / 'analysis.json'
        source_path.write_text(json.dumps(sources))
        with patch('sys.argv', ['analyze', str(root), str(source_path), str(output)]):
            analyzer.main()
            assert json.loads(output.read_text())['reported_codec_outputs_qualified']
            try:
                analyzer.main()
            except FileExistsError:
                pass
            else:
                raise AssertionError('overwrote analyzer output')
        # Correct hashes/receipts must not bypass the actual numerical validator.
        report_path = root / '01-AP.report.json'
        report = json.loads(report_path.read_bytes())
        report['measurements'][1]['first_logits'][0] = 99.
        raw = json.dumps(report).encode()
        report_path.write_bytes(raw)
        result = json.loads((root / 'sequence-result.json').read_bytes())
        receipt = result['jobs'][0]['receipt']
        receipt['report_capture'].update(sha256=digest(raw), size_bytes=len(raw))
        (root / '01-AP.receipt.json').write_text(json.dumps(receipt))
        (root / 'sequence-result.json').write_text(json.dumps(result))
        try:
            analyzer.analyze(root, sources)
        except ValueError as error:
            assert 'logit tolerance' in str(error), str(error)
        else:
            raise AssertionError('bad logits passed integrated qualification')
        return {'groups': 4, 'scopes_per_group': 12, 'bad_logits_rejected': True}


class ReadonlyAnalysisTests(unittest.TestCase):
    def child(self, function):
        result = subprocess.run([sys.executable, '-c',
                                 f'import json; from test_analyze_readonly_trace import {function}; '
                                 f'print(json.dumps({function}()))'],
                                cwd=Path(__file__).parent, capture_output=True, text=True, check=True,
                                env={'PATH': os.defpath, 'PYTHONNOUSERSITE': '1'})
        return json.loads(result.stdout)

    def test_synthetic_full_sequence_and_negative_bindings(self):
        self.assertEqual(self.child('synthetic_cases'), 13)

    def test_archived_quality_integration_and_exclusive_cli(self):
        self.assertEqual(self.child('archived_cases'), {'groups': 4, 'scopes_per_group': 12,
                                                      'bad_logits_rejected': True})

    def test_bounded_input_rejects_symlink_fifo_and_oversize(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'target'
            target.write_bytes(b'12345')
            with self.assertRaises(ValueError):
                analyzer.read_bounded(target, 4)
            link = root / 'link'
            link.symlink_to(target)
            with self.assertRaises(OSError):
                analyzer.read_bounded(link, 8)
            fifo = root / 'fifo'
            os.mkfifo(fifo)
            with self.assertRaises(ValueError):
                analyzer.read_bounded(fifo, 8)


if __name__ == '__main__':
    unittest.main()
