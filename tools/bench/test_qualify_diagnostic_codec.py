import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from qualify_diagnostic_codec import (EVIDENCE, LAYOUT_SOURCES, NATIVE_SOURCES, ROOT,
                                      frozen_references, main, qualify, validate_group)
from phase2_summary import reports as acceptance_reports


def archived_cases():
    """Run large fixtures in a child to preserve the parent suite's RSS baseline."""
    evidence = json.loads(EVIDENCE.read_bytes())
    archive_path = ROOT / evidence['archive']['path']
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == evidence['archive']['sha256']
    with tarfile.open(archive_path) as archive:
        raw = [archive.extractfile(f'phase2-linux-diagnostic-v2/candidate/codec-rust-{i}.json').read()
               for i in (1, 2, 3)]
    runs = [json.loads(r) for r in raw]
    sources = runs[0]['source_sha256']
    result = qualify(raw, sources)
    assert result['reported_codec_outputs_qualified'] and result['diagnostic_only']
    assert not result['performance_acceptance_evaluated'] and not result['launch_provenance_verified']
    assert len(result['quality']) == 12
    assert all('runtimes' not in row for row in result['quality'])
    assert len(result['frozen_reference_binding']['reports']) == 3

    changes = [
        lambda r: r.update(warmups=4),
        lambda r: r['source_sha256'].update({'Cargo.lock': '0'*64}),
        lambda r: r['measurements'].pop(),
        lambda r: r['measurements'][0].update(output_ids_sha256='0'*64),
        lambda r: r['measurements'][1]['first_logits'].__setitem__(0, 99.),
        lambda r: r['measurements'][0]['trials_ms'].__setitem__(0, float('nan')),
        lambda r: r['load_trials_ms'].pop(),
        lambda r: r['instrumented_stages'].pop(),
    ]
    for change in changes:
        changed = copy.deepcopy(runs)
        change(changed[0])
        try:
            qualify([json.dumps(r).encode() for r in changed], sources)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid report passed full output qualification')
    # Whitespace changes do not make copied reports independent.
    try:
        qualify([raw[0], json.dumps(runs[0], indent=1).encode(), raw[2]], sources)
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate report accepted')

    profiled = copy.deepcopy(runs)
    for run in profiled:
        run['diagnostic_profiling'] = True
    diagnostic = qualify([json.dumps(r).encode() for r in profiled], sources)
    assert diagnostic['reported_codec_outputs_qualified']
    assert not diagnostic['performance_acceptance_evaluated']
    references, _ = frozen_references()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for i, (native, reference) in enumerate(zip(profiled, references), 1):
            (root / f'codec-rust-{i}.json').write_text(json.dumps(native))
            (root / f'codec-python-{i}.json').write_text(json.dumps(reference))
        try:
            acceptance_reports(root, 'codec')
        except ValueError as error:
            assert 'profiled diagnostic' in str(error)
        else:
            raise AssertionError('acceptance reader accepted instrumented reports')
        source_path, output = root / 'sources.json', root / 'qualified.json'
        source_path.write_text(json.dumps(sources))
        args = ['qualify', str(output), *(str(root / f'codec-rust-{i}.json') for i in (1, 2, 3)),
                '--expected-sources', str(source_path)]
        with patch('sys.argv', args):
            main()
            assert json.loads(output.read_bytes())['reported_codec_outputs_qualified']
            try:
                main()
            except FileExistsError:
                pass
            else:
                raise AssertionError('existing output overwritten')
    return {'qualified_scopes': 12, 'invalid_report_cases_rejected': len(changes) + 1,
            'profiled_acceptance_rejected': True, 'cli_verified': True}


def small_group():
    return [dict(schema_version=1, runtime='rust-candle', device='cuda', precision='fp32',
                 warmups=5, trials=30, source_sha256={}, diagnostic_profiling=False,
                 index_dtype='u32', distinct=i) for i in range(3)]


class DiagnosticQualificationTests(unittest.TestCase):
    def test_group_protocol_and_duplicates(self):
        validate_group(small_group(), 'rust-candle', {})
        for key, value in (('schema_version', True), ('schema_version', 2), ('runtime', 'python'),
                           ('device', 'metal'), ('precision', 'fp16'), ('warmups', 0), ('trials', 0),
                           ('source_sha256', {'bad': 'hash'}), ('diagnostic_profiling', 'true'),
                           ('diagnostic_profiling', True), ('index_dtype', 'i64')):
            group = small_group(); group[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_group(group, 'rust-candle', {})
        for group in (small_group()[:2], [small_group()[0]]*3):
            with self.assertRaises(ValueError):
                validate_group(group, 'rust-candle', {})

    def test_source_inventory_guard_precedes_reference_loading(self):
        valid = {p: LAYOUT_SOURCES.get(p, 'f'*64) for p in NATIVE_SOURCES}
        wrong_layout = dict(valid); wrong_layout[next(iter(LAYOUT_SOURCES))] = '0'*64
        invalid_hash = dict(valid); invalid_hash['Cargo.lock'] = 'invalid'
        with patch('qualify_diagnostic_codec.frozen_references', side_effect=AssertionError('unexpected load')):
            for sources in ({}, LAYOUT_SOURCES, wrong_layout, invalid_hash):
                with self.subTest(sources=sources), self.assertRaises(ValueError):
                    qualify([b'{}']*3, sources)
            with self.assertRaises(ValueError):
                qualify([], valid)

    def test_reference_archive_and_member_hash_corruption_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive, evidence = Path(tmp) / 'reference.tar.xz', Path(tmp) / 'evidence.json'
            name = 'phase2-linux-diagnostic-v2/candidate/codec-python-1.json'
            raw = b'{}'
            with tarfile.open(archive, 'w:xz') as bundle:
                member = tarfile.TarInfo(name); member.size = len(raw)
                bundle.addfile(member, io.BytesIO(raw))
            metadata = {'archive': {'path': str(archive), 'bytes': archive.stat().st_size,
                                   'sha256': '0'*64, 'members': [dict(path=name, bytes=2, sha256='0'*64)]}}
            with patch('qualify_diagnostic_codec.EVIDENCE', evidence):
                evidence.write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, 'archive mismatch'):
                    frozen_references()
                metadata['archive']['sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
                evidence.write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, 'report mismatch'):
                    frozen_references()

    def test_real_archived_outputs_negative_cases_and_acceptance_separation(self):
        child = subprocess.run([sys.executable, '-c', 'import json; from test_qualify_diagnostic_codec '
                                'import archived_cases; print(json.dumps(archived_cases()))'],
                               cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=60)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(json.loads(child.stdout), {'qualified_scopes': 12, 'invalid_report_cases_rejected': 9,
                                                  'profiled_acceptance_rejected': True, 'cli_verified': True})


if __name__ == '__main__':
    unittest.main()
