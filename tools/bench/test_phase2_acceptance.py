import copy
import json
from pathlib import Path
import unittest
import tarfile
import tempfile
from unittest.mock import patch
from summarize_codec_load_control import summarize as codec_load_control
from audit_phase2_performance import quality, bind_control_sources, bind_control_environment, audit
from run_codec_lifecycle import validate

ROOT=Path(__file__).resolve().parents[2]


class AcceptanceTests(unittest.TestCase):
    def test_quality_is_not_an_empty_or_relaxed_gate(self):
        raw=json.loads((ROOT/'docs/results/2026-11-10-phase2-cuda-remediation/raw/qualification.json').read_bytes())
        quality(raw)
        for mutate in [lambda q:q.update(traces=[]),lambda q:q.update(all_passed=False),
                       lambda q:q.update(left_trace_sha256='wrong'),lambda q:q['policy']['primitive'].update(atol=1),
                       lambda q:q['traces'][0].update(mismatches=1),lambda q:q.update(selections={})]:
            q=copy.deepcopy(raw);mutate(q)
            with self.assertRaises(ValueError):quality(q)

    def test_control_binding_includes_python_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);controls=root/'controls';matrix=root/'matrix'
            roots={name:matrix/name for name in ('previous','candidate')}
            for name in ('previous','candidate','python'):
                (controls/name).mkdir(parents=True)
                target=roots['candidate' if name=='python' else name];target.mkdir(parents=True,exist_ok=True)
                filename=f"tokenizer-bert-{'python' if name=='python' else 'rust'}-1.json"
                raw=json.dumps({'source_sha256':{'source':'a'*64}})
                (controls/name/filename).write_text(raw);(target/filename).write_text(raw)
            bind_control_sources(controls,roots,'bert')
            (controls/'python/tokenizer-bert-python-1.json').write_text(json.dumps({'source_sha256':{'source':'b'*64}}))
            with self.assertRaises(ValueError):bind_control_sources(controls,roots,'bert')

    def test_codec_load_control_validates_full_reports(self):
        archive=ROOT/'docs/results/2026-11-10-phase2-shared-loading/raw/linux-token-buffer-v1.tar.gz'
        with tempfile.TemporaryDirectory() as tmp, patch('summarize_codec_load_control.interval', return_value={}):
            root=Path(tmp)
            with tarfile.open(archive) as bundle: bundle.extractall(root, filter='data')
            result=codec_load_control(root,'cuda')
            self.assertEqual(result['measurements'][0]['scope'],'checked_load')
            path=root/'candidate/codec-rust-1.json';original=path.read_bytes()
            for mutate in [lambda r:r.update(load_trials_ms=[1.]), lambda r:r.update(runtime='wrong'),
                           lambda r:r.update(diagnostic_profiling=True), lambda r:r['measurements'][1]['first_logits'].__setitem__(0,1e10)]:
                raw=json.loads(original);mutate(raw);path.write_text(json.dumps(raw))
                with self.assertRaises(ValueError):codec_load_control(root,'cuda')
                path.write_bytes(original)
            path.unlink()
            with self.assertRaises(ValueError):codec_load_control(root,'cuda')

    def test_codec_execution_controls_validate_and_keep_all_scopes(self):
        archive=ROOT/'docs/results/2026-11-10-phase2-shared-loading/raw/linux-token-buffer-v1.tar.gz'
        with tempfile.TemporaryDirectory() as tmp, patch('summarize_codec_load_control.interval', return_value={}):
            root=Path(tmp)
            with tarfile.open(archive) as bundle: bundle.extractall(root, filter='data')
            original=codec_load_control(root,'cuda')
            expanded=codec_load_control(root,'cuda',include_execution=True)
            self.assertEqual(expanded['measurements'][-1],original['measurements'][0])
            self.assertEqual(len(expanded['measurements']),13)
            self.assertEqual({(r['batch'],r['scope']) for r in expanded['measurements'][:-1]},
                             {(b,s) for b in (1,2,4) for s in ('encode','decode','reconstruct','resident_decoder_block')})
            for row in expanded['measurements']:
                for timing in row['timings'].values():
                    self.assertEqual(timing['n'],90)
                    self.assertEqual(len(timing['process_medians_ms']),3)
            path=root/'candidate/codec-rust-3.json'
            report=json.loads(path.read_bytes());report['measurements'][0]['trials_ms'].pop()
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):codec_load_control(root,'cuda',include_execution=True)

    def test_control_environment_binds_both_binaries_and_python(self):
        environment={'binary_sha256':{n:{'codec_bench':'a'*64,'asset_bench':'b'*64} for n in ('previous','candidate')},
                     'venv':{'python':'3.14','packages':'pinned'},'tokenizers-venv':{'python':'3.14','packages':'also pinned'},
                     'harness_sha256':'c'*64,'cpu_affinity':2}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'environment.json'
            path.write_text(json.dumps(environment))
            for profile in ('codec','bert'):bind_control_environment(root,environment,profile,'cuda')
            for mutate in [lambda e:e['binary_sha256']['previous'].update(codec_bench='d'*64),
                           lambda e:e['binary_sha256']['candidate'].update(codec_bench='d'*64),
                           lambda e:e['venv'].update(packages='changed'),
                           lambda e:e.update(harness_sha256='d'*64),lambda e:e.update(cpu_affinity=3)]:
                bad=copy.deepcopy(environment);mutate(bad);path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):bind_control_environment(root,environment,'codec','cuda')

    def test_execution_controls_route_into_all_codec_cells_without_widening_limits(self):
        base=ROOT/'docs/results/2026-11-11-phase2-low-risk/raw'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with tarfile.open(base/'reconstruction-threeway-v1.tar.gz') as bundle:bundle.extractall(root,filter='data')
            matrix=root/'phase2-optimization-reuse-threeway-v1'
            cfg={'device':'cuda','previous':str(matrix/'previous'),'candidate':str(matrix/'candidate'),
                 'qualification':str(base/'qualification.json'),'lifecycle':str(base/'lifecycle.json'),
                 'controls':{'codec':str(matrix)},'load_controls':['codec'],'codec_execution_controls':True,
                 'control_matrix_environment':str(matrix/'environment.json')}
            # Intentionally failing upper bound: the audit must not substitute
            # favorable original points or ignore the new execution controls.
            interval={'median_ratio':[1.0,1.06],'p95_ratio':[1.0,1.0]}
            with patch('summarize_codec_load_control.interval',return_value=interval):
                result=audit(cfg)
            self.assertEqual(len(result['cells']),78)
            cells=[r for r in result['cells'] if r['cell'].startswith('codec/') and 'median_ratio' in r]
            self.assertEqual(len(cells),26)
            self.assertTrue(all('controlled_followup' in r and not r['passed'] for r in cells))
            self.assertEqual(result['limits'],{'median':1.05,'p95':1.10,'comparable_host_memory':1.05})
            cfg.pop('control_matrix_environment')
            with self.assertRaises(ValueError):audit(cfg)

    def test_lifecycle_requires_every_phase_and_identity(self):
        rows=[{'schema_version':1,'phase':p,'internal_ms':1.,'output_ids_sha256':'a'*64 if p=='execute' else None}
              for p in ('context','load','execute','drop')]
        self.assertEqual(validate(rows),'a'*64)
        with self.assertRaises(ValueError):validate(rows[:-1])
        with self.assertRaises(ValueError):validate(rows[::-1])
        bad=copy.deepcopy(rows);bad[2]['output_ids_sha256']=None
        with self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(rows);bad[0]['internal_ms']=float('nan')
        with self.assertRaises(ValueError):validate(bad)


if __name__=='__main__':unittest.main()
