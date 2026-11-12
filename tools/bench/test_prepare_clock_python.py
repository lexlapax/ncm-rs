"""Synthetic metadata/ELF markers only; no interpreter, package or GPU execution."""
from copy import deepcopy
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

import prepare_clock_python as preparation


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class PythonPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def fixture(self):
        self.root = self.base / str(len(list(self.base.iterdir())))
        self.root.mkdir()
        sources = {}
        for key, name in preparation.SOURCES.items():
            path = self.root / 'tools/reference' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# synthetic ' + key)
            sources[key] = sha(path.read_bytes())
        (self.root / 'tools/reference/requirements-linux.lock').write_text('torch==2.15.0\nDemo==1.0\n')
        self.report = dict(runtime='pytorch-pinned-eager-snapshot-fix', torch='2.15.0+cu132',
                           upstream_commit=preparation.UPSTREAM, reference_versions={'torch':'2.15.0','Demo':'1.0'},
                           source_sha256=sources, manifest_sha256='a'*64, workload_sha256='b'*64,
                           precision='fp32', index_dtype='i64', warmups=5, trials=30)
        self.reports = [deepcopy(self.report) for _ in range(6)]
        self.archive()
        self.venv = self.root / '.reference/venv'
        (self.venv / 'bin').mkdir(parents=True)
        image = self.root / 'python-image'
        image.write_bytes(b'\x7fELFnot-an-executable-test-marker')
        image.chmod(0o700)
        (self.venv / 'bin/python3').symlink_to(image)
        (self.venv / 'bin/python').symlink_to('python3')
        (self.venv / 'pyvenv.cfg').write_text(f'home = {self.root}\ninclude-system-site-packages = false\n'
                                            f'version = 3.14.9\nexecutable = {image}\ncommand = not-for-output\n')
        self.site = self.venv / 'lib/python3.14/site-packages'
        self.site.mkdir(parents=True)
        for name, version in (('torch','2.15.0'),('Demo','1.0'),('pip','26.2.2')):
            directory = self.site / f'{name}-{version}.dist-info'
            directory.mkdir()
            (directory/'METADATA').write_text(f'Name: {name}\nVersion: {version}\n\nnot-for-output\n')
        (self.site/'torch').mkdir()
        (self.site/'torch/version.py').write_text("__version__ = '2.15.0+cu132'\nraise RuntimeError('must not import')\n")
        (self.site/'hook.pth').write_text("import must_never_execute\n")
        (self.site/'helper.py').write_text("raise RuntimeError('must not execute')\n")
        self.review = {name: sha((self.site/name).read_bytes()) for name in ('hook.pth','helper.py')}

    def archive(self):
        prefix = 'phase2-linux-diagnostic-v2/'
        records = {prefix + f'python/codec-python-{i}.json': report for i, report in enumerate(self.reports,1)}
        records[prefix+'environment.json'] = {'venv': {'python': 'Python 3.14.9'}, 'candidate_status': 'not-for-output'}
        path = self.root/'archive.tar.xz'
        pins = []
        with tarfile.open(path,'w:xz') as archive:
            for name, record in records.items():
                raw = json.dumps(record).encode()
                member = tarfile.TarInfo(name); member.size = len(raw)
                archive.addfile(member,io.BytesIO(raw))
                pins.append(dict(path=name,bytes=len(raw),sha256=sha(raw)))
        self.evidence = dict(schema_version=1, measured_commit=preparation.MANIFEST_HEAD,
                             production_candidate=preparation.PRODUCTION_HEAD,
                             archive=dict(path='archive.tar.xz',bytes=path.stat().st_size,sha256=sha(path.read_bytes()),members=pins))
        self.save_evidence()

    def save_evidence(self):
        path = self.root / preparation.EVIDENCE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.evidence))

    def prepare(self):
        return preparation.prepare(self.root, reviewed_startup=self.review)

    def test_separates_historical_sources_from_current_snapshots_without_execution(self):
        self.fixture()
        before = {name for name in sys.modules if name == 'torch' or name.startswith('torch.')}
        with patch.object(subprocess, 'Popen', side_effect=AssertionError('no probes')):
            result = self.prepare()
        self.assertEqual(before, {name for name in sys.modules if name == 'torch' or name.startswith('torch.')})
        self.assertEqual(result['distribution_versions']['torch'], '2.15.0')
        self.assertEqual(result['torch_source_build_tag'], '2.15.0+cu132')
        self.assertEqual(result['extra_distributions'], {'pip':'26.2.2'})
        self.assertEqual(len(result['historical']['selected_members']), 7)
        self.assertEqual(len(result['historical_source_identities']), 3)
        self.assertEqual(len(result['interpreter_links']), 2)
        self.assertEqual(len(result['current_only_file_identities']), 9)
        for key in ('historical_interpreter_bytes_verified','historical_library_bytes_verified','runtime_version_probed',
                    'torch_imported','startup_import_closure_verified','library_resolution_verified',
                    'current_upstream_worktree_verified','launch_ready'):
            self.assertFalse(result[key])
        self.assertNotIn('not-for-output', json.dumps(result))
        self.assertTrue(all(row['expected_digest_verified'] for row in result['historical_source_identities']))
        self.assertFalse(result['interpreter_image']['expected_digest_verified'])

    def test_historical_tampering_and_current_source_drift_fail(self):
        for mode in ('archive','member','disagreement','upstream','source','schema'):
            with self.subTest(mode=mode):
                self.fixture()
                if mode=='archive': self.evidence['archive']['sha256']='0'*64; self.save_evidence()
                if mode=='schema': self.evidence['schema_version']=True; self.save_evidence()
                if mode=='member': self.evidence['archive']['members'][0]['sha256']='0'*64; self.save_evidence()
                if mode=='disagreement': self.reports[-1]['torch']='different'; self.archive()
                if mode=='upstream':
                    for report in self.reports: report['upstream_commit']='0'*40
                    self.archive()
                if mode=='source': (self.root/'tools/reference/codec_trace.py').write_text('changed')
                with self.assertRaises(ValueError): self.prepare()

    def test_configuration_and_distribution_guardrails(self):
        for mode in ('version','system_site','duplicate_config','metadata_version','duplicate_name','duplicate_header','lock'):
            with self.subTest(mode=mode):
                self.fixture()
                cfg=self.venv/'pyvenv.cfg'
                if mode=='version': cfg.write_text(cfg.read_text().replace('3.14.9','3.14.8'))
                if mode=='system_site': cfg.write_text(cfg.read_text().replace('false','true'))
                if mode=='duplicate_config': cfg.write_text(cfg.read_text()+'version = 3.14.9\n')
                meta=self.site/'torch-2.15.0.dist-info/METADATA'
                if mode=='metadata_version': meta.write_text('Name: torch\nVersion: 2.15.0+cu132\n')
                if mode=='duplicate_name': (self.site/'Demo-1.0.dist-info/METADATA').write_text('Name: TORCH\nVersion: 2.15.0\n')
                if mode=='duplicate_header': meta.write_text('Name: torch\nName: torch\nVersion: 2.15.0\n')
                if mode=='lock': (self.root/'tools/reference/requirements-linux.lock').write_text('torch==2.15.0\n')
                with self.assertRaises(ValueError): self.prepare()

    def test_unknown_startup_inputs_and_symlinked_metadata_need_review(self):
        for mode in ('new_hook','changed_hook','sitecustomize','package_customization','outside','metadata_link'):
            with self.subTest(mode=mode):
                self.fixture()
                if mode=='new_hook': (self.site/'new.pth').write_text('import do_not_execute')
                if mode=='changed_hook': (self.site/'hook.pth').write_text('changed')
                if mode=='sitecustomize': (self.site/'sitecustomize.py').write_text('raise RuntimeError()')
                if mode=='package_customization': (self.site/'usercustomize').mkdir()
                if mode=='outside': self.review['../escape.py']='0'*64
                if mode=='metadata_link':
                    meta=self.site/'Demo-1.0.dist-info/METADATA'; saved=self.root/'metadata'
                    meta.rename(saved); meta.symlink_to(saved)
                with self.assertRaises(ValueError): self.prepare()

    def test_torch_build_declaration_is_literal_and_separate_from_metadata(self):
        for source in ("__version__ = 'wrong'", "__version__ = compute()", "__version__='2.15.0+cu132'\n__version__='2.15.0+cu132'"):
            with self.subTest(source=source):
                self.fixture(); (self.site/'torch/version.py').write_text(source)
                with self.assertRaises(ValueError): self.prepare()

    def test_inventory_or_link_changes_during_preparation_fail(self):
        for mode in ('hook','distribution','link','image_replaced'):
            with self.subTest(mode=mode):
                self.fixture()
                actual = preparation.file_identity
                changed = False
                def snapshot(path, *args, **kwargs):
                    nonlocal changed
                    value = actual(path,*args,**kwargs)
                    if Path(path).name=='METADATA' and not changed:
                        changed=True
                        if mode=='hook': (self.site/'new.pth').write_text('unreviewed')
                        if mode=='distribution': (self.site/'new-1.dist-info').mkdir()
                        if mode=='link':
                            link=self.venv/'bin/python'; link.unlink(); link.symlink_to(self.root/'python-image')
                        if mode=='image_replaced':
                            image=self.root/'python-image'; saved=self.root/'old-image'
                            image.rename(saved); image.write_bytes(saved.read_bytes()); image.chmod(0o700)
                    return value
                with patch.object(preparation,'file_identity',side_effect=snapshot):
                    with self.assertRaises(ValueError): self.prepare()
                self.assertTrue(changed)

    def test_explicit_metadata_and_archive_bounds(self):
        for mode in ('metadata','archive','startup_file','startup_count','distributions'):
            with self.subTest(mode=mode):
                self.fixture()
                if mode=='metadata': (self.site/'torch-2.15.0.dist-info/METADATA').write_bytes(b'x'*(preparation.TEXT_BYTES+1))
                if mode=='archive': self.evidence['archive']['bytes']=preparation.ARCHIVE_BYTES+1; self.save_evidence()
                if mode=='startup_file': (self.site/'hook.pth').write_bytes(b'x'*(preparation.TEXT_BYTES+1))
                if mode=='startup_count': self.review={f'{i}.pth':'0'*64 for i in range(33)}
                if mode=='distributions':
                    for i in range(preparation.MAX_DISTRIBUTIONS): (self.site/f'extra-{i}.dist-info').mkdir()
                with self.assertRaises(ValueError): self.prepare()


if __name__ == '__main__':
    unittest.main()
