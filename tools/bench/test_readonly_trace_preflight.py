"""Offline identity/environment tests; fake files are never executed."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import readonly_trace_preflight as preflight
from readonly_trace_sequence import ORDER


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'collection'
        self.base = self.root / '.reference/phase2-optimization-baseline'
        self.base.mkdir(parents=True)
        self.binaries = {'A': self.root / '.reference/phase2-optimization-baseline-bin/codec_bench',
                         'C': self.root / 'target/release/examples/codec_bench'}
        reports, binary_hashes = {}, {}
        for key, tree in (('A', self.base), ('C', self.root)):
            self.put(self.binaries[key], ('not executable code ' + key).encode(), executable=True)
            binary_hashes[key] = sha(self.binaries[key].read_bytes())
            sources = {}
            for name in preflight.NATIVE_SOURCES:
                data = (key + name).encode()
                self.put(tree / name, data)
                sources[name] = sha(data)
            self.put(tree / preflight.FIXTURE, b'fixture')
            reports[key] = dict(source_sha256=sources, workload_sha256=sha(b'fixture'),
                                manifest_sha256=sha(b'manifest'))
        self.put(self.root / preflight.ARTIFACT / 'manifest.json', b'manifest')
        payload = []
        for name in preflight.PAYLOAD_NAMES:
            relative = preflight.ARTIFACT / name
            self.put(self.root / relative, name.encode())
            payload.append(dict(path=str(relative), bytes=len(name), sha256=sha(name.encode())))
        self.injection = self.root / '.reference/phase2-cupti/libcupti_trace_injection.so'
        self.put(self.injection, b'injection')
        self.sample = self.root / 'sample.cpp'
        self.put(self.sample, b'sample')
        self.profile = self.root / 'profile.json'
        self.put(self.profile, json.dumps({'profiling': {
            'original_injection_library_sha256': sha(b'injection'),
            'sample_sha256': {str(self.sample): sha(b'sample')}}}).encode())
        self.taskset = self.root / 'taskset'
        self.put(self.taskset, b'never execute', executable=True)
        frozen = dict(reports=reports, payload=payload, binaries=binary_hashes, archive_sha256='a' * 64)
        for mock in (patch.object(preflight, 'frozen_inputs', return_value=frozen),
                     patch.object(preflight, 'PROFILE_EVIDENCE', self.profile),
                     patch.object(preflight, 'validate_observer')):
            started = mock.start()
            if getattr(mock, 'attribute', None) == 'validate_observer':
                self.topology = started
            self.addCleanup(mock.stop)

    def put(self, path, raw, executable=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        if executable:
            path.chmod(0o700)

    def prepare(self, **kwargs):
        return preflight.prepare(self.root, self.output, self.root, taskset=self.taskset,
                                 inherited=kwargs.get('inherited', {}))

    def test_fixed_commands_and_injection_only_in_traced_children(self):
        with patch('subprocess.Popen', side_effect=AssertionError('preflight executed a child')):
            result = self.prepare(inherited={'PRIVATE_TEST_KEY': 'notpropagated', 'CUDA_PATH': '/ignored'})
        self.topology.assert_called_once_with(2, 0)
        self.assertEqual([j['condition'] for j in result['jobs']], list(ORDER))
        self.assertFalse(self.output.exists())
        self.assertFalse(result['launch_ready'])
        self.assertFalse(result['library_resolution_verified'])
        self.assertEqual(result['observer_env'], {'PATH': os.defpath, 'HOME': str(self.root),
                                               'LANG': 'C', 'LC_ALL': 'C', 'PYTHONNOUSERSITE': '1'})
        self.assertNotIn('notpropagated', json.dumps(result))
        for ordinal, job in enumerate(result['jobs'], 1):
            expected = dict(result['observer_env'])
            if job['condition'].endswith('T'):
                expected['CUDA_INJECTION64_PATH'] = str(self.injection)
            self.assertEqual(job['env'], expected)
            self.assertEqual(job['argv'][:4], [str(self.taskset), '-c', '2',
                                              str(self.binaries[job['condition'][0]])])
            self.assertEqual(job['argv'][-1], str(self.output / f'{ordinal:02d}-{job["condition"]}.report.json'))
            self.assertNotIn('--profile-block4', job['argv'])
        result['jobs'][2]['env']['LD_PRELOAD'] = 'test'
        self.assertNotIn('LD_PRELOAD', result['observer_env'])
        self.assertNotIn('LD_PRELOAD', result['jobs'][3]['env'])

    def test_contaminated_environment_rejected_before_file_checks(self):
        for name in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'LD_AUDIT', 'CUDA_INJECTION64_PATH',
                     'CUDA_VISIBLE_DEVICES', 'CUDA_LAUNCH_BLOCKING', 'NVTX_INJECTION64_PATH',
                     'CUPTI_TEST', 'NCM_TEST', 'CUBLAS_WORKSPACE_CONFIG'):
            with self.subTest(name=name), patch.object(preflight, 'frozen_inputs') as frozen:
                with self.assertRaisesRegex(ValueError, 'inherited native-control'):
                    self.prepare(inherited={name: 'present'})
                frozen.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_changed_frozen_files_fail_without_outputs(self):
        paths = [self.binaries['A'], self.binaries['C'], self.root / 'Cargo.lock',
                 self.base / preflight.FIXTURE, self.root / preflight.ARTIFACT / 'manifest.json',
                 self.root / preflight.ARTIFACT / 'codec.safetensors', self.injection, self.sample]
        for path in paths:
            raw = path.read_bytes()
            path.write_bytes(b'changed')
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, '(digest|size) mismatch'):
                    self.prepare()
                self.assertFalse(self.output.exists())
            path.write_bytes(raw)

    def test_existing_output_missing_executable_and_topology_guards(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.prepare()
        self.output.rmdir()
        self.binaries['A'].chmod(0o600)
        with self.assertRaisesRegex(ValueError, 'not executable'):
            self.prepare()
        self.binaries['A'].chmod(0o700)
        self.topology.side_effect = ValueError('bad topology')
        with self.assertRaisesRegex(ValueError, 'bad topology'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_file_identity_rejects_symlink_fifo_size_digest_and_visible_replacement(self):
        target = self.root / 'identity-target'
        target.write_bytes(b'abc')
        with self.assertRaisesRegex(ValueError, 'size mismatch'):
            preflight.file_identity(target, sha(b'abc'), 4)
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            preflight.file_identity(target, sha(b'def'))
        link = self.root / 'link'
        link.symlink_to(target)
        with self.assertRaises(OSError):
            preflight.file_identity(link)
        fifo = self.root / 'fifo'
        os.mkfifo(fifo)
        with self.assertRaisesRegex(ValueError, 'not a regular'):
            preflight.file_identity(fifo)
        actual = os.stat

        def replace(path, **kwargs):
            target.unlink()
            target.write_bytes(b'changed')
            return actual(path, **kwargs)

        with patch.object(preflight.os, 'stat', side_effect=replace):
            with self.assertRaisesRegex(ValueError, 'changed during'):
                preflight.file_identity(target)


class FrozenInputTests(unittest.TestCase):
    def test_real_archive_pins_without_model_loading(self):
        command = ('import json; from readonly_trace_preflight import frozen_inputs; '
                   'f=frozen_inputs(); print(json.dumps({"binaries":f["binaries"], '
                   '"payload_files":len(f["payload"]), '
                   '"sources":[len(r["source_sha256"]) for r in f["reports"].values()]}))')
        result = subprocess.run([sys.executable, '-c', command], cwd=Path(__file__).parent,
                                env={'PATH': os.defpath, 'PYTHONNOUSERSITE': '1'},
                                check=True, capture_output=True, text=True)
        data = json.loads(result.stdout)
        self.assertEqual(data['payload_files'], 6)
        self.assertEqual(data['sources'], [12, 12])
        self.assertEqual(data['binaries'], {
            'A': '26437962fc06d0f43280e2b2b5608cacfa6d5931ab14f1f9d89c4a4f6cfc35a9',
            'C': '034b609cb16af0a8d7c95eb11ffb056e49d0f0cdb2e98d775e716673adbe41ac'})


if __name__ == '__main__':
    unittest.main()
