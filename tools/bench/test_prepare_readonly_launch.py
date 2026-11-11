"""Preparation contract tests: metadata probes mocked, no GPU/model commands run."""
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import unittest
from unittest.mock import patch

import prepare_readonly_launch as launch
import readonly_trace_preflight as preflight
import test_readonly_trace_preflight as fixtures

UUID = 'GPU-12345678-1234-1234-1234-123456789abc'
PCI = '00000000:01:00.0'
HEAD = 'a' * 40


class LaunchPreparationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PreflightTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.root = self.fixture.root
        self.collection = self.root / 'collection'
        self.evidence = self.root / 'preparation'
        for directory in launch.TOOL_DIRS:
            self.fixture.put(self.root / directory / 'fixture.py', b'# source inventory fixture\n')
        self.python, self.smi = self.root / 'python', self.root / 'nvidia-smi'
        for path in (self.python, self.smi):
            self.fixture.put(path, b'not executable code', executable=True)
        self.zone_root = self.root / 'zoneinfo'
        # Minimal fixed-offset UTC TZif v1, parsed by the actual ZoneInfo reader.
        data = b'TZif\0' + bytes(15) + struct.pack('>6l', 0, 0, 0, 0, 1, 4) + struct.pack('>lBB', 0, 0, 0) + b'UTC\0'
        self.fixture.put(self.zone_root / 'Etc/UTC', data)
        self.localtime = self.root / 'localtime'
        self.localtime.symlink_to(self.zone_root / 'Etc/UTC')
        self.zone = launch.timezone_identity(self.localtime, self.zone_root)
        self.library_root = self.root / 'libraries-root'
        self.library_root.mkdir()
        self.git, self.loader = self.root / 'git', self.root / 'loader'
        self.library = self.library_root / 'libfixture.so'
        for path in (self.git, self.loader, self.library):
            self.fixture.put(path, b'metadata fixture only', executable=True)
        self.probe_effect = None
        self.events = []

        def worktrees(root, head, output):
            self.events.append('worktrees')
            return {'states': {'candidate': {'head': head}}, 'worktree_state_verified': True,
                    'git_identity': preflight.file_identity(self.git)}

        def libraries(prepared, output):
            self.events.append('libraries')
            ids = {value['path']: value for value in prepared['file_identities']}
            jobs = prepared['jobs']
            targets = {name: ids[path] for name, path in (
                ('accepted', jobs[0]['argv'][3]), ('candidate', jobs[1]['argv'][3]),
                ('injection', jobs[2]['env']['CUDA_INJECTION64_PATH']), ('taskset', jobs[0]['argv'][0]))}
            return {'targets': targets, 'libraries': {str(self.library): preflight.file_identity(self.library)},
                    'loader_identity': preflight.file_identity(self.loader), 'listings': {},
                    'static_loader_resolution_verified': True}

        def probe(argv, cwd, env, directory, name):
            self.events.append('gpu_identity')
            self.assertEqual(argv, [str(self.smi), '--query-gpu=uuid,pci.bus_id,name', '--format=csv,noheader'])
            self.assertNotIn('CUDA_INJECTION64_PATH', env)
            if self.probe_effect:
                self.probe_effect()
            return f'{UUID}, {PCI}, fixture device\n', {'status': 'complete', 'argv': argv}

        for mock in (patch.object(launch, 'verify_worktrees', side_effect=worktrees),
                     patch.object(launch, 'inspect_libraries', side_effect=libraries),
                     patch.object(launch, 'metadata_probe', side_effect=probe),
                     patch.object(launch, 'timezone_identity', return_value=self.zone),
                     patch.object(launch, 'DEFAULT_LIBRARY_ROOTS', (self.library_root,))):
            mock.start()
            self.addCleanup(mock.stop)

    def prepare(self, **kwargs):
        return launch.prepare_contract(self.root, self.collection, self.evidence,
                expected_candidate_head=HEAD, expected_gpu_pci=PCI, expected_gpu_name='fixture device',
                home=self.root, inherited={}, python=self.python, nvidia_smi=self.smi,
                taskset=self.fixture.taskset, **kwargs)

    def test_private_contract_binds_sources_commands_timezone_without_collection(self):
        with patch('subprocess.Popen', side_effect=AssertionError('unexpected process launch')):
            result = self.prepare()
        self.assertEqual(self.events, ['worktrees', 'libraries', 'gpu_identity'])
        self.assertFalse(self.collection.exists())
        self.assertFalse(result['launch_ready'])
        self.assertFalse(result['collection_started'])
        self.assertEqual(stat.S_IMODE(self.evidence.stat().st_mode), 0o700)
        raw = Path(result['manifest']).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), result['sha256'])
        contract = json.loads(raw)
        self.assertEqual(contract['gpu']['uuid'], UUID)
        self.assertEqual(contract['timezone']['zone'], 'Etc/UTC')
        self.assertEqual(len(contract['prepared']['jobs']), 12)
        self.assertEqual(len(contract['tool_identities']), 2)
        cpu = contract['telemetry_commands']['cpu']
        self.assertEqual(cpu[:5], [str(self.fixture.taskset), '-c', '0,2', str(self.python), '-S'])
        self.assertEqual(contract['telemetry_commands']['gpu'][1:3], ['-i', UUID])
        for item in contract['prepared']['jobs']:
            self.assertEqual(Path(item['report_path']).parent, self.collection / 'sequence')
        self.assertNotIn('site-packages', raw.decode())
        with self.assertRaises(ValueError):
            self.prepare()

    def test_reserved_parent_is_exclusive_and_does_not_create_directories(self):
        with patch('subprocess.Popen', side_effect=AssertionError('no child allowed')):
            preflight.prepare(self.root, self.collection / 'sequence', self.root, inherited={},
                              taskset=self.fixture.taskset, reserved_parent=self.collection)
        self.assertFalse(self.collection.exists())
        for parent in (self.root / 'wrong', self.root / 'missing' / 'nested'):
            with self.assertRaises(ValueError):
                preflight.prepare(self.root, self.collection / 'sequence', self.root, inherited={},
                                  taskset=self.fixture.taskset, reserved_parent=parent)
        self.collection.mkdir()
        with self.assertRaises(ValueError):
            preflight.prepare(self.root, self.collection / 'sequence', self.root, inherited={},
                              taskset=self.fixture.taskset, reserved_parent=self.collection)

    def test_device_parser_rejects_multi_device_or_mismatched_declared_hardware(self):
        good = f'{UUID}, {PCI}, fixture device\n'
        self.assertEqual(launch.parse_device(good, expected_pci=PCI, expected_name='fixture device')['uuid'], UUID)
        for raw in ('', good+good, good.replace(UUID, 'GPU-invalid'), good.replace(PCI, '00000000:02:00.0'),
                    good.replace('fixture device', 'other device'), 'unexpected error\n'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                launch.parse_device(raw, expected_pci=PCI, expected_name='fixture device')

    def test_changed_tools_or_timezone_during_probes_leave_no_manifest(self):
        self.probe_effect = lambda: (self.root / 'tools/bench/fixture.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, '(size|digest) mismatch'):
            self.prepare()
        self.assertFalse((self.evidence / 'manifest.json').exists())
        self.assertTrue(self.evidence.is_dir())
        self.assertFalse(self.collection.exists())

    def test_structural_contract_tampering_and_stronger_claims_rejected(self):
        contract = json.loads(Path(self.prepare()['manifest']).read_bytes())
        for mutate in (lambda c: c.update(launch_ready=True), lambda c: c.update(availability_confirmed=True),
                       lambda c: c.update(collection_started=True),
                       lambda c: c['prepared']['jobs'][0]['env'].update(CUDA_INJECTION64_PATH='wrong'),
                       lambda c: c['prepared']['jobs'][0]['argv'].__setitem__(2, '3'),
                       lambda c: c['limits'].update(telemetry_seconds=60),
                       lambda c: c['telemetry_commands']['cpu'].remove('-S'),
                       lambda c: c['prepared']['file_identities'].pop(),
                       lambda c: c['worktrees']['states']['candidate'].update(head='b'*40),
                       lambda c: c['worktrees'].update(worktree_state_verified=False),
                       lambda c: c['libraries'].update(static_loader_resolution_verified=False)):
            changed = copy.deepcopy(contract)
            mutate(changed)
            with self.subTest(mutate=mutate), self.assertRaises((ValueError, KeyError)):
                launch.validate_contract(changed)

    def test_library_mutation_during_gpu_probe_fails_recheck(self):
        self.probe_effect = lambda: self.library.write_bytes(b'changed after loader listing')
        with self.assertRaisesRegex(ValueError, '(size|digest) mismatch'):
            self.prepare()
        self.assertFalse((self.evidence / 'manifest.json').exists())

    def test_timezone_link_mutation_during_probes_fails_recheck(self):
        other = self.zone_root / 'Other'
        other.write_bytes((self.zone_root / 'Etc/UTC').read_bytes())
        def change():
            self.localtime.unlink()
            self.localtime.symlink_to(other)
        self.probe_effect = change
        with self.assertRaisesRegex(ValueError, 'timezone link changed'):
            self.prepare()
        self.assertFalse((self.evidence / 'manifest.json').exists())

    def test_output_alias_cannot_create_collection_as_preparation(self):
        (self.root / 'other').mkdir()
        self.evidence = self.root / 'other' / '..' / 'collection'
        with self.assertRaisesRegex(ValueError, 'directories must be separate'):
            self.prepare()
        self.assertFalse(self.collection.exists())
        self.assertEqual(self.events, [])

    def test_failed_worktree_guard_never_queries_device(self):
        with patch.object(launch, 'verify_worktrees', side_effect=ValueError('dirty worktree')), \
                patch.object(launch, 'metadata_probe') as probe, self.assertRaises(ValueError):
            self.prepare()
        probe.assert_not_called()
        self.assertFalse((self.evidence / 'manifest.json').exists())
        self.assertFalse(self.collection.exists())

    def test_cli_has_no_collection_mode_and_requires_no_site(self):
        script = Path(launch.__file__).resolve()
        env = {'PATH': os.defpath, 'HOME': str(self.root), 'PYTHONNOUSERSITE': '1'}
        run = subprocess.run([sys.executable, '-S', str(script), '--help'], env=env,
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('NEVER collect', run.stdout)
        run = subprocess.run([sys.executable, str(script), str(self.root), str(self.collection), str(self.evidence),
                              '--candidate-head', HEAD, '--gpu-pci', PCI, '--gpu-name', 'fixture device'],
                             env=env, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('invoke preparation with python -S', run.stderr)
        self.assertFalse(self.evidence.exists())


class TimezonePreparationTests(unittest.TestCase):
    def test_zoneinfo_symlink_is_parsed_without_sampling(self):
        # Read-only metadata; no sampler, model or GPU command. Unsupported copied
        # localtime configurations must not be guessed into an IANA zone.
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            zones = root / 'zoneinfo'
            (zones / 'Etc').mkdir(parents=True)
            raw = b'TZif\0'+bytes(15)+struct.pack('>6l',0,0,0,0,1,4)+struct.pack('>lBB',0,0,0)+b'UTC\0'
            zone = zones / 'Etc/UTC'
            zone.write_bytes(raw)
            link = root / 'localtime'
            link.symlink_to(zone)
            result = launch.timezone_identity(link, zones)
            self.assertEqual(result['zone'], 'Etc/UTC')
            self.assertTrue(result['observed_local_with_offset'].endswith('+00:00'))
            link.unlink()
            link.write_bytes(raw)
            with self.assertRaises(ValueError):
                launch.timezone_identity(link, zones)
            link.unlink()
            outside = root / 'outside'
            outside.write_bytes(raw)
            link.symlink_to(outside)
            with self.assertRaises(ValueError):
                launch.timezone_identity(link, zones)


if __name__ == '__main__':
    unittest.main()
