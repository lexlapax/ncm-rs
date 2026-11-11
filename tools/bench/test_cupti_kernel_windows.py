import json
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from cupti_kernel_windows import main, summarize, union_ns
from test_cupti_trace_boundaries import fixture


def launch(start, end, corr, name='cuLaunchKernel'):
    return (f'DRIVER [ {start}, {end} ] duration {end-start}, "{name}", cbid 307, '
            f'processId 1, threadId 2, correlationId {corr}')


def kernel(start, end, corr, stream=13, graph=0):
    return (f'CONCURRENT_KERNEL [ {start}, {end} ] duration {end-start}, "test_kernel", '
            f'correlationId {corr}, cacheConfigRequested 0, cacheConfigExecuted 0\n'
            f'\tdeviceId 0, contextId 1, streamId {stream}, graphId {graph}, graphNodeId 0, '
            'channelId 1, channelType COMPUTE, priority 0')


def records():
    result = fixture()
    for i in range(36):
        base, corr = i * 30, 1000 + i
        result.extend([launch(base + 13, base + 14, corr), kernel(base + 15, base + 18, corr)])
    return result


def encode(rows):
    return ('\n'.join(rows) + '\n').encode()


def archived_results():
    root = Path(__file__).resolve().parents[2] / 'docs/results/2026-11-10-phase2-cuda-remediation'
    rows = []
    with tarfile.open(root / 'raw/cupti-profiles.tar.gz') as archive:
        for suffix in ('', '-b', '-retained'):
            name = f'codec-diagnostic-block4{suffix}-cupti'
            report = summarize(archive.extractfile(name + '.txt').read(), 'historical-block4')
            expected = json.loads((root / f'profiles/{name}.json').read_text())
            rows.append({'name': name, 'sha256': report['trace_sha256'],
                         'expected_sha256': expected['trace_sha256'],
                         'windows': len(report['windows']),
                         'kernels': sum(len(w['kernels']) for w in report['windows']),
                         'expected_kernels': expected['totals']['CONCURRENT_KERNEL']['count'],
                         'complete': report['activity_attribution_complete']})
    return rows


class KernelWindowTests(unittest.TestCase):
    def test_union_does_not_double_count_or_hide_reversed_intervals(self):
        self.assertEqual(union_ns([(4, 8), (1, 6), (2, 3), (10, 13)]), 10)
        self.assertEqual(union_ns([]), 0)
        self.assertEqual(union_ns([(1, 1)]), 0)
        with self.assertRaises(ValueError):
            union_ns([(9, 8)])

    def test_all_windows_labels_intervals_and_scope(self):
        result = summarize(encode(records()), 'historical-block4')
        self.assertFalse(result['activity_attribution_complete'])
        self.assertEqual(len(result['windows']), 36)
        for row in result['windows']:
            self.assertEqual(row['kernel_sum_ns'], 3)
            self.assertEqual(row['kernel_union_ns'], 3)
            self.assertEqual(row['kernel_hull_ns'], 3)
            self.assertEqual(row['kernel_hull_uncovered_ns'], 0)
            self.assertEqual(row['host_api_summary']['cuLaunchKernel']['sum_ns'], 1)
        self.assertEqual(result['windows'][-1]['assumed_work']['index'], 29)

    def test_shuffled_activity_blocks(self):
        rows = records()
        expected = summarize(encode(rows), 'historical-block4')['windows']
        random.Random(12).shuffle(rows)
        self.assertEqual(summarize(encode(rows), 'historical-block4')['windows'], expected)

    def test_missing_duplicate_and_wrong_launch_correlations(self):
        rows = records()
        cases = [rows[:-1], rows + [rows[-1]], rows[:108] + rows[109:],
                 [r.replace('correlationId 1000, cache', 'correlationId 9999, cache') for r in rows],
                 [r.replace('"cuLaunchKernel"', '"cuGraphLaunch"') for r in rows]]
        for i, case in enumerate(cases):
            with self.subTest(case=i), self.assertRaises(ValueError):
                summarize(encode(case), 'historical-block4')

    def test_invalid_kernel_metadata_and_duration(self):
        rows = records()
        for change in (
            lambda s: s.replace('duration 3, "test_kernel"', 'duration 4, "test_kernel"', 1),
            lambda s: s.replace('CONCURRENT_KERNEL [ 15, 18 ]', 'CONCURRENT_KERNEL [ 18, 15 ]', 1),
            lambda s: s.replace('streamId 13', 'unknownStream 13', 1),
            lambda s: s.replace('graphId 0', 'graphId 1', 1),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                summarize(change(encode(rows).decode()).encode(), 'historical-block4')

    def test_multiple_streams_and_crossing_boundaries(self):
        for replacement in (kernel(15, 18, 1000, stream=7), kernel(15, 21, 1000),
                            kernel(11, 18, 1000), kernel(45, 48, 1000)):
            rows = records()
            rows[109] = replacement
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                summarize(encode(rows), 'historical-block4')
        rows = records()
        rows[108] = launch(11, 14, 1000)
        with self.assertRaises(ValueError):
            summarize(encode(rows), 'historical-block4')

    def test_runtime_launches_and_ambiguous_runtime_records(self):
        for api in ('cudaLaunchKernel_v7000', 'cudaLaunchKernelExC_v11060'):
            rows = records()
            rows[108] = launch(13, 14, 1000, api).replace('DRIVER', 'RUNTIME', 1)
            with self.subTest(api=api):
                result = summarize(encode(rows), 'historical-block4')
                self.assertEqual(result['windows'][0]['host_apis'][0]['domain'], 'RUNTIME')
        rows = records()
        runtime = launch(13, 14, 5000, 'cudaGetLastError_v3020').replace('DRIVER', 'RUNTIME', 1)
        for extra in (runtime.replace('duration 1', 'duration 2'),
                      runtime.replace('threadId 2', 'threadId 3'),
                      runtime.replace('correlationId 5000', 'correlationId 1000'),
                      'RUNTIME malformed'):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                summarize(encode(rows + [extra]), 'historical-block4')

    def test_overlapping_kernels_and_uncovered_hull(self):
        rows = records() + [launch(14, 15, 2000), kernel(16, 19, 2000)]
        row = summarize(encode(rows), 'historical-block4')['windows'][0]
        self.assertEqual((row['kernel_sum_ns'], row['kernel_union_ns'], row['kernel_hull_ns']),
                         (6, 4, 4))
        rows[109] = kernel(15, 16, 1000)
        rows[-1] = kernel(18, 19, 2000)
        row = summarize(encode(rows), 'historical-block4')['windows'][0]
        self.assertEqual(row['kernel_hull_uncovered_ns'], 2)

    def test_synthetic_full_layout_selects_all_36_b1_block_windows(self):
        rows = fixture('codec-bench-b1')
        for ordinal in range(110, 146):
            start = ((ordinal - 1) * 3 + 1) * 10
            corr = 10000 + ordinal
            rows.extend([launch(start + 3, start + 4, corr), kernel(start + 5, start + 8, corr)])
        result = summarize(encode(rows), 'codec-bench-b1')
        self.assertEqual([r['ordinal'] for r in result['windows']], list(range(110, 146)))
        self.assertEqual([r['assumed_work']['index'] for r in result['windows'][6:]], list(range(30)))

    def test_dropped_or_truncated_capture_fails_before_attribution(self):
        for data in (encode(records()) + b'Dropped 1 records', encode(records())[:100]):
            with self.assertRaises(ValueError):
                summarize(data, 'historical-block4')

    def test_empty_windows_fail(self):
        with self.assertRaisesRegex(ValueError, 'empty or unmatched'):
            summarize(encode(fixture()), 'historical-block4')

    def test_copies_overhead_memory_and_unknown_activity_are_not_hidden(self):
        rows = records() + [
            'MEMCPY "HTOD" [ 14, 17 ] duration 3, size 4, correlationId 4000',
            'MEMSET [ 15, 17 ] duration 2, bytes 4',
            'OVERHEAD CUPTI_INSTRUMENTATION [ 15, 19 ] duration 4, THREAD, id 2',
            'MEMORY2 [ 16 ] memoryOperationType ALLOCATION, size 4',
            'OTHER_ACTIVITY [ 17, 19 ] duration 2, unknown detail',
        ]
        row = summarize(encode(rows), 'historical-block4')['windows'][0]
        self.assertEqual({x['kind'] for x in row['unattributed_activity']},
                         {'MEMCPY', 'MEMSET', 'OVERHEAD', 'MEMORY2', 'OTHER_ACTIVITY'})
        self.assertEqual(row['kernel_union_ns'], 3)
        for suffix in ('MEMCPY malformed', 'MEMCPY "HTOD" [ 15, 17 ] duration 3, size 4'):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                summarize(encode(records() + [suffix]), 'historical-block4')

    def test_archived_b4_correlations_in_isolated_process(self):
        # Keep large fixture allocator/RSS growth out of the parent resource tests.
        child = subprocess.run(
            [sys.executable, '-c', 'import json; from test_cupti_kernel_windows import '
             'archived_results; print(json.dumps(archived_results()))'],
            cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=30)
        self.assertEqual(child.returncode, 0, child.stderr)
        rows = json.loads(child.stdout)
        self.assertEqual(len(rows), 3)
        for row in rows:
            with self.subTest(name=row['name']):
                self.assertEqual(row['sha256'], row['expected_sha256'])
                self.assertEqual(row['windows'], 36)
                self.assertEqual(row['kernels'], 972)
                self.assertEqual(row['kernels'], row['expected_kernels'])
                self.assertFalse(row['complete'])

    def test_cli_guards_and_source_bindings(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / 'trace.txt', Path(tmp) / 'out.json'
            source.write_bytes(encode(records()))
            with patch('sys.argv', ['kernel-windows', str(source), str(output),
                                    '--layout', 'historical-block4']):
                main()
                report = json.loads(output.read_text())
                self.assertEqual(set(report['source_sha256']),
                                 {'cupti_kernel_windows.py', 'cupti_trace_boundaries.py'})
                with self.assertRaises(FileExistsError):
                    main()
                output.unlink()
                source.write_text('truncated')
                with self.assertRaises(ValueError):
                    main()
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
