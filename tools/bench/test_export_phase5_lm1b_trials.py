import copy
import json
import unittest

import export_phase5_lm1b_trials as tool


def fixture():
    summary, sources = {'runs': []}, []
    for size, batches in [('small', [1, 4, 16]), ('medium', [1, 4, 16]),
                          ('large', [1]), ('large', [4]), ('large', [8])]:
        manifest = {'size': size, 'batches': batches, 'warmups': 5, 'trials': 30, 'repetitions': 3}
        rows = [{'batch': b, 'runtime': runtime, 'rep': rep, 'seconds': [.1] * 30,
                 'pid': 123, 'signatures': ['private-output']}
                for b in batches for runtime in ('rust', 'python') for rep in range(3)]
        m, r = json.dumps(manifest).encode(), json.dumps({'rows': rows}).encode()
        sources.append((m, r))
        summary['runs'].append({'size': size, 'input_sha256': {'manifest.json': tool.sha(m), 'result.json': tool.sha(r)},
                                'batches': {str(b): {rt: {'process_medians_ms': [100.] * 3}
                                                       for rt in ('rust', 'python')} for b in batches}})
    return summary, sources


class LmTrialsTests(unittest.TestCase):
    def test_coverage_and_privacy(self):
        summary, sources = fixture()
        result = tool.extract(summary, sources)
        self.assertNotIn('private-output', json.dumps(result))
        self.assertNotIn('pid', json.dumps(result))
        self.assertEqual(len(result['sizes']['large']['8']['rust'][0]), 30)

    def test_changed_file_duplicate_coverage_and_wrong_median_fail(self):
        summary, sources = fixture()
        with self.assertRaises(ValueError):
            tool.extract(summary, sources[:-1] + [(sources[-1][0], b'{}')])
        bad = copy.deepcopy(summary)
        bad['runs'][0]['batches']['1']['rust']['process_medians_ms'][0] = 101
        with self.assertRaises(ValueError):
            tool.extract(bad, sources)
        bad = copy.deepcopy(summary)
        bad['runs'][4] = copy.deepcopy(bad['runs'][3])
        with self.assertRaises(ValueError):
            tool.extract(bad, sources[:4] + [sources[3]])


if __name__ == '__main__':
    unittest.main()
