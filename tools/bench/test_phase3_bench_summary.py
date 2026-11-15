"""Exercise evidence rejection for the Phase 3 timing summary."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name('phase3_bench_summary.py')

class BenchSummaryTests(unittest.TestCase):
    def test_valid_capture_and_mismatched_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); capture = root / 'capture'; capture.mkdir()
            (root/'source').write_bytes(b'pinned')
            manifest = {'source_sha256': {'source': hashlib.sha256(b'pinned').hexdigest()},
                        'batches':[1,4,32,128], 'warmup_per_process':5,
                        'trials_per_process':30, 'processes_per_runtime_batch':3, 'scope':'synthetic'}
            (capture/'manifest.json').write_text(json.dumps(manifest))
            events=[]; telemetry=[]; i=0
            for batch in manifest['batches']:
                for runtime in ('rust','python'):
                    for rep in range(3):
                        stem=f'{runtime}-b{batch}-r{rep}'; parent=100+i; child=parent+1000; t=i*3
                        report={'batch':batch,'warmups':5,'trials':[0.1]*30,'outputs_sha256_i64le':'same'}
                        if runtime=='rust':report['cuda_pool']={'used_peak':100,'reserved_peak':200}
                        else:report.update(torch_allocated_peak=300,torch_reserved_peak=400)
                        (capture/(stem+'.json')).write_text(json.dumps(report))
                        (capture/(stem+'.time')).write_text('Maximum resident set size (kbytes): 100\nExit status: 0\n')
                        events.extend([{'name':stem,'event':'start','pid':parent,'monotonic':t},
                                       {'name':stem,'event':'end','returncode':0,'monotonic':t+2}])
                        telemetry.append({'monotonic':t+1,'queries':[{'status':0},{'status':0,'stdout':f'{child}, task, 123 MiB\n'}],
                                          'parents':{str(child):parent}})
                        i+=1
            (capture/'process-events.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events))
            (capture/'activity.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in telemetry))
            def run():return subprocess.run([sys.executable,str(SCRIPT),str(capture),'--source-root',str(root)],capture_output=True,text=True)
            result=run();self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)['batches']['1']['rust']['gpu_process_sampled_memory'][0]['peak_mib'],123)
            path=capture/'rust-b1-r0.json'; data=json.loads(path.read_text());data['outputs_sha256_i64le']='wrong';path.write_text(json.dumps(data))
            result=run();self.assertNotEqual(result.returncode,0);self.assertIn('output mismatch',result.stderr)

if __name__ == '__main__': unittest.main()
