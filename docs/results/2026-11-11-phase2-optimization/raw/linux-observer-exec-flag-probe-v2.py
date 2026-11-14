"""CPU-only handoff stress with failure-only, non-sensitive syscall diagnostics."""
import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, '/home/lexlapax/projects/lexlapax/ncm-rs/tools/bench')
import test_clock_worker_handoff as fixture

needle = 'from clock_process_observation import process_factory\n'
instrument = '''from clock_process_observation import process_factory
import observe_readonly_process as probe_observation
_original_observe = probe_observation.ProcessObservation.observe
def _probe_observe(self, pid):
    try:
        return _original_observe(self, pid)
    except probe_observation.ObservationMismatch as error:
        if 'live child command mismatch' in str(error):
            info={'mismatch':str(error),'accepted_samples':len(self.samples)}
            try:
                data=Path('/proc/%d/stat'%pid).read_text().rsplit(') ',1)[1].split()
                info['fork_no_exec_flag']=bool(int(data[6]) & 0x40)
            except (OSError,ValueError,IndexError):info['fork_no_exec_flag']=None
            try:
                value=Path('/proc/%d/syscall'%pid).read_bytes().split()
                info['syscall_number']=(value[0].decode('ascii') if value else 'empty')
            except OSError as problem:info['syscall_number']=type(problem).__name__
            try:
                info['still_matches_collector']=(Path('/proc/%d/cmdline'%pid).read_bytes()==Path('/proc/self/cmdline').read_bytes())
            except OSError:info['still_matches_collector']=None
            try:
                child=os.stat('/proc/%d/exe'%pid); current=os.stat('/proc/self/exe')
                info['same_collector_image']=(child.st_dev,child.st_ino)==(current.st_dev,current.st_ino)
            except OSError:info['same_collector_image']=None
            print('MISMATCH_PROBE '+json.dumps(info,sort_keys=True),file=sys.stderr,flush=True)
        raise
probe_observation.ProcessObservation.observe=_probe_observe
'''
assert fixture.WORKER.count(needle) == 1
fixture.WORKER = fixture.WORKER.replace(needle, instrument)
results=[]
path=Path('/tmp/ncm-observer-exec-flag-probe-v2.json')
for index in range(1, 101):
    suite=unittest.TestSuite([fixture.ClockHandoffTests('test_default_budget_handoff_preserves_deadline_and_all_eighteen_real_observations')])
    output=io.StringIO()
    result=unittest.TextTestRunner(stream=output,verbosity=1).run(suite)
    results.append({'iteration':index,'success':result.wasSuccessful(),'output':output.getvalue()})
    path.write_text(json.dumps(results,indent=2))
    print('iteration',index,'ok',result.wasSuccessful(),flush=True)
    if not result.wasSuccessful(): break
